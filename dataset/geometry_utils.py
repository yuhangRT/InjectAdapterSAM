"""Geometry helpers for S01 ISAT-to-COCO conversion.

This module is intentionally dependency-light. It provides polygon
normalization, rasterization, axis-aligned and oriented bounding boxes,
and principal-axis metadata without relying on NumPy/OpenCV.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from math import atan2, cos, degrees, hypot, isclose, sin, sqrt
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

Point = tuple[float, float]


def load_isat_json(path: str | Path) -> dict[str, Any]:
    annotation_path = Path(path).expanduser().resolve()
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected JSON mapping at {annotation_path}, got {type(loaded).__name__}")
    return loaded


def normalize_polygon(segmentation: object) -> list[Point]:
    if not segmentation:
        return []
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        first = segmentation[0]
        if len(first) == 2 and all(isinstance(value, (int, float)) for value in first):
            return [(float(x), float(y)) for x, y in segmentation]
        if len(segmentation) == 1:
            return normalize_polygon(segmentation[0])
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], tuple):
        if len(segmentation[0]) == 2:
            return [(float(x), float(y)) for x, y in segmentation]
    if isinstance(segmentation, list) and len(segmentation) % 2 == 0:
        points = []
        for index in range(0, len(segmentation), 2):
            points.append((float(segmentation[index]), float(segmentation[index + 1])))
        return points
    raise ValueError(f"Unsupported polygon format: {segmentation!r}")


def flatten_polygon(points: Sequence[Point]) -> list[float]:
    flattened: list[float] = []
    for x_coord, y_coord in points:
        flattened.extend([float(x_coord), float(y_coord)])
    return flattened


def polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _cross(o: Point, a: Point, b: Point) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _lexicographic_sort(points: Sequence[Point]) -> list[Point]:
    return sorted((float(x), float(y)) for x, y in points)


def convex_hull(points: Sequence[Point]) -> list[Point]:
    ordered = _lexicographic_sort(points)
    if len(ordered) <= 1:
        return list(ordered)
    lower: list[Point] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[Point] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def bbox_xyxy(points: Sequence[Point]) -> list[float]:
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def bbox_xyxy_from_mask(mask: list[list[int]] | list[list[bool]]) -> list[float]:
    min_x = min_y = None
    max_x = max_y = None
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value:
                if min_x is None or x < min_x:
                    min_x = x
                if min_y is None or y < min_y:
                    min_y = y
                if max_x is None or x > max_x:
                    max_x = x
                if max_y is None or y > max_y:
                    max_y = y
    if min_x is None or min_y is None or max_x is None or max_y is None:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(min_x), float(min_y), float(max_x + 1), float(max_y + 1)]


def bbox_xywh_from_mask(mask: list[list[int]] | list[list[bool]]) -> list[float]:
    bbox = bbox_xyxy_from_mask(mask)
    x1, y1, x2, y2 = bbox
    return [float(x1), float(y1), float(max(x2 - x1, 0.0)), float(max(y2 - y1, 0.0))]


def polygon_to_mask(height: int, width: int, polygon: Sequence[Point]) -> list[list[int]]:
    mask = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(mask)
    if len(polygon) >= 3:
        draw.polygon([(float(x), float(y)) for x, y in polygon], fill=1, outline=1)
    pixels = list(mask.getdata())
    rows: list[list[int]] = []
    row_width = int(width)
    for y in range(int(height)):
        start = y * row_width
        rows.append([1 if pixels[start + x] else 0 for x in range(row_width)])
    return rows


def multi_polygon_to_mask(height: int, width: int, polygons: Sequence[Sequence[Point]]) -> list[list[int]]:
    mask_image = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(mask_image)
    for polygon in polygons:
        if len(polygon) >= 3:
            draw.polygon([(float(x), float(y)) for x, y in polygon], fill=1, outline=1)
    pixels = list(mask_image.getdata())
    rows: list[list[int]] = []
    row_width = int(width)
    for y in range(int(height)):
        start = y * row_width
        rows.append([1 if pixels[start + x] else 0 for x in range(row_width)])
    return rows


def _mask_to_points(mask: list[list[int]] | list[list[bool]]) -> list[Point]:
    points: list[Point] = []
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value:
                points.append((float(x), float(y)))
    return points


def _oriented_box_from_points(points: Sequence[Point]) -> list[float]:
    if not points:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    if len(points) == 1:
        x, y = points[0]
        return [float(x), float(y), 0.0, 0.0, 0.0]
    if len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        width = hypot(x2 - x1, y2 - y1)
        angle = degrees(atan2(y2 - y1, x2 - x1))
        return [float(center_x), float(center_y), float(width), 0.0, float(angle)]

    hull = convex_hull(points)
    if len(hull) <= 2:
        return _oriented_box_from_points(hull)

    best: tuple[float, float, float, float, float, float] | None = None
    for index in range(len(hull)):
        x1, y1 = hull[index]
        x2, y2 = hull[(index + 1) % len(hull)]
        angle = -atan2(y2 - y1, x2 - x1)
        cos_a = cos(angle)
        sin_a = sin(angle)
        rotated = [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in hull]
        xs = [point[0] for point in rotated]
        ys = [point[1] for point in rotated]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        if best is None or area < best[0]:
            center_rot_x = min_x + width * 0.5
            center_rot_y = min_y + height * 0.5
            inv_cos = cos(-angle)
            inv_sin = sin(-angle)
            center_x = center_rot_x * inv_cos - center_rot_y * inv_sin
            center_y = center_rot_x * inv_sin + center_rot_y * inv_cos
            best = (area, center_x, center_y, width, height, degrees(-angle))
    assert best is not None
    _, center_x, center_y, width, height, angle_deg = best
    return [float(center_x), float(center_y), float(width), float(height), float(angle_deg)]


def oriented_box_from_points(points: Sequence[Point]) -> list[float]:
    return _oriented_box_from_points(list(points))


def oriented_box_from_mask(mask: list[list[int]] | list[list[bool]]) -> list[float]:
    return _oriented_box_from_points(_mask_to_points(mask))


def principal_axis_from_points(points: Sequence[Point]) -> list[float]:
    pts = list(points)
    if len(pts) <= 1:
        return [1.0, 0.0]
    mean_x = sum(point[0] for point in pts) / len(pts)
    mean_y = sum(point[1] for point in pts) / len(pts)
    centered = [(x - mean_x, y - mean_y) for x, y in pts]
    cov_xx = sum(x * x for x, _ in centered) / len(centered)
    cov_xy = sum(x * y for x, y in centered) / len(centered)
    cov_yy = sum(y * y for _, y in centered) / len(centered)
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    radicand = max(trace * trace * 0.25 - det, 0.0)
    lambda_1 = trace * 0.5 + sqrt(radicand)
    vx = cov_xy
    vy = lambda_1 - cov_xx
    norm = hypot(vx, vy)
    if norm <= 1e-12:
        vx, vy = 1.0, 0.0
        norm = 1.0
    return [float(vx / norm), float(vy / norm)]


def principal_axis_from_mask(mask: list[list[int]] | list[list[bool]]) -> list[float]:
    return principal_axis_from_points(_mask_to_points(mask))


def polygon_mask_summary(height: int, width: int, polygons: Sequence[Sequence[Point]]) -> dict[str, Any]:
    mask = multi_polygon_to_mask(height, width, polygons)
    return {
        "mask": mask,
        "bbox": bbox_xywh_from_mask(mask),
        "oriented_box": oriented_box_from_mask(mask),
        "principal_axis": principal_axis_from_mask(mask),
        "area": float(sum(sum(row) for row in mask)),
    }


def polygon_raster_area(height: int, width: int, polygons: Sequence[Sequence[Point]]) -> float:
    mask = multi_polygon_to_mask(height, width, polygons)
    return float(sum(sum(row) for row in mask))


def safe_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]
