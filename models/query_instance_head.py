"""Coarse query instance head for WireCR-HQInstSAM."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

__all__ = ["QueryInstanceHead"]


class MLP(nn.Module):
    """Small MLP used for box and mask projections."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("MLP requires at least one layer.")

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers = []
        for idx in range(num_layers):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < num_layers - 1:
                layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class QueryDecoderLayer(nn.Module):
    """A lightweight DETR-style decoder block."""

    def __init__(self, hidden_dim: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        self_attended, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = self.norm1(queries + self.dropout(self_attended))

        cross_attended, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        queries = self.norm2(queries + self.dropout(cross_attended))

        queries = self.norm3(queries + self.dropout(self.ffn(queries)))
        return queries


class QueryInstanceHead(nn.Module):
    """Generate coarse instance logits, boxes, and masks from stride-4 features."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_queries: int = 64,
        decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_feature_levels: int = 4,
        position_temperature: float = 10000.0,
    ) -> None:
        super().__init__()
        if decoder_layers < 1:
            raise ValueError("decoder_layers must be at least 1.")
        if num_queries < 1:
            raise ValueError("num_queries must be at least 1.")

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.decoder_layers = decoder_layers
        self.num_feature_levels = num_feature_levels
        self.position_temperature = float(position_temperature)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Parameter(torch.zeros(num_feature_levels, hidden_dim))
        self.register_buffer(
            "reference_boxes",
            self._build_reference_boxes(num_queries),
            persistent=False,
        )
        self.decoder = nn.ModuleList(
            [
                QueryDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.box_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=3)

        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.normal_(self.level_embed, std=0.02)
        final_box_layer = self.box_embed.layers[-1]
        if isinstance(final_box_layer, nn.Linear):
            nn.init.zeros_(final_box_layer.weight)
            nn.init.zeros_(final_box_layer.bias)

    @staticmethod
    def _inverse_sigmoid(value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        value = value.clamp(min=eps, max=1.0 - eps)
        return torch.log(value / (1.0 - value))

    @staticmethod
    def _build_reference_boxes(num_queries: int) -> torch.Tensor:
        cols = max(int(torch.ceil(torch.sqrt(torch.tensor(float(num_queries)))).item()), 1)
        rows = max((num_queries + cols - 1) // cols, 1)
        step_x = 1.0 / float(cols)
        step_y = 1.0 / float(rows)
        box_width = step_x * 0.8
        box_height = step_y * 0.8

        boxes = []
        for index in range(num_queries):
            row = index // cols
            col = index % cols
            center_x = (col + 0.5) * step_x
            center_y = (row + 0.5) * step_y
            x0 = max(center_x - box_width * 0.5, 0.0)
            y0 = max(center_y - box_height * 0.5, 0.0)
            x1 = min(center_x + box_width * 0.5, 1.0)
            y1 = min(center_y + box_height * 0.5, 1.0)
            boxes.append((x0, y0, x1, y1))
        return torch.tensor(boxes, dtype=torch.float32)

    def _build_spatial_position_encoding(
        self,
        *,
        height: int,
        width: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if channels < 4:
            raise ValueError(f"hidden_dim must be at least 4 for 2D position encoding, got {channels}.")

        y_coords = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x_coords = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        band_count = max(channels // 4, 1)
        dim_range = torch.arange(band_count, device=device, dtype=dtype)
        dim_t = self.position_temperature ** (dim_range / max(band_count - 1, 1))

        pos_x = (2.0 * torch.pi * grid_x.unsqueeze(-1)) / dim_t
        pos_y = (2.0 * torch.pi * grid_y.unsqueeze(-1)) / dim_t
        pos_x = torch.cat((pos_x.sin(), pos_x.cos()), dim=-1)
        pos_y = torch.cat((pos_y.sin(), pos_y.cos()), dim=-1)
        position = torch.cat((pos_y, pos_x), dim=-1)

        if position.shape[-1] < channels:
            pad = channels - position.shape[-1]
            position = torch.cat((position, position.new_zeros((height, width, pad))), dim=-1)
        elif position.shape[-1] > channels:
            position = position[..., :channels]
        return position.view(1, height * width, channels)

    def _build_memory(self, multi_scale_memory: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(multi_scale_memory) != self.num_feature_levels:
            raise ValueError(
                f"Expected {self.num_feature_levels} feature levels, got {len(multi_scale_memory)}."
            )

        tokens = []
        for level, feature in enumerate(multi_scale_memory):
            if feature.dim() != 4:
                raise ValueError(f"Expected BCHW feature maps, got shape {tuple(feature.shape)}.")
            batch_size, channels, height, width = feature.shape
            feature_tokens = feature.flatten(2).transpose(1, 2).contiguous()
            position_tokens = self._build_spatial_position_encoding(
                height=height,
                width=width,
                channels=channels,
                device=feature.device,
                dtype=feature.dtype,
            )
            feature_tokens = feature_tokens + self.level_embed[level].view(1, 1, channels) + position_tokens
            tokens.append(feature_tokens)
        return torch.cat(tokens, dim=1)

    @staticmethod
    def _to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        x0, y0, x1, y1 = boxes.unbind(dim=-1)
        left = torch.minimum(x0, x1)
        top = torch.minimum(y0, y1)
        right = torch.maximum(x0, x1)
        bottom = torch.maximum(y0, y1)
        return torch.stack((left, top, right, bottom), dim=-1)

    def _predict(
        self,
        queries: torch.Tensor,
        mask_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pred_logits = self.class_embed(queries)
        reference_boxes = self.reference_boxes.to(device=queries.device, dtype=queries.dtype).unsqueeze(0)
        pred_boxes = self._to_xyxy((self.box_embed(queries) + self._inverse_sigmoid(reference_boxes)).sigmoid())
        pred_mask_embeddings = self.mask_embed(queries)
        pred_masks = torch.einsum("bqc,bchw->bqhw", pred_mask_embeddings, mask_features)
        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
        }

    def forward(
        self,
        mask_features: torch.Tensor,
        multi_scale_memory: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor | list[Dict[str, torch.Tensor]]]:
        if mask_features.dim() != 4:
            raise ValueError(f"Expected mask_features in BCHW format, got {tuple(mask_features.shape)}.")

        batch_size = mask_features.shape[0]
        memory = self._build_memory(multi_scale_memory)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

        layer_outputs = []
        for layer in self.decoder:
            queries = layer(queries, memory)
            layer_outputs.append(self._predict(queries, mask_features))

        return {
            **layer_outputs[-1],
            "aux_outputs": layer_outputs[:-1],
        }
