#!/usr/bin/env python3
"""
Aggregate and analyze results from SAM+CRNet adapter grid search experiments.

This script parses log files from grid_search.sh and generates comparison tables
and visualizations.
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict


def parse_log_file(log_path):
    """Parse a single log file and extract metrics."""
    results = {
        'adapter_size': None,
        'compression_ratio': None,
        'prompt_strategy': None,
        'epochs': None,
        'best_iou': None,
        'best_dice': None,
        'final_iou': None,
        'final_dice': None,
        'training_time': None,
    }

    with open(log_path, 'r') as f:
        content = f.read()

        # Extract adapter configuration
        match = re.search(r'Adapter Config: size=(\w+), compression=1/(\d+)', content)
        if match:
            results['adapter_size'] = match.group(1)
            results['compression_ratio'] = int(match.group(2))

        # Extract prompt strategy
        match = re.search(r'Prompt strategy: (\w+)', content)
        if match:
            results['prompt_strategy'] = match.group(1)

        # Extract epochs
        match = re.search(r'Epochs: (\d+)', content)
        if match:
            results['epochs'] = int(match.group(1))

        # Extract best IoU
        match = re.search(r'Best IoU: ([\d.]+)', content)
        if match:
            results['best_iou'] = float(match.group(1))

        # Extract final test IoU
        match = re.search(r'Mean IoU: ([\d.]+)', content)
        if match:
            results['final_iou'] = float(match.group(1))

        # Extract final test Dice
        match = re.search(r'Mean Dice: ([\d.]+)', content)
        if match:
            results['final_dice'] = float(match.group(1))

    return results


def aggregate_results(log_dir):
    """Aggregate results from all log files in a directory."""
    log_dir = Path(log_dir)

    all_results = []

    for log_file in sorted(log_dir.glob('*.log')):
        print(f"Parsing: {log_file.name}")
        results = parse_log_file(log_file)
        results['log_file'] = log_file.name
        all_results.append(results)

    return all_results


def generate_summary_table(all_results):
    """Generate summary table grouped by adapter size and compression ratio."""
    # Group by configuration
    groups = defaultdict(list)

    for r in all_results:
        key = (r['adapter_size'], r['compression_ratio'])
        groups[key].append(r)

    # Compute averages
    summary = []
    for (size, ratio), group in groups.items():
        ious = [r['final_iou'] for r in group if r['final_iou'] is not None]
        dices = [r['final_dice'] for r in group if r['final_dice'] is not None]

        if ious:
            summary.append({
                'adapter_size': size,
                'compression_ratio': ratio,
                'mean_iou': sum(ious) / len(ious),
                'mean_dice': sum(dices) / len(dices) if dices else None,
                'num_experiments': len(group),
            })

    # Sort by IoU
    summary.sort(key=lambda x: -x['mean_iou'])

    return summary


def print_summary_table(summary):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("Summary Table (grouped by adapter size and compression ratio)")
    print("=" * 80)
    print(f"{'Adapter Size':<15} {'Compression':<12} {'Mean IoU':<12} {'Mean Dice':<12} {'Count':<8}")
    print("-" * 80)

    for s in summary:
        print(f"{s['adapter_size']:<15} {s['compression_ratio']:<12} "
              f"{s['mean_iou']:<12.4f} {s['mean_dice']:<12.4f} {s['num_experiments']:<8}")

    print("=" * 80)

    # Find best configuration
    if summary:
        best = max(summary, key=lambda x: x['mean_iou'])
        print(f"\nBest configuration:")
        print(f"  Adapter size: {best['adapter_size']}")
        print(f"  Compression ratio: 1/{best['compression_ratio']}")
        print(f"  Mean IoU: {best['mean_iou']:.4f}")
        print(f"  Mean Dice: {best['mean_dice']:.4f}")


def save_results_csv(all_results, output_path):
    """Save results to CSV file."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'adapter_size', 'compression_ratio', 'prompt_strategy',
            'final_iou', 'final_dice', 'best_iou', 'log_file'
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                'adapter_size': r['adapter_size'],
                'compression_ratio': r['compression_ratio'],
                'prompt_strategy': r['prompt_strategy'],
                'final_iou': r['final_iou'],
                'final_dice': r['final_dice'],
                'best_iou': r['best_iou'],
                'log_file': r['log_file'],
            })

    print(f"\nResults saved to: {output_path}")


def plot_results(all_results, output_dir):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nNote: Install matplotlib to generate plots")
        print("  pip install matplotlib")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    sizes = ['small', 'medium', 'large']
    ratios = [4, 8, 16, 32, 64]

    # Create matrix of IoU values
    iou_matrix = {}
    for size in sizes:
        iou_matrix[size] = {}
        for ratio in ratios:
            values = [r['final_iou'] for r in all_results
                     if r['adapter_size'] == size and r['compression_ratio'] == ratio
                     and r['final_iou'] is not None]
            iou_matrix[size][ratio] = sum(values) / len(values) if values else None

    # Plot 1: IoU vs Compression Ratio
    fig, ax = plt.subplots(figsize=(10, 6))

    for size in sizes:
        x = [r for r in ratios if iou_matrix[size][r] is not None]
        y = [iou_matrix[size][r] for r in x]
        ax.plot(x, y, marker='o', label=f'{size.capitalize()}')

    ax.set_xlabel('Compression Ratio (1/x)')
    ax.set_ylabel('Mean IoU')
    ax.set_title('SAM+CRNet Adapter: IoU vs Compression Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'iou_vs_compression.png', dpi=150)
    print(f"Plot saved: {output_dir / 'iou_vs_compression.png'}")

    # Plot 2: Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    for size in sizes:
        row = [iou_matrix[size][r] if iou_matrix[size][r] is not None else 0
               for r in ratios]
        data.append(row)

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(ratios)))
    ax.set_yticks(range(len(sizes)))
    ax.set_xticklabels([f'1/{r}' for r in ratios])
    ax.set_yticklabels([s.capitalize() for s in sizes])
    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('Adapter Size')

    # Add text annotations
    for i, size in enumerate(sizes):
        for j, ratio in enumerate(ratios):
            value = iou_matrix[size][ratio]
            if value is not None:
                text = ax.text(j, i, f'{value:.3f}',
                              ha="center", va="center", color="black", fontsize=10)

    ax.set_title('SAM+CRNet Adapter: IoU Heatmap')
    plt.colorbar(im, ax=ax, label='IoU')
    plt.tight_layout()
    plt.savefig(output_dir / 'iou_heatmap.png', dpi=150)
    print(f"Plot saved: {output_dir / 'iou_heatmap.png'}")

    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate SAM+CRNet adapter experiment results')
    parser.add_argument('--log-dir', type=str, default='./logs/grid_search',
                       help='Directory containing log files')
    parser.add_argument('--output', type=str, default='./logs/grid_search/results.csv',
                       help='Output CSV file path')
    parser.add_argument('--plot-dir', type=str, default='./logs/grid_search/plots',
                       help='Directory to save plots')

    args = parser.parse_args()

    print("=" * 60)
    print("SAM+CRNet Adapter Results Aggregator")
    print("=" * 60)

    # Aggregate results
    all_results = aggregate_results(args.log_dir)

    if not all_results:
        print(f"\nNo log files found in {args.log_dir}")
        return

    print(f"\nParsed {len(all_results)} log files")

    # Generate and print summary table
    summary = generate_summary_table(all_results)
    print_summary_table(summary)

    # Save to CSV
    save_results_csv(all_results, args.output)

    # Generate plots
    plot_results(all_results, args.plot_dir)


if __name__ == '__main__':
    main()
