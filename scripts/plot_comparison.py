"""
Comparison plot: mean and median pixel error vs. training-set size
for every model variant tested in the generalization sweeps.

Reads *_results.csv from ../results/sweeps/ and writes a single figure to
../figures/comparison.png.

Usage:
    python plot_comparison.py
"""

import csv
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'sweeps')
FIG_PATH = os.path.join(SCRIPT_DIR, '..', 'figures', 'comparison.png')

# (csv_filename, display_label, color, linestyle)
RUNS = [
    ('generalization_results.csv',                   'Heatmap (baseline)',            '#888888', '--'),
    ('nomask_results.csv',                           'Heatmap, no mask',              '#AAAA33', '--'),
    ('skipconn_results.csv',                         'Heatmap + skip',                '#3366AA', '--'),
    ('nomask_skip_results.csv',                      'Heatmap + skip, no mask',       '#1F77B4', '-'),
    ('regression_results.csv',                       'Direct regression',             '#FF7F0E', '-'),
    ('softargmax_results.csv',                       'Soft-argmax',                   '#2CA02C', '-'),
    ('softargmax_skip_results.csv',                  'Soft-argmax + skip',            '#D62728', '-'),
    ('nomask_skip_no_segmentation_results.csv',      'Heatmap + skip, no seg input',  '#9467BD', ':'),
    ('softargmax_skip_no_segmentation_results.csv',  'Soft-argmax + skip, no seg',    '#8C564B', ':'),
]


def load_csv(path):
    """Return list of (total, mean_px, median_px) sorted by total ascending."""
    rows = []
    with open(path, 'r') as f:
        for r in csv.DictReader(f):
            rows.append((
                int(r['total']),
                float(r['mean_px_err']),
                float(r['median_px_err']),
            ))
    rows.sort(key=lambda x: x[0])
    return rows


def main():
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    for fname, label, color, ls in RUNS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"Skipping (not found): {fname}")
            continue
        rows = load_csv(path)
        ns = [r[0] for r in rows]
        means = [r[1] for r in rows]
        medians = [r[2] for r in rows]

        axes[0].plot(ns, medians, marker='o', linewidth=2, color=color, linestyle=ls, label=label)
        axes[1].plot(ns, means,   marker='o', linewidth=2, color=color, linestyle=ls, label=label)

    for ax, title, ylabel in [
        (axes[0], 'Median pixel error vs. training-set size', 'Median pixel error'),
        (axes[1], 'Mean pixel error vs. training-set size',   'Mean pixel error'),
    ]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Train + val set size (N)')
        ax.set_ylabel(f'{ylabel} (px, on 224×224)')
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    plt.suptitle('Model comparison: generalization as a function of labeled dataset size',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=140, bbox_inches='tight')
    print(f"Saved: {FIG_PATH}")


if __name__ == '__main__':
    main()
