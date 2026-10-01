"""
src/visualize_results.py  (Semester 2 upgrade)
================================================
Generates professional plots from actual result files.
Does NOT hardcode any values.

Saves all figures to results/figures/

Usage:
    python src/visualize_results.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

from config import PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor':   '#16213e',
    'axes.edgecolor':   '#0f3460',
    'axes.labelcolor':  '#e0e0e0',
    'axes.titlecolor':  '#ffffff',
    'xtick.color':      '#c0c0c0',
    'ytick.color':      '#c0c0c0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#0f3460',
    'grid.alpha':       0.4,
    'legend.facecolor': '#16213e',
    'legend.edgecolor': '#0f3460',
    'font.family':      'DejaVu Sans',
})

PALETTE = ['#4cc9f0', '#f72585', '#7209b7', '#3a0ca3', '#4361ee', '#06d6a0']


def save_fig(fig, name):
    os.makedirs(PATHS.FIGURES, exist_ok=True)
    path = PATHS.FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"Saved: {path}")


# ── 1. Model accuracy comparison ───────────────────────────────────────
def plot_model_comparison():
    path = PATHS.FINAL_COMPARISON_CSV
    if not path.exists():
        log.warning(f"Missing: {path}")
        return
    df = pd.read_csv(path)
    if df.empty or 'model' not in df.columns:
        return

    metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (m, color) in enumerate(zip(metrics, PALETTE)):
        if m in df.columns:
            bars = ax.bar(x + i*width, df[m], width, label=m.replace('_', ' ').title(), color=color, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f'{h:.3f}',
                        ha='center', va='bottom', fontsize=8, color='white')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('_', '\n') for m in df['model']], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Accuracy, F1 Macro, F1 Weighted', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='y')
    save_fig(fig, '01_model_comparison')


# ── 2. Precision comparison ─────────────────────────────────────────────
def plot_precision_recall():
    path = PATHS.FINAL_COMPARISON_CSV
    if not path.exists(): return
    df = pd.read_csv(path)
    if df.empty: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, title in zip(axes, ['precision_weighted', 'recall_weighted'], ['Precision (Weighted)', 'Recall (Weighted)']):
        if metric not in df.columns: continue
        bars = ax.barh(df['model'], df[metric], color=PALETTE[:len(df)], alpha=0.85)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f'{w:.3f}',
                    va='center', fontsize=9, color='white')
        ax.set_xlim(0, 1.1)
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='x')
    fig.suptitle('Precision & Recall Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, '02_precision_recall')


# ── 3. Federated round accuracy ─────────────────────────────────────────
def plot_federated_rounds():
    path = PATHS.FED_ROUNDS_CSV
    if not path.exists():
        log.warning(f"Missing: {path}")
        return
    df = pd.read_csv(path)
    if df.empty: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy over rounds
    ax = axes[0]
    ax.plot(df['Round'], df['Global_Accuracy'], 'o-', color=PALETTE[0], linewidth=2, markersize=6, label='Global Accuracy')
    if 'Avg_Client_ValAcc' in df.columns:
        ax.plot(df['Round'], df['Avg_Client_ValAcc'], 's--', color=PALETTE[1], linewidth=1.5, markersize=5, label='Avg Client Val Acc')
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy')
    ax.set_title('Federated Accuracy per Round', fontweight='bold')
    ax.legend()
    ax.grid(True)

    # F1 over rounds
    ax = axes[1]
    if 'Global_F1_Weighted' in df.columns:
        ax.plot(df['Round'], df['Global_F1_Weighted'], 'o-', color=PALETTE[2], linewidth=2, markersize=6, label='F1 Weighted')
    if 'Global_F1_Macro' in df.columns:
        ax.plot(df['Round'], df['Global_F1_Macro'], 's--', color=PALETTE[3], linewidth=1.5, markersize=5, label='F1 Macro')
    ax.set_xlabel('Round')
    ax.set_ylabel('F1 Score')
    ax.set_title('Federated F1 per Round', fontweight='bold')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    save_fig(fig, '03_federated_rounds')
    save_fig(fig, 'federated_convergence')


# ── 4. Client data distribution ─────────────────────────────────────────
def plot_client_distribution():
    path = PATHS.CLIENT_DIST_CSV
    if not path.exists(): return
    df = pd.read_csv(path)

    count_cols = [c for c in df.columns if c.startswith('Count_')]
    if not count_cols: return
    classes = [c.replace('Count_', '') for c in count_cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.8 / len(classes)
    for i, (cls, color) in enumerate(zip(classes, PALETTE)):
        vals = df[f'Count_{cls}'].fillna(0)
        bars = ax.bar(x + i*width, vals, width, label=cls, color=color, alpha=0.85)
    ax.set_xticks(x + width * len(classes) / 2)
    ax.set_xticklabels(df['Client'], fontsize=11)
    ax.set_ylabel('Sample Count')
    ax.set_title('Non-IID Client Data Distribution', fontweight='bold', fontsize=13)
    ax.legend(loc='upper right')
    ax.grid(axis='y')
    save_fig(fig, '04_client_distribution')


# ── 5. Attack class distribution ────────────────────────────────────────
def plot_attack_distribution():
    path = PATHS.CLIENT_DIST_CSV
    if not path.exists(): return
    df = pd.read_csv(path)

    count_cols = [c for c in df.columns if c.startswith('Count_') and 'Normal' not in c]
    if not count_cols: return
    classes = [c.replace('Count_', '') for c in count_cols]
    totals  = [df[c].sum() for c in count_cols]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(classes, totals, color=PALETTE[:len(classes)], alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 20, f'{int(h):,}',
                ha='center', va='bottom', fontsize=10, color='white')
    ax.set_ylabel('Total Samples')
    ax.set_title('Attack Class Distribution (All Clients)', fontweight='bold', fontsize=13)
    ax.grid(axis='y')
    save_fig(fig, '05_attack_distribution')


# ── 6. Federated loss per round ─────────────────────────────────────────
def plot_federated_loss():
    path = PATHS.FED_ROUNDS_CSV
    if not path.exists(): return
    df = pd.read_csv(path)
    if 'Global_Loss' not in df.columns: return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Round'], df['Global_Loss'], 'o-', color=PALETTE[4], linewidth=2, markersize=6, label='Global Test Loss')
    if 'Avg_Train_Loss' in df.columns:
        ax.plot(df['Round'], df['Avg_Train_Loss'], 's--', color=PALETTE[5], linewidth=1.5, markersize=5, label='Avg Client Train Loss')
    ax.set_xlabel('Round')
    ax.set_ylabel('Loss')
    ax.set_title('Federated Training Loss per Round', fontweight='bold')
    ax.legend()
    ax.grid(True)
    save_fig(fig, '06_federated_loss')


# ── 7. Centralized vs Federated summary ─────────────────────────────────
def plot_centralized_vs_federated():
    path = PATHS.FINAL_COMPARISON_CSV
    if not path.exists(): return
    df = pd.read_csv(path)
    if df.empty: return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    bars = ax.bar(df['model'], df['accuracy'], color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f'{h:.4f}',
                ha='center', va='bottom', fontsize=10, color='white', fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Centralized vs Federated IDS — Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_xticklabels([m.replace('_', '\n') for m in df['model']], fontsize=9)
    ax.grid(axis='y')
    save_fig(fig, '07_centralized_vs_federated')


def run_all():
    """Generate all plots."""
    PATHS.ensure_dirs()
    os.makedirs(PATHS.FIGURES, exist_ok=True)
    log.info("Generating all visualizations...")
    plot_model_comparison()
    plot_precision_recall()
    plot_federated_rounds()
    plot_client_distribution()
    plot_attack_distribution()
    plot_federated_loss()
    plot_centralized_vs_federated()
    log.info(f"All figures saved to: {PATHS.FIGURES}")


if __name__ == "__main__":
    run_all()