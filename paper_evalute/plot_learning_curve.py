"""
plot_learning_curve.py
======================
Parse log .txt dari hasil training SE2 dan plot Learning Curve (Loss, Dice, IoU per epoch).

Usage (di server DGX):
    python paper_evalute/plot_learning_curve.py --log training/training_se2_ct_YYYYMMDD_HHMMSS.txt
    python paper_evalute/plot_learning_curve.py --log training/training_se2_ct_YYYYMMDD_HHMMSS.txt --out curves_ct.png

Output:
    - PNG learning curve (Loss + Dice + IoU per epoch)
"""

import re
import argparse
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ================================================================
# PARSER — extract metrics from log lines
# ================================================================
def parse_log(log_path: str):
    """
    Parse training log and extract per-epoch metrics.
    Expected log line format:
        Ep  42 [CT] Loss 0.1234 | Dice 0.7890 | IoU 0.6543 | Prec 0.8123 | Rec 0.7654 | LR 1.00e-04
    Returns dict of lists: epochs, losses, dices, ious, precs, recs, lrs
    """
    pattern = re.compile(
        r"Ep\s+(\d+)"                          # epoch
        r"(?:\s+\[.*?\])?"                     # optional [DATASET] tag
        r"\s*[|\s]+Loss\s+([\d.]+)"            # loss
        r"\s*\|\s*Dice\s+([\d.]+)"             # dice
        r"\s*\|\s*IoU\s+([\d.]+)"              # iou
        r"(?:\s*\|\s*Prec\s+([\d.]+))?"        # precision (optional)
        r"(?:\s*\|\s*Rec\s+([\d.]+))?"         # recall (optional)
        r"(?:\s*\|\s*LR\s+([\d.e+-]+))?"       # lr (optional)
    )

    epochs, losses, dices, ious, precs, recs, lrs = [], [], [], [], [], [], []
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            clean_line = ansi_escape.sub('', line)
            m = pattern.search(clean_line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                dices.append(float(m.group(3)))
                ious.append(float(m.group(4)))
                precs.append(float(m.group(5)) if m.group(5) else 0.0)
                recs.append(float(m.group(6)) if m.group(6) else 0.0)
                lrs.append(float(m.group(7)) if m.group(7) else 0.0)

    if not epochs:
        raise ValueError(
            f"❌ No matching metric lines found in: {log_path}\n"
            "Make sure the log file is from train_se2_by_dataset.py"
        )

    print(f"✅ Parsed {len(epochs)} epochs from: {log_path}")
    print(f"   First epoch: {epochs[0]} | Last epoch: {epochs[-1]}")
    print(f"   Best IoU : {max(ious):.4f} (Ep {epochs[ious.index(max(ious))]})")
    print(f"   Best Dice: {max(dices):.4f} (Ep {epochs[dices.index(max(dices))]})")
    print(f"   Final LR : {lrs[-1]:.2e}")

    return {
        'epochs': epochs,
        'loss':   losses,
        'dice':   dices,
        'iou':    ious,
        'prec':   precs,
        'rec':    recs,
        'lr':     lrs,
    }


# ================================================================
# PLOT
# ================================================================
def plot_curves(data: dict, out_path: str, title: str = ""):
    """
    Plot 3-panel learning curve:
    - Panel 1: Training Loss
    - Panel 2: IoU + Dice per epoch
    - Panel 3: Precision + Recall per epoch
    """
    epochs = data['epochs']

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    fig.patch.set_facecolor('#0f0f1a')

    colors = {
        'loss':  '#ff6b6b',
        'iou':   '#4ecdc4',
        'dice':  '#ffe66d',
        'prec':  '#a29bfe',
        'rec':   '#fd79a8',
        'fill':  '#2d3436',
        'grid':  '#2c3e50',
        'text':  '#dfe6e9',
    }

    def style_ax(ax, ylabel, ylim=(0, 1)):
        ax.set_facecolor('#1a1a2e')
        ax.set_ylabel(ylabel, color=colors['text'], fontsize=12, labelpad=10)
        ax.tick_params(colors=colors['text'], labelsize=10)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax.set_ylim(*ylim)
        ax.grid(True, color=colors['grid'], linewidth=0.5, linestyle='--', alpha=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors['grid'])

    # ── Panel 1: Loss ──
    ax = axes[0]
    ax.plot(epochs, data['loss'], color=colors['loss'], linewidth=1.8, label='Train Loss')
    ax.fill_between(epochs, data['loss'], alpha=0.15, color=colors['loss'])
    # Mark minimum loss
    min_loss_ep = epochs[data['loss'].index(min(data['loss']))]
    min_loss_val = min(data['loss'])
    ax.axvline(min_loss_ep, color=colors['loss'], linestyle=':', alpha=0.5)
    ax.annotate(f'Min {min_loss_val:.4f}\nEp {min_loss_ep}',
                xy=(min_loss_ep, min_loss_val), color=colors['loss'],
                fontsize=9, ha='left', va='top',
                xytext=(min_loss_ep + max(epochs)*0.01, min_loss_val))
    style_ax(ax, 'Loss', ylim=(max(0, min(data['loss'])*0.9), max(data['loss'])*1.1))
    ax.legend(loc='upper right', facecolor='#1a1a2e', labelcolor=colors['text'], fontsize=10)

    # ── Panel 2: IoU + Dice ──
    ax = axes[1]
    ax.plot(epochs, data['iou'],  color=colors['iou'],  linewidth=2.0, label='IoU (Validation)')
    ax.plot(epochs, data['dice'], color=colors['dice'], linewidth=2.0, label='Dice (Validation)', linestyle='--')
    ax.fill_between(epochs, data['iou'], alpha=0.12, color=colors['iou'])
    # Mark best IoU
    best_iou_ep = epochs[data['iou'].index(max(data['iou']))]
    best_iou_val = max(data['iou'])
    ax.axvline(best_iou_ep, color=colors['iou'], linestyle=':', alpha=0.5)
    ax.annotate(f'Best IoU {best_iou_val:.4f}\nEp {best_iou_ep}',
                xy=(best_iou_ep, best_iou_val), color=colors['iou'],
                fontsize=9, ha='left', va='top',
                xytext=(best_iou_ep + max(epochs)*0.01, best_iou_val))
    # Reference line at 0.80
    ax.axhline(0.80, color='white', linestyle=':', linewidth=0.8, alpha=0.4, label='Target IoU 0.80')
    style_ax(ax, 'IoU / Dice Score')
    ax.legend(loc='lower right', facecolor='#1a1a2e', labelcolor=colors['text'], fontsize=10)

    # ── Panel 3: Precision + Recall ──
    ax = axes[2]
    ax.plot(epochs, data['prec'], color=colors['prec'], linewidth=2.0, label='Precision')
    ax.plot(epochs, data['rec'],  color=colors['rec'],  linewidth=2.0, label='Recall', linestyle='--')
    style_ax(ax, 'Precision / Recall')
    ax.set_xlabel('Epoch', color=colors['text'], fontsize=12, labelpad=10)
    ax.legend(loc='lower right', facecolor='#1a1a2e', labelcolor=colors['text'], fontsize=10)

    # ── Title ──
    dataset_label = title if title else os.path.basename(out_path)
    fig.suptitle(
        f'Learning Curve — Mod-Seg-SE(2)\n{dataset_label}',
        color=colors['text'], fontsize=14, fontweight='bold', y=0.99
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n📊 Learning curve saved → {out_path}")


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning curve from SE2 training log")
    parser.add_argument('--log',   required=True, help="Path to training log .txt file")
    parser.add_argument('--out',   default=None,  help="Output PNG path (default: same dir as log)")
    parser.add_argument('--title', default='',    help="Custom title for the plot")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        raise FileNotFoundError(f"Log file not found: {args.log}")

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(args.log)[0]
        out_path = base + '_learning_curve.png'

    data = parse_log(args.log)
    plot_curves(data, out_path, title=args.title)
