"""
plot_ablation_curves.py
=======================
Reads the training logs from `training/logs_ablation/` and generates
a comparative learning curve figure (Val Dice & Val IoU) for the ablation study.
"""

import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file(filepath):
    epochs = []
    dice_scores = []
    iou_scores = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # Example format: "➜ Val IoU: 0.8543 | Val Dice: 0.9213 | Train Loss: 0.0412"
        # Since I used emojis, let's parse robustly
        if "Val IoU:" in line and "Val Dice:" in line:
            try:
                parts = line.split('|')
                iou = float(parts[0].split(':')[1].strip())
                dice = float(parts[1].split(':')[1].strip())
                
                epochs.append(len(epochs) + 1)
                iou_scores.append(iou)
                dice_scores.append(dice)
            except Exception as e:
                pass
                
    return pd.DataFrame({
        'Epoch': epochs,
        'Val Dice': dice_scores,
        'Val IoU': iou_scores
    })

if __name__ == '__main__':
    LOG_DIR = os.path.expanduser("~/Clara/brain-ctc-seg/training/logs_ablation")
    OUT_DIR = os.path.dirname(__file__)
    
    log_files = glob.glob(os.path.join(LOG_DIR, "*.txt"))
    if not log_files:
        print("❌ No logs found in", LOG_DIR)
        exit(1)
        
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    colors = sns.color_palette("husl", len(log_files))
    
    for i, log_path in enumerate(sorted(log_files)):
        filename = os.path.basename(log_path)
        # Extract variant name (e.g. A1_ct_2024... -> A1)
        variant = filename.split('_')[0] 
        
        df = parse_log_file(log_path)
        if len(df) > 0:
            plt.plot(df['Epoch'], df['Val Dice'], label=f'Variant {variant}', linewidth=2, color=colors[i])
            
    plt.title("Ablation Study: Validation Dice Convergence", fontsize=16, fontweight='bold')
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Validation Dice Score", fontsize=14)
    plt.legend(title="Model Variants", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(OUT_DIR, "ablation_learning_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 Saved learning curve figure to {save_path}")
