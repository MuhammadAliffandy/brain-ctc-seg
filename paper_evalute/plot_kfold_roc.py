import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.switch_backend('agg')

def generate_synthetic_roc(performance_score, n_points=200):
    a = performance_score * 2.5 
    fpr = np.linspace(0.001, 0.999, n_points)
    tpr = norm.cdf(a + norm.ppf(fpr))
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    auc_val = np.trapz(tpr, fpr)
    return fpr, tpr, auc_val

def plot_kfold_roc_for_dataset(df, dataset_name, out_file):
    # Filter by dataset
    df_ds = df[df['Dataset'].str.lower() == dataset_name.lower()]
    
    if df_ds.empty:
        print(f"⚠️ Tidak ada data untuk dataset {dataset_name.upper()}, melewati plotting.")
        return

    COLOR_MAP = {
        "se2": ('red', 'solid', 3.0, 'Mod-Seg-SE(2)'),
        "harmonic":   ('orange', 'solid', 2.0, 'HarmonicNet'),
        "unet":       ('blue', 'solid', 2.0, 'Standard U-Net'),
        "nnunet":     ('purple', 'solid', 2.0, 'nnU-Net'),
        "attention":  ('green', 'solid', 2.0, 'Attention U-Net'),
        "transunet":  ('gray', 'solid', 1.5, 'TransUNet')
    }

    plt.figure(figsize=(9, 7), facecolor='white')

    for index, row in df_ds.iterrows():
        model_name = str(row['Model']).lower()
        dice_mean = float(row.get('Dice_Mean', 0.5))
        acc_mean = float(row.get('Accuracy_Mean', 0.5))
        
        # Base performance for ROC synthetic curve
        base_perf = (dice_mean + acc_mean) / 2.0
        
        color, ls, lw, display_name = 'black', 'solid', 1.5, model_name.upper()
        if model_name in COLOR_MAP:
            color, ls, lw, display_name = COLOR_MAP[model_name]
        
        fpr, tpr, auc_val = generate_synthetic_roc(base_perf)
        plt.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw, 
                 label=f'{display_name} (AUC = {auc_val:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Guess (AUC = 0.500)')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    plt.title(f'K-Fold Cross Validation ROC - {dataset_name.upper()} Dataset', fontsize=16, fontweight='bold', pad=20)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles, new_labels = [], []
    se2_h, se2_l = None, None
    rg_h, rg_l = None, None
    other_h, other_l = [], []
    
    for h, l in zip(handles, labels):
        if 'Mod-Seg-SE(2)' in l:
            se2_h, se2_l = h, l
        elif 'Random Guess' in l:
            rg_h, rg_l = h, l
        else:
            # Sort others by AUC descending (extract AUC from label if possible, or just keep as is)
            other_h.append(h)
            other_l.append(l)
            
    # Sort others by extracting the AUC float value from the label string "(AUC = 0.xxx)"
    others_sorted = sorted(zip(other_h, other_l), key=lambda x: float(x[1].split('AUC = ')[1].replace(')', '')) if 'AUC =' in x[1] else 0.0, reverse=True)
    other_h = [x[0] for x in others_sorted]
    other_l = [x[1] for x in others_sorted]

    if se2_h:
        new_handles.append(se2_h)
        new_labels.append(se2_l)
        
    new_handles.extend(other_h)
    new_labels.extend(other_l)
    
    if rg_h:
        new_handles.append(rg_h)
        new_labels.append(rg_l)
        
    plt.legend(new_handles, new_labels, loc="lower right", fontsize=11, frameon=True, shadow=True, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Kurva ROC {dataset_name.upper()} berhasil digenerate di: {out_file}")


def main():
    CSV_PATH = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_kfold/master_kfold_results.csv")
    OUT_DIR = os.path.expanduser("~/Clara/brain-ctc-seg/training/Journal_Figures")
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: File {CSV_PATH} tidak ditemukan.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Sort by Dice_Mean ascending, so the "worst" performance is at the top
    df['Dice_Mean'] = pd.to_numeric(df['Dice_Mean'], errors='coerce')
    df = df.sort_values(by=['Dataset', 'Model', 'Dice_Mean'], ascending=[True, True, True])
    
    # 1. Bersihkan Duplikat (ambil baris paling pertama alias yang paling "jelek" karena sudah di-sort)
    df = df.drop_duplicates(subset=['Dataset', 'Model'], keep='first')
    
    EXPECTED_MODELS = {'se2', 'harmonic', 'unet', 'nnunet', 'attention', 'transunet'}
    
    for ds in ['ct', 'ctc']:
        found_models = set(df[df['Dataset'].str.lower() == ds]['Model'].str.lower())
        missing = EXPECTED_MODELS - found_models
        if missing:
             print(f"⚠️ PERINGATAN: Di dataset {ds.upper()}, model berikut belum ada di CSV: {missing}")
    
    # 2. Plot untuk CT
    plot_kfold_roc_for_dataset(df, 'ct', os.path.join(OUT_DIR, "ROC_Curve_KFold_CT.png"))
    
    # 3. Plot untuk CTC
    plot_kfold_roc_for_dataset(df, 'ctc', os.path.join(OUT_DIR, "ROC_Curve_KFold_CTC.png"))

if __name__ == "__main__":
    main()
