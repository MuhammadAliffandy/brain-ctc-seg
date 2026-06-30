import os
import pandas as pd
from tabulate import tabulate

def print_clinical_summary():
    print("\n" + "="*85)
    print("📊 CLINICAL SUMMARY REPORT: CT NON-CONTRAST, CTC, STROKE, & HEMORRHAGE".center(85))
    print("="*85 + "\n")

    # Define the 4 sources
    sources = [
        ("CT Non-Contrast (Private)", os.path.expanduser("~/Clara/comparison_eval_ct.csv")),
        ("CT With Contrast (Private)", os.path.expanduser("~/Clara/comparison_eval_ctc.csv")),
        ("Stroke (Public Kaggle)", os.path.join(os.path.dirname(__file__), "..", "public_dataset", "public_intra_eval_metrics.csv")),
        ("Hemorrhage (Public Kaggle)", os.path.join(os.path.dirname(__file__), "..", "public_dataset", "public_intra_hemorrhage_eval_metrics.csv"))
    ]

    all_data = []

    for dataset_name, csv_path in sources:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.insert(0, 'Condition', dataset_name)
                all_data.append(df)
            except Exception as e:
                print(f"⚠️ Error reading {dataset_name}: {e}")
        else:
            print(f"⚠️ Data untuk '{dataset_name}' belum tersedia. (File tidak ditemukan: {csv_path})")

    if not all_data:
        print("\n❌ Tidak ada satupun file evaluasi yang ditemukan.")
        print("Pastikan Anda sudah menjalankan script evaluasi untuk CT, CTC, Stroke, dan Hemorrhage.")
        return

    # Combine all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort logically: First by Condition, then by F1 (Dice) descending
    condition_order = ["CT Non-Contrast (Private)", "CT With Contrast (Private)", "Stroke (Public Kaggle)", "Hemorrhage (Public Kaggle)"]
    combined_df['Condition'] = pd.Categorical(combined_df['Condition'], categories=condition_order, ordered=True)
    
    dice_col = 'F1 (Dice)' if 'F1 (Dice)' in combined_df.columns else 'Dice'
    if dice_col not in combined_df.columns and 'F1_Mean' in combined_df.columns:
        dice_col = 'F1_Mean'
        
    try:
        combined_df = combined_df.sort_values(by=['Condition', dice_col], ascending=[True, False])
    except:
        combined_df = combined_df.sort_values(by=['Condition'])

    # Format floats to 4 decimal places
    for col in combined_df.columns:
        if combined_df[col].dtype == 'float64':
            combined_df[col] = combined_df[col].map(lambda x: f"{x:.4f}")

    # Print using tabulate
    print(tabulate(combined_df, headers='keys', tablefmt='grid', showindex=False))
    print("\n💡 Catatan untuk Klien: Evaluasi telah dikelompokkan sesuai jenis kondisi medis.")
    print("Pastikan nilai Akurasi sudah tinggi (>0.90) yang menandakan bug kalkulasi True Negatives telah diperbaiki.\n")

if __name__ == "__main__":
    print_clinical_summary()
