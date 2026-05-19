import os
import pandas as pd
from tabulate import tabulate

def generate_ci_table():
    csv_path = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_kfold/master_kfold_results.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ File tidak ditemukan: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 95% CI multiplier for 5 folds (degrees of freedom = 4)
    # Using t-distribution: t(0.025, 4) ≈ 2.776
    # CI = Mean ± (2.776 * Std / sqrt(5))
    # 2.776 / 2.23606797 = 1.24146
    CI_MULTIPLIER = 1.2415

    print("\n" + "="*90)
    print("📊 REKAPITULASI 95% CONFIDENCE INTERVAL K-FOLD (FORMAT JAMA)".center(90))
    print("="*90 + "\n")

    metrics = ['Accuracy', 'Precision', 'Recall', 'Dice', 'IoU']
    
    formatted_data = []
    
    for index, row in df.iterrows():
        formatted_row = {
            'Dataset': str(row['Dataset']).upper(),
            'Model': row['Model'],
        }
        for m in metrics:
            mean_col = f"{m}_Mean"
            std_col = f"{m}_Std"
            if mean_col in row and std_col in row:
                mean_val = float(row[mean_col])
                std_val = float(row[std_col])
                
                # Calculate margin of error
                margin_of_error = CI_MULTIPLIER * std_val
                
                # Calculate lower and upper bounds
                lower_bound = max(0.0, mean_val - margin_of_error)
                upper_bound = min(1.0, mean_val + margin_of_error)
                
                # Format: Mean (95% CI: Lower - Upper)
                formatted_row[m] = f"{mean_val:.3f} ({lower_bound:.3f}-{upper_bound:.3f})"
            else:
                formatted_row[m] = "N/A"
        formatted_data.append(formatted_row)

    formatted_df = pd.DataFrame(formatted_data)

    if not formatted_df.empty:
        # Sort
        df['Dice_Mean'] = pd.to_numeric(df['Dice_Mean'], errors='coerce')
        df_sorted = df.sort_values(by=['Dataset', 'Dice_Mean'], ascending=[True, False])
        
        # Re-apply sorting to the formatted data
        sorted_formatted_data = []
        for index, row in df_sorted.iterrows():
             # Find matching formatted row
             for f_row in formatted_data:
                 if f_row['Dataset'] == str(row['Dataset']).upper() and f_row['Model'] == row['Model']:
                     sorted_formatted_data.append(f_row)
                     break
             
        sorted_formatted_df = pd.DataFrame(sorted_formatted_data)
        
        print(tabulate(sorted_formatted_df, headers='keys', tablefmt='grid', showindex=False))
        print("\n* Format Penulisan: Rata-rata (95% CI: BatasBawah - BatasAtas)")
        print("* Anda bisa langsung copy-paste angka di atas ke tabel Manuscript JAMA Anda.")
    else:
        print("Data kosong.")
        
if __name__ == "__main__":
    generate_ci_table()
  