import os
import pandas as pd
from tabulate import tabulate

def print_kfold_table():
    csv_path = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_kfold/master_kfold_results.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ File tidak ditemukan: {csv_path}")
        print("Pastikan script kfold sudah selesai berjalan dan men-generate file ini.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error membaca CSV: {e}")
        return

    print("\n" + "="*80)
    print("📊 REKAPITULASI HASIL 5-FOLD CROSS VALIDATION (CT & CTC)".center(80))
    print("="*80 + "\n")

    # Format the columns for better readability (Mean ± Std)
    metrics = ['Accuracy', 'Precision', 'Recall', 'Dice', 'IoU']
    
    # Initialize a new dataframe for the formatted table
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
                # Format to 4 decimal places
                formatted_row[m] = f"{row[mean_col]:.4f} ± {row[std_col]:.4f}"
            else:
                formatted_row[m] = "N/A"
        formatted_data.append(formatted_row)

    formatted_df = pd.DataFrame(formatted_data)

    if not formatted_df.empty:
        # Sort by Dataset then by Dice (descending) to see best models easily
        # We need to extract the mean value for sorting
        df['Dice_Mean'] = pd.to_numeric(df['Dice_Mean'], errors='coerce')
        df_sorted = df.sort_values(by=['Dataset', 'Dice_Mean'], ascending=[True, False])
        
        # Re-format sorted data
        sorted_formatted_data = []
        for index, row in df_sorted.iterrows():
             formatted_row = {
                 'Dataset': str(row['Dataset']).upper(),
                 'Model': row['Model'],
             }
             for m in metrics:
                 mean_col = f"{m}_Mean"
                 std_col = f"{m}_Std"
                 if mean_col in row and std_col in row:
                     formatted_row[m] = f"{row[mean_col]:.4f} ± {row[std_col]:.4f}"
                 else:
                     formatted_row[m] = "N/A"
             sorted_formatted_data.append(formatted_row)
             
        sorted_formatted_df = pd.DataFrame(sorted_formatted_data)
        
        print(tabulate(sorted_formatted_df, headers='keys', tablefmt='grid', showindex=False))
        print("\n* Diurutkan berdasarkan nilai rata-rata Dice tertinggi per Dataset.")
    else:
        print("Data kosong.")
        
if __name__ == "__main__":
    print_kfold_table()
