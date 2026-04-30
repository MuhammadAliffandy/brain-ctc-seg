import os
import re
import pandas as pd

# ==========================================
# 1. HELPER FUNCTION TO COUNT SLICES
# ==========================================
def count_valid_slices(dataframe, root_dir):
    patient_col = 'Patient_Folder' if 'Patient_Folder' in dataframe.columns else 'Patient'
    total_patients = len(dataframe[patient_col].unique())
    total_slices = 0
    
    for patient in dataframe[patient_col].unique():
        patient_dir = os.path.join(root_dir, patient)
        if os.path.exists(patient_dir):
            img_files = [f for f in os.listdir(patient_dir) if f.endswith('_img.npy')]
            
            # Only count slices that have both image and mask (valid pairs)
            for img_name in img_files:
                mask_name = img_name.replace('_img.npy', '_mask.npy')
                if os.path.exists(os.path.join(patient_dir, mask_name)):
                    total_slices += 1
                    
    return total_patients, total_slices

# ==========================================
# 2. MAIN PROFILER EXECUTION
# ==========================================
def run_profiler():
    print("\n" + "="*50)
    print("📊 TASK 1: DATASET PROFILING FOR JOURNAL")
    print("="*50 + "\n")

    CSV_REPORT = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    
    # Validation split used during training (15%)
    VALIDATION_SPLIT = 0.15 

    try:
        df = pd.read_csv(CSV_REPORT)
    except FileNotFoundError:
        print(f"❌ Cannot find CSV file at: {CSV_REPORT}")
        return

    print("Reading CSV and splitting data (random_state=42)...")
    # Use the exact same random_state=42 to ensure identical splits as training
    train_df = df.sample(frac=(1 - VALIDATION_SPLIT), random_state=42)
    val_df = df.drop(train_df.index)

    print("Scanning local workspace for valid image-mask pairs...\n")
    
    train_patients, train_slices = count_valid_slices(train_df, LOCAL_DATA_PATH)
    val_patients, val_slices = count_valid_slices(val_df, LOCAL_DATA_PATH)
    
    total_all_patients = train_patients + val_patients
    total_all_slices = train_slices + val_slices

    print("📁 DATA DISTRIBUTION SUMMARY:")
    print("-" * 50)
    print(f"  • Total Patients (Train)     : {train_patients} Patients")
    print(f"  • Total CT Slices (Train)    : {train_slices} Slices")
    print(f"  • Total Patients (Val/Test)  : {val_patients} Patients")
    print(f"  • Total CT Slices (Val/Test) : {val_slices} Slices")
    print("-" * 50)
    print(f"  • GRAND TOTAL PATIENTS       : {total_all_patients} Patients")
    print(f"  • GRAND TOTAL SLICES         : {total_all_slices} Slices")
    print("="*50 + "\n")
    print("💡 Tip: You can safely copy these numbers into your paper's Dataset section.")

if __name__ == "__main__":
    run_profiler()