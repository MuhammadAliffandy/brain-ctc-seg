import os
import pandas as pd

# ==========================================
# 1. HELPER TO COUNT RAW & VALID DATA
# ==========================================
def profile_dataset(dataframe, root_dir):
    patient_col = 'Patient_Folder' if 'Patient_Folder' in dataframe.columns else 'Patient'
    total_patients = len(dataframe[patient_col].unique())
    
    raw_img_slices = 0
    raw_mask_slices = 0
    valid_pairs = 0
    missing_masks = 0
    
    for patient in dataframe[patient_col].unique():
        patient_dir = os.path.join(root_dir, patient)
        if os.path.exists(patient_dir):
            all_files = os.listdir(patient_dir)
            
            # Count RAW original files
            img_files = [f for f in all_files if f.endswith('_img.npy')]
            mask_files = [f for f in all_files if f.endswith('_mask.npy')]
            
            raw_img_slices += len(img_files)
            raw_mask_slices += len(mask_files)
            
            # Count VALID pairs (Inclusion Criteria)
            for img_name in img_files:
                mask_name = img_name.replace('_img.npy', '_mask.npy')
                if os.path.exists(os.path.join(patient_dir, mask_name)):
                    valid_pairs += 1
                else:
                    missing_masks += 1
                    
    return {
        'patients': total_patients,
        'raw_images': raw_img_slices,
        'raw_masks': raw_mask_slices,
        'valid_pairs': valid_pairs,
        'excluded': missing_masks
    }

# ==========================================
# 2. MAIN EXECUTION
# ==========================================
def run_profiler():
    print("\n" + "="*60)
    print("📊 TASK 1: COMPREHENSIVE DATASET PROFILING FOR JOURNAL")
    print("="*60 + "\n")

    CSV_REPORT = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    
    VALIDATION_SPLIT = 0.15 

    try:
        df = pd.read_csv(CSV_REPORT)
    except FileNotFoundError:
        print(f"❌ Cannot find CSV file at: {CSV_REPORT}")
        return

    train_df = df.sample(frac=(1 - VALIDATION_SPLIT), random_state=42)
    val_df = df.drop(train_df.index)

    print("Scanning local workspace for Raw Files and Valid Pairs...\n")
    
    train_stats = profile_dataset(train_df, LOCAL_DATA_PATH)
    val_stats = profile_dataset(val_df, LOCAL_DATA_PATH)
    
    total_patients = train_stats['patients'] + val_stats['patients']
    total_raw_images = train_stats['raw_images'] + val_stats['raw_images']
    total_raw_masks = train_stats['raw_masks'] + val_stats['raw_masks']
    total_valid = train_stats['valid_pairs'] + val_stats['valid_pairs']
    total_excluded = train_stats['excluded'] + val_stats['excluded']

    print("📁 1. ORIGINAL RAW DATASET (Acquisition Stage)")
    print("-" * 60)
    print(f"  • Total Patients Scanned      : {total_patients} Patients")
    print(f"  • Total Raw CT Slices (_img)  : {total_raw_images} .npy files")
    print(f"  • Total Raw GT Masks (_mask)  : {total_raw_masks} .npy files")
    
    print("\n✂️ 2. DATA FILTERING (Exclusion Criteria)")
    print("-" * 60)
    print(f"  • Slices excluded (No GT Mask): {total_excluded} slices dropped")
    print(f"  • Slices included (Valid Pair): {total_valid} slices retained")

    print("\n📌 3. FINAL DATASET FOR TRAINING & EVALUATION")
    print("-" * 60)
    print(f"  • Training Set (85%)          : {train_stats['valid_pairs']} slices ({train_stats['patients']} patients)")
    print(f"  • Validation/Test Set (15%)   : {val_stats['valid_pairs']} slices ({val_stats['patients']} patients)")
    print("="*60 + "\n")
    
    print("💡 Tip for Paper: 'Initially, {} raw CT slices were extracted. After removing {} slices without corresponding ground truth annotations, a total of {} valid pairs were utilized for the experiments...'".format(total_raw_images, total_excluded, total_valid))

if __name__ == "__main__":
    run_profiler()