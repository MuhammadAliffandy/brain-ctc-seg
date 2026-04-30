import os
import pandas as pd

def run_profiler():
    print("\n" + "="*70)
    print("📊 TASK 1: COMPREHENSIVE DATASET PROFILING FOR JOURNAL")
    print("="*70 + "\n")

    CSV_REPORT = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace") 
    VALIDATION_SPLIT = 0.15 

    # ==========================================
    # 1. SCAN DIRECTLY FROM LOCAL WORKSPACE FOLDERS
    # ==========================================
    print("Scanning local workspace for ALL Raw Files (CT & CTC)...\n")
    
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"❌ Cannot find local data path: {LOCAL_DATA_PATH}")
        return

    all_patient_folders = sorted([d for d in os.listdir(LOCAL_DATA_PATH) if os.path.isdir(os.path.join(LOCAL_DATA_PATH, d))])
    
    # Categorize patients
    ct_patients = [p for p in all_patient_folders if p.startswith('CT_')]
    ctc_patients = [p for p in all_patient_folders if p.startswith('CTC_')]
    other_patients = [p for p in all_patient_folders if not (p.startswith('CT_') or p.startswith('CTC_'))]

    raw_img_total = 0
    raw_mask_total = 0
    valid_pairs_total = 0
    missing_masks_total = 0
    
    patient_valid_counts = {} # To store valid slices per patient 

    for patient in all_patient_folders:
        patient_dir = os.path.join(LOCAL_DATA_PATH, patient)
        all_files = os.listdir(patient_dir)
        
        img_files = [f for f in all_files if f.endswith('_img.npy')]
        mask_files = [f for f in all_files if f.endswith('_mask.npy')]
        
        raw_img_total += len(img_files)
        raw_mask_total += len(mask_files)
        
        valid_for_this_patient = 0
        for img_name in img_files:
            mask_name = img_name.replace('_img.npy', '_mask.npy')
            if os.path.exists(os.path.join(patient_dir, mask_name)):
                valid_pairs_total += 1
                valid_for_this_patient += 1
            else:
                missing_masks_total += 1
        
        if valid_for_this_patient > 0:
            patient_valid_counts[patient] = valid_for_this_patient

    print("📁 1. ORIGINAL RAW DATASET (Acquisition Stage from Workspace)")
    print("-" * 70)
    print(f"  • Total Patients Found        : {len(all_patient_folders)} Patients")
    print(f"       - 'CT_'  type patients   : {len(ct_patients)} Patients")
    print(f"       - 'CTC_' type patients   : {len(ctc_patients)} Patients")
    if len(other_patients) > 0:
        print(f"       - Other type patients    : {len(other_patients)} Patients")
    print(f"  • Total Raw Slices (_img)     : {raw_img_total} .npy files")
    print(f"  • Total Raw GT Masks (_mask)  : {raw_mask_total} .npy files")
    
    print("\n✂️ 2. DATA FILTERING (Exclusion Criteria)")
    print("-" * 70)
    print(f"  • Slices excluded (No GT Mask): {missing_masks_total} slices dropped")
    print(f"  • Slices included (Valid Pair): {valid_pairs_total} slices retained")

    # ==========================================
    # 3. VERIFY WITH CSV SPLIT
    # ==========================================
    print("\n📌 3. FINAL DATASET FOR TRAINING & EVALUATION (Based on CSV Split)")
    print("-" * 70)
    try:
        df = pd.read_csv(CSV_REPORT)
        patient_col = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
        csv_patients = df[patient_col].unique().tolist()
        
        train_df = df.sample(frac=(1 - VALIDATION_SPLIT), random_state=42)
        val_df = df.drop(train_df.index)
        
        train_patients = train_df[patient_col].unique().tolist()
        val_patients = val_df[patient_col].unique().tolist()
        
        train_slices = sum([patient_valid_counts.get(p, 0) for p in train_patients])
        val_slices = sum([patient_valid_counts.get(p, 0) for p in val_patients])
        
        print(f"  • Training Set (85%)          : {train_slices} slices ({len(train_patients)} patients)")
        print(f"  • Validation/Test Set (15%)   : {val_slices} slices ({len(val_patients)} patients)")
        
        # Cross-check: Are there patients in the workspace that are MISSING from the CSV?
        missing_in_csv = [p for p in patient_valid_counts.keys() if p not in csv_patients]
        if missing_in_csv:
            print(f"\n  ⚠️ WARNING: Found {len(missing_in_csv)} patients in your local workspace that are NOT inside the CSV!")
            print(f"     Missing patients: {missing_in_csv}")
            print("     (Note: These patients were NOT used during training because they aren't in the CSV report)")
            
    except FileNotFoundError:
        print(f"❌ Cannot find CSV file at: {CSV_REPORT}")

    print("="*70 + "\n")

if __name__ == "__main__":
    run_profiler()