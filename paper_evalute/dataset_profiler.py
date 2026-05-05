import os
import pandas as pd
import re
import zipfile

def run_profiler():
    print("\n" + "="*70)
    print("📊 TASK 1: COMPREHENSIVE DATASET PROFILING FOR JOURNAL")
    print("="*70 + "\n")

    CSV_REPORT = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    LOCAL_DATA_PATH = os.path.expanduser("~/Clara/local_ct_workspace_full") 
    GDRIVE_DATA_DIR = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Preprocessed_NPY")
    VALIDATION_SPLIT = 0.15 

    # ==========================================
    # 0. SCAN ORIGINAL ZIP FILES (TRUE TOTAL)
    # ==========================================
    print("Scanning original ZIP files in GDrive for TRUE totals...\n")
    true_ct_img = 0
    true_ctc_img = 0
    
    if os.path.exists(GDRIVE_DATA_DIR):
        zip_files = [f for f in os.listdir(GDRIVE_DATA_DIR) if f.endswith('.zip')]
        for z_file in zip_files:
            is_ct = z_file.startswith('CT_')
            is_ctc = z_file.startswith('CTC_') or z_file.startswith('CTW_')
            
            try:
                with zipfile.ZipFile(os.path.join(GDRIVE_DATA_DIR, z_file), 'r') as zr:
                    # Count files ending with _img.npy inside the zip
                    img_count = sum(1 for name in zr.namelist() if name.endswith('_img.npy'))
                    if is_ct: true_ct_img += img_count
                    elif is_ctc: true_ctc_img += img_count
            except Exception as e:
                pass

    # ==========================================
    # 1. SCAN DIRECTLY FROM LOCAL WORKSPACE FOLDERS
    # ==========================================
    print("Scanning local workspace for ALL Raw Files (CT & CTC)...\n")
    
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"❌ Cannot find local data path: {LOCAL_DATA_PATH}")
        return

    all_patient_folders = sorted([d for d in os.listdir(LOCAL_DATA_PATH) if os.path.isdir(os.path.join(LOCAL_DATA_PATH, d))])
    
    # Initialize specific counters for slice types
    count_ct_img = 0
    count_ctc_img = 0
    valid_ct_pairs = 0
    valid_ctc_pairs = 0
    
    raw_img_total = 0
    raw_mask_total = 0
    missing_masks_total = 0
    
    patient_valid_counts = {} 

    for patient in all_patient_folders:
        patient_dir = os.path.join(LOCAL_DATA_PATH, patient)
        all_files = os.listdir(patient_dir)
        
        img_files = [f for f in all_files if f.endswith('_img.npy')]
        mask_files = [f for f in all_files if f.endswith('_mask.npy')]
        
        # Determine category based on folder name
        is_ct = patient.startswith('CT_')
        is_ctc = patient.startswith('CTC_') or patient.startswith('CTW_')
        
        # Count raw images per category
        if is_ct:
            count_ct_img += len(img_files)
        elif is_ctc:
            count_ctc_img += len(img_files)
            
        raw_img_total += len(img_files)
        raw_mask_total += len(mask_files)
        
        valid_for_this_patient = 0
        for img_name in img_files:
            mask_name = img_name.replace('_img.npy', '_mask.npy')
            if os.path.exists(os.path.join(patient_dir, mask_name)):
                valid_for_this_patient += 1
                # Count valid pairs per category
                if is_ct:
                    valid_ct_pairs += 1
                elif is_ctc:
                    valid_ctc_pairs += 1
            else:
                missing_masks_total += 1
        
        if valid_for_this_patient > 0:
            patient_valid_counts[patient] = valid_for_this_patient

    # ==========================================
    # 2. OUTPUT DETAILED STATISTICS
    # ==========================================
    print("📁 1. ORIGINAL RAW DATASET (From ZIP files)")
    print("-" * 70)
    print(f"  • TRUE CT Slices found in ZIPs  : {true_ct_img} slices")
    print(f"  • TRUE CTC Slices found in ZIPs : {true_ctc_img} slices")
    print(f"  • TRUE Total Slices             : {true_ct_img + true_ctc_img} slices")
    
    print("\n📁 2. EXTRACTED DATASET (In Local Workspace)")
    print("-" * 70)
    print(f"  • Total Extracted Slices        : {raw_img_total} .npy files")
    print(f"       - CT type slices         : {count_ct_img} files")
    print(f"       - CTC type slices        : {count_ctc_img} files")
    
    if raw_img_total < (true_ct_img + true_ctc_img):
        print("\n⚠️  WARNING: EXTRACTED SLICES < TRUE SLICES!")
        print("   This means the extraction process was interrupted previously.")
        print("   We are using a new workspace: ~/Clara/local_ct_workspace_full")
        print("   If it is still incomplete, please delete it to force re-extraction.")
    
    print("\n✂️ 3. DATA FILTERING (Exclusion Criteria on Extracted Data)")
    print("-" * 70)
    print(f"  • Slices excluded (No Mask)   : {missing_masks_total} slices")
    print(f"  • Slices included (Valid Pair): {valid_ct_pairs + valid_ctc_pairs} slices")
    print(f"       - Valid CT Pairs         : {valid_ct_pairs} slices")
    print(f"       - Valid CTC Pairs        : {valid_ctc_pairs} slices")

    # ==========================================
    # 3. FINAL DATASET SPLIT CALCULATION
    # ==========================================
    print("\n📌 3. FINAL DATASET FOR TRAINING & EVALUATION (Split 85/15)")
    print("-" * 70)
    
    # Calculate splits mathematically based on valid pairs found
    ct_train = int(valid_ct_pairs * 0.85)
    ct_test = valid_ct_pairs - ct_train
    
    ctc_train = int(valid_ctc_pairs * 0.85)
    ctc_test = valid_ctc_pairs - ctc_train

    print(f"  • CT Private Dataset          : {ct_train} Train | {ct_test} Test")
    print(f"  • CTC Private Dataset         : {ctc_train} Train | {ctc_test} Test")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_profiler()