import os
import re

def check_public_dataset():
    print("\n" + "="*70)
    print("📊 DATASET PROFILER: PUBLIC DATASET (EVALUATE_CT_OF_BRAIN)")
    print("="*70 + "\n")

    PUBLIC_DATA_PATH = os.path.expanduser("~/Clara/public_dataset_npy")
    
    if not os.path.exists(PUBLIC_DATA_PATH):
        print(f"❌ Folder tidak ditemukan: {PUBLIC_DATA_PATH}")
        print("Pastikan Anda sudah menaruh dataset publik di folder ini (untuk evaluate_ct_of_brain.py).")
        return

    # Scan for patient folders or direct NPY files
    all_patient_folders = sorted([d for d in os.listdir(PUBLIC_DATA_PATH) if os.path.isdir(os.path.join(PUBLIC_DATA_PATH, d))])
    
    total_images = 0
    total_valid_pairs = 0
    missing_masks = 0

    print(f"🔍 Memindai direktori: {PUBLIC_DATA_PATH}")
    
    if len(all_patient_folders) > 0:
        print(f"   Ditemukan {len(all_patient_folders)} sub-folder/pasien.\n")
        
        for patient in all_patient_folders:
            patient_dir = os.path.join(PUBLIC_DATA_PATH, patient)
            all_files = os.listdir(patient_dir)
            
            img_files = [f for f in all_files if f.endswith('_img.npy')]
            total_images += len(img_files)
            
            for img_name in img_files:
                mask_name = img_name.replace('_img.npy', '_mask.npy')
                if os.path.exists(os.path.join(patient_dir, mask_name)):
                    total_valid_pairs += 1
                else:
                    missing_masks += 1
    else:
        # Jika tidak ada sub-folder, mungkin file .npy ditaruh langsung di root folder
        all_files = os.listdir(PUBLIC_DATA_PATH)
        img_files = [f for f in all_files if f.endswith('_img.npy')]
        total_images += len(img_files)
        print(f"   Tidak ada sub-folder. Memindai {total_images} file .npy langsung di root.\n")
        
        for img_name in img_files:
            mask_name = img_name.replace('_img.npy', '_mask.npy')
            if os.path.exists(os.path.join(PUBLIC_DATA_PATH, mask_name)):
                total_valid_pairs += 1
            else:
                missing_masks += 1

    print("📁 1. RINGKASAN DATASET PUBLIK (EXTERNAL VALIDATION)")
    print("-" * 70)
    if len(all_patient_folders) > 0:
        print(f"  • Total Sub-Folder / Pasien   : {len(all_patient_folders)}")
    print(f"  • Total Raw Slices (Images)   : {total_images} files")
    
    print("\n✂️ 2. STATUS GROUND TRUTH (LABEL DOKTER)")
    print("-" * 70)
    print(f"  • Slices lengkap dengan Mask  : {total_valid_pairs} slices (Valid for Evaluation)")
    print(f"  • Slices tanpa Mask           : {missing_masks} slices (Cannot calculate Dice)")

    print("\n📌 3. DATA PENULISAN UNTUK PAPER (DENGAN SPLIT 85/15)")
    print("-" * 70)
    
    test_split = int(total_valid_pairs * 0.15) + 1 # Approximate calculation for display
    train_split = total_valid_pairs - test_split
    
    print(f"Tulis di metodologi: Dataset publik eksternal yang diuji memiliki total")
    print(f"sebanyak {total_valid_pairs} data slice. Untuk menjaga konsistensi dengan")
    print(f"pipeline training utama, dataset publik ini juga dibagi menggunakan")
    print(f"rasio 85/15 (Seed=42). Evaluasi akhir murni (Testing) dilakukan")
    print(f"menggunakan 15% dari data tersebut (± {test_split} slices) untuk mengukur")
    print("performa generalisasi arsitektur Mod-Seg-SE(2).")
    print("="*70 + "\n")

if __name__ == "__main__":
    check_public_dataset()
