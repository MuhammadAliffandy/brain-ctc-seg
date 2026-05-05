import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

def profile_original_data():
    print("\n" + "="*70)
    print("📊 TASK: PROFILING ORIGINAL 3D NIFTI DATASET (PRE-SLICING)")
    print("="*70 + "\n")

    # Sesuaikan path ini dengan lokasi data original Anda di DGX / Drive
    RAW_DATA_PATH = os.path.expanduser("~/Clara/new_drive/CT Brain Data/Original Data")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Error: Folder data original tidak ditemukan di {RAW_DATA_PATH}")
        print("Silakan ganti path RAW_DATA_PATH di dalam script ini jika lokasinya berbeda.")
        return

    all_folders = [f for f in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, f))]
    
    ct_folders = sorted([f for f in all_folders if "CTC" not in f.upper() and "CT" in f.upper()])
    ctc_folders = sorted([f for f in all_folders if "CTC" in f.upper() or "CTW" in f.upper()])

    print(f"🔍 Ditemukan {len(ct_folders)} pasien CT dan {len(ctc_folders)} pasien CTC.")

    def count_slices(folders_list, category_name):
        total_slices = 0
        total_labeled_slices = 0
        
        print(f"\n📁 Menghitung volume {category_name}...")
        for folder in tqdm(folders_list, desc=category_name):
            patient_path = os.path.join(RAW_DATA_PATH, folder)
            images = [f for f in os.listdir(patient_path) if f.endswith('.nii.gz') and '.seg.' not in f]
            
            for img_name in images:
                drive_img_path = os.path.join(patient_path, img_name)
                mask_name = img_name.replace('.nii.gz', '.seg.nii.gz')
                drive_mask_path = os.path.join(patient_path, mask_name)

                try:
                    # Load NIfTI header (lebih cepat daripada get_fdata() seluruhnya)
                    img_obj = nib.load(drive_img_path)
                    z_slices = img_obj.shape[2]  # Dimensi Z adalah jumlah slice
                    total_slices += z_slices
                    
                    if os.path.exists(drive_mask_path):
                        mask_obj = nib.load(drive_mask_path)
                        mask_data = mask_obj.get_fdata()
                        # Hitung berapa slice yang punya label
                        labeled = sum(1 for z in range(z_slices) if np.any(mask_data[:, :, z] == 1))
                        total_labeled_slices += labeled

                except Exception as e:
                    pass
                    
        return total_slices, total_labeled_slices

    ct_total, ct_labeled = count_slices(ct_folders, "CT Dataset")
    ctc_total, ctc_labeled = count_slices(ctc_folders, "CTC Dataset")

    print("\n" + "="*70)
    print("📝 HASIL PROFILING UNTUK PAPER (METHODOLOGY SECTION)")
    print("="*70)
    print(f"Total Pasien CT       : {len(ct_folders)} Pasien")
    print(f"Total Slices CT (Raw) : {ct_total} slices")
    print(f"CT Slices w/ Lesion   : {ct_labeled} slices (Dipakai untuk Training)\n")

    print(f"Total Pasien CTC       : {len(ctc_folders)} Pasien")
    print(f"Total Slices CTC (Raw) : {ctc_total} slices")
    print(f"CTC Slices w/ Lesion   : {ctc_labeled} slices (Dipakai untuk Training)")
    print("="*70 + "\n")

if __name__ == "__main__":
    profile_original_data()
