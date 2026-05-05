"""
prepare_full_dataset.py
=======================
Extracts ALL patient zip files from Google Drive to local_ct_workspace,
then runs a complete profiling showing CT vs CTC counts.

Run this ONCE before any training to ensure all data is available locally.

Usage (DGX):
    python ~/Clara/brain-ctc-seg/paper_evalute/prepare_full_dataset.py
"""

import os, re, zipfile, shutil
import numpy as np
from tqdm import tqdm


# ── Paths ────────────────────────────────────────────────────────────────────
GDRIVE_ROOT   = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive")
GDRIVE_DATA   = os.path.join(GDRIVE_ROOT, "Dataset_CT_Preprocessed_NPY")
LOCAL_ROOT    = os.path.expanduser("~/Clara/local_ct_workspace")


def scan_directory(path, label):
    """Scan a directory and return (ct_pairs, ctc_pairs, other_pairs) counts."""
    if not os.path.exists(path):
        print(f"  ❌ Path not found: {path}")
        return 0, 0, 0, []

    folders = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    ct_pairs = ctc_pairs = other_pairs = 0
    all_folders = []

    for folder in folders:
        fd = os.path.join(path, folder)
        imgs  = [f for f in os.listdir(fd) if f.endswith('_img.npy')]
        valid = sum(1 for f in imgs if os.path.exists(os.path.join(fd, f.replace('_img.npy', '_mask.npy'))))

        if folder.startswith('CT_'):
            ct_pairs += valid
        elif folder.startswith('CTC_') or folder.startswith('CTW_'):
            ctc_pairs += valid
        else:
            other_pairs += valid

        all_folders.append((folder, valid))

    return ct_pairs, ctc_pairs, other_pairs, all_folders


def scan_zips(path):
    """Scan zip files in Google Drive source."""
    if not os.path.exists(path):
        return 0, 0, 0
    zips = [f for f in os.listdir(path) if f.endswith('.zip')]
    ct_zips = sum(1 for z in zips if z.startswith('CT_'))
    ctc_zips = sum(1 for z in zips if z.startswith('CTC_') or z.startswith('CTW_'))
    other_zips = len(zips) - ct_zips - ctc_zips
    return ct_zips, ctc_zips, other_zips, zips


def extract_all(force=False):
    """Extract ALL zips from Google Drive to local workspace."""
    os.makedirs(LOCAL_ROOT, exist_ok=True)

    # Case 1: Zips in root GDRIVE_DATA
    zips_in_root = [f for f in os.listdir(GDRIVE_DATA) if f.endswith('.zip')] if os.path.exists(GDRIVE_DATA) else []

    # Case 2: Pre-extracted folders in GDRIVE_DATA
    folders_in_root = [d for d in os.listdir(GDRIVE_DATA) if os.path.isdir(os.path.join(GDRIVE_DATA, d))] if os.path.exists(GDRIVE_DATA) else []

    extracted = 0; skipped = 0; copied = 0

    if zips_in_root:
        print(f"\n  📦 Found {len(zips_in_root)} zip files in source. Extracting...")
        for zf in tqdm(zips_in_root, desc="Extracting"):
            patient = zf.replace('.zip', '')
            dst = os.path.join(LOCAL_ROOT, patient)
            if os.path.exists(dst) and not force:
                skipped += 1; continue
            try:
                with zipfile.ZipFile(os.path.join(GDRIVE_DATA, zf), 'r') as zr:
                    zr.extractall(dst)
                extracted += 1
            except Exception as e:
                print(f"  ⚠️  Error extracting {zf}: {e}")

    elif folders_in_root:
        print(f"\n  📁 Found {len(folders_in_root)} patient folders in source. Copying...")
        for folder in tqdm(folders_in_root, desc="Copying"):
            src = os.path.join(GDRIVE_DATA, folder)
            dst = os.path.join(LOCAL_ROOT, folder)
            if os.path.exists(dst) and not force:
                skipped += 1; continue
            try:
                shutil.copytree(src, dst)
                copied += 1
            except Exception as e:
                print(f"  ⚠️  Error copying {folder}: {e}")

    print(f"\n  ✅ Extracted: {extracted} | Copied: {copied} | Skipped (already exist): {skipped}")
    return extracted + copied


def full_profile():
    print("\n" + "="*70)
    print("  📊 FULL DATASET PROFILER — CT vs CTC")
    print("="*70)

    # ── 1. Source scan ────────────────────────────────────────────────────
    print(f"\n  [SOURCE] Scanning: {GDRIVE_DATA}")
    if os.path.exists(GDRIVE_DATA):
        zips = [f for f in os.listdir(GDRIVE_DATA) if f.endswith('.zip')]
        src_folders = [d for d in os.listdir(GDRIVE_DATA) if os.path.isdir(os.path.join(GDRIVE_DATA, d))]

        if zips:
            ct_z  = sum(1 for z in zips if z.startswith('CT_'))
            ctc_z = sum(1 for z in zips if z.startswith('CTC_') or z.startswith('CTW_'))
            print(f"  Zip files found   : {len(zips)} total → CT: {ct_z} | CTC: {ctc_z}")
        if src_folders:
            ct_src_ct, ctc_src, oth_src, _ = scan_directory(GDRIVE_DATA, "SOURCE")
            print(f"  Folders found     : {len(src_folders)} total → CT: {ct_src_ct} slices | CTC: {ctc_src} slices")
    else:
        print(f"  ⚠️  Source path not found: {GDRIVE_DATA}")

    # ── 2. Local scan ─────────────────────────────────────────────────────
    print(f"\n  [LOCAL] Scanning: {LOCAL_ROOT}")
    ct_local, ctc_local, other_local, folders = scan_directory(LOCAL_ROOT, "LOCAL")
    total_local = ct_local + ctc_local + other_local

    print(f"  Valid pairs found : {total_local} total")
    print(f"    CT  (CT_*)      : {ct_local:>5} slices")
    print(f"    CTC (CTC_/CTW_) : {ctc_local:>5} slices")
    if other_local > 0:
        print(f"    Other prefix    : {other_local:>5} slices  ← CHECK FOLDER NAMES!")

    # Show folder name samples
    if folders:
        sample = folders[:5]
        print(f"\n  Sample folder names: {[f[0] for f in sample]}")
        if other_local > 0:
            others = [f[0] for f in folders if not f[0].startswith('CT_') and not f[0].startswith('CTC_') and not f[0].startswith('CTW_')]
            print(f"  ⚠️  Unclassified folders: {others[:10]}")

    # ── 3. Split calculation ──────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  EXPECTED (client spec): CT=3696 | CTC=4303 | Total=7999")
    print(f"  FOUND   (local):        CT={ct_local:<5} | CTC={ctc_local:<5} | Total={total_local}")
    gap_ct  = 3696 - ct_local
    gap_ctc = 4303 - ctc_local
    if gap_ct > 0 or gap_ctc > 0:
        print(f"  GAP (missing locally): CT={gap_ct} | CTC={gap_ctc}")
        print(f"\n  ⚠️  Data is INCOMPLETE. Run with --extract to extract from source.")
    else:
        print(f"\n  ✅ All data is available locally!")

        # Print final split table
        ct_train  = int(ct_local  * 0.85); ct_test  = ct_local  - ct_train
        ctc_train = int(ctc_local * 0.85); ctc_test = ctc_local - ctc_train
        print(f"\n  SPLIT 85/15 (seed=42):")
        print(f"    CT  → Train: {ct_train:>5} | Val: {ct_test:>4}")
        print(f"    CTC → Train: {ctc_train:>5} | Val: {ctc_test:>4}")

    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profile and optionally extract CT/CTC dataset")
    parser.add_argument('--extract', action='store_true',
                        help="Extract/copy all patient data from Google Drive source to local workspace")
    parser.add_argument('--force', action='store_true',
                        help="Re-extract even if already present locally (use with --extract)")
    args = parser.parse_args()

    if args.extract:
        print("\n" + "="*70)
        print("  🚚 DATA EXTRACTION MODE")
        print("="*70)
        n = extract_all(force=args.force)
        print(f"\n  Processed {n} patients.")

    full_profile()
