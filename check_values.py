import os, glob, numpy as np
data_dir = os.path.expanduser("~/Clara/local_ct_workspace_full")
ct_files = glob.glob(os.path.join(data_dir, "CT_*", "*_img.npy"))[:5]
ctc_files = glob.glob(os.path.join(data_dir, "CTC_*", "*_img.npy"))[:5]
print("CT Files:")
for f in ct_files:
    arr = np.load(f)
    print(f"  {os.path.basename(f)} -> min: {arr.min()}, max: {arr.max()}, mean: {arr.mean()}")
print("\nCTC Files:")
for f in ctc_files:
    arr = np.load(f)
    print(f"  {os.path.basename(f)} -> min: {arr.min()}, max: {arr.max()}, mean: {arr.mean()}")
