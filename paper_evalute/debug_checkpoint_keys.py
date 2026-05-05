"""
debug_checkpoint_keys.py
========================
Prints the actual key names inside the SE2 checkpoint vs the model.
Run this on DGX to find which keys are mismatching.

Usage:
    python ~/Clara/brain-ctc-seg/paper_evalute/debug_checkpoint_keys.py
"""
import os
import torch
from escnn import gspaces
import escnn.nn as enn
import torch.nn as nn

WEIGHTS = os.path.expanduser(
    "~/Clara/brain-ctc-seg/training/saved_models_25D/se2_unet_epoch_100.pth"
)

ckpt = torch.load(WEIGHTS, map_location='cpu', weights_only=True)

print("\n" + "="*60)
print("  CHECKPOINT KEYS (first 30):")
print("="*60)
for i, k in enumerate(list(ckpt.keys())[:30]):
    print(f"  {k}  → shape {list(ckpt[k].shape)}")

print(f"\n  ... total keys in checkpoint: {len(ckpt)}")
print("="*60)
