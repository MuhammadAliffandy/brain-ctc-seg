#!/bin/bash
# Script untuk mentraining khusus 3 model yang terlewat (N/A) di Kaggle Stroke dan Kaggle Hemorrhage
# Waktu eksekusi: sekitar 1.5 - 2 jam di DGX

echo "=========================================================="
echo "🚀 STARTING TRAINING FOR MISSING KAGGLE MODELS (HD95 N/A)"
echo "=========================================================="

# Gunakan direktori script ini
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo -e "\n[1/3] Training HarmonicNet on Kaggle Stroke..."
# Kaggle Stroke script is train_all_intra.py
CUDA_VISIBLE_DEVICES=0 python public_dataset/train_all_intra.py "HarmonicNet"
echo "✅ HarmonicNet Stroke done."

echo -e "\n[2/3] Training HarmonicNet on Kaggle Hemorrhage..."
# Kaggle Hemorrhage script is train_all_intra_hemorrhage.py
CUDA_VISIBLE_DEVICES=0 python public_dataset/train_all_intra_hemorrhage.py "HarmonicNet"
echo "✅ HarmonicNet Hemorrhage done."

echo -e "\n[3/3] Training Standard U-Net on Kaggle Hemorrhage..."
CUDA_VISIBLE_DEVICES=0 python public_dataset/train_all_intra_hemorrhage.py "Standard U-Net"
echo "✅ Standard U-Net Hemorrhage done."

echo -e "\n=========================================================="
echo "✅ ALL MISSING TRAINING COMPLETED!"
echo "Sekarang jalankan evaluasi HD95: python paper_evalute/generate_remaining_hd95.py"
echo "=========================================================="
