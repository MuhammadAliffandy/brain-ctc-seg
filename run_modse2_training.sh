#!/bin/bash
# Script untuk mentraining ulang model Mod-SE(2) murni di dataset eksternal (Kaggle & CECT)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "============================================="
echo "🚀 STARTING TRAINING MOD-SE(2) BASELINE..."
echo "============================================="

# 1. Jalankan training Mod-SE(2)
python paper_evalute/train_modse2_baseline.py

echo "============================================="
echo "✅ TRAINING & EVALUATION COMPLETED!"
echo "Hasil metrik telah diprint di log ini."
echo "============================================="
