#!/bin/bash
# Menjalankan 5 model pembanding secara berurutan untuk dataset CTC

echo "=========================================================="
echo "🚀 STARTING COMPARISON MODELS TRAINING FOR [CTC] DATASET"
echo "=========================================================="

echo -e "\n[1/5] Training HarmonicNet (CTC)..."
python train_comparison_models.py --model harmonic --dataset ctc

echo -e "\n[2/5] Training Standard UNet (CTC)..."
python train_comparison_models.py --model unet --dataset ctc

echo -e "\n[3/5] Training nnUNet (CTC)..."
python train_comparison_models.py --model nnunet --dataset ctc

echo -e "\n[4/5] Training Attention UNet (CTC)..."
python train_comparison_models.py --model attention --dataset ctc

echo -e "\n[5/5] Training TransUNet (CTC)..."
python train_comparison_models.py --model transunet --dataset ctc

echo -e "\n=========================================================="
echo "✅ ALL CTC COMPARISON MODELS COMPLETED!"
echo "=========================================================="
