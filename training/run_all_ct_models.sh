#!/bin/bash
# Menjalankan 5 model pembanding secara berurutan untuk dataset CT

echo "=========================================================="
echo "🚀 STARTING COMPARISON MODELS TRAINING FOR [CT] DATASET"
echo "=========================================================="

echo -e "\n[1/5] Training HarmonicNet (CT)..."
python train_comparison_models.py --model harmonic --dataset ct

echo -e "\n[2/5] Training Standard UNet (CT)..."
python train_comparison_models.py --model unet --dataset ct

echo -e "\n[3/5] Training nnUNet (CT)..."
python train_comparison_models.py --model nnunet --dataset ct

echo -e "\n[4/5] Training Attention UNet (CT)..."
python train_comparison_models.py --model attention --dataset ct

echo -e "\n[5/5] Training TransUNet (CT)..."
python train_comparison_models.py --model transunet --dataset ct

echo -e "\n=========================================================="
echo "✅ ALL CT COMPARISON MODELS COMPLETED!"
echo "=========================================================="
