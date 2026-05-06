#!/bin/bash
# Menjalankan 5 model pembanding secara berurutan untuk dataset CT
# Menggunakan nohup agar training tidak terhenti jika SSH terputus

echo "=========================================================="
echo "🚀 STARTING COMPARISON MODELS TRAINING FOR [CT] DATASET"
echo "=========================================================="

echo -e "\n[1/5] Training HarmonicNet (CT)..."
nohup python train_comparison_models.py --model harmonic --dataset ct > log_harmonic_ct.txt 2>&1
echo "✅ HarmonicNet CT done. Log: log_harmonic_ct.txt"

echo -e "\n[2/5] Training Standard UNet (CT)..."
nohup python train_comparison_models.py --model unet --dataset ct > log_unet_ct.txt 2>&1
echo "✅ Standard UNet CT done. Log: log_unet_ct.txt"

echo -e "\n[3/5] Training nnUNet (CT)..."
nohup python train_comparison_models.py --model nnunet --dataset ct > log_nnunet_ct.txt 2>&1
echo "✅ nnUNet CT done. Log: log_nnunet_ct.txt"

echo -e "\n[4/5] Training Attention UNet (CT)..."
nohup python train_comparison_models.py --model attention --dataset ct > log_attention_ct.txt 2>&1
echo "✅ Attention UNet CT done. Log: log_attention_ct.txt"

echo -e "\n[5/5] Training TransUNet (CT)..."
nohup python train_comparison_models.py --model transunet --dataset ct > log_transunet_ct.txt 2>&1
echo "✅ TransUNet CT done. Log: log_transunet_ct.txt"

echo -e "\n=========================================================="
echo "✅ ALL CT COMPARISON MODELS COMPLETED!"
echo "=========================================================="
