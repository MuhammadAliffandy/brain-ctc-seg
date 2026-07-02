#!/bin/bash
# Menjalankan 5 model pembanding secara berurutan untuk dataset CTC
# Menggunakan nohup agar training tidak terhenti jika SSH terputus

echo "=========================================================="
echo "🚀 STARTING COMPARISON MODELS TRAINING FOR [CTC] DATASET"
echo "=========================================================="

DATE=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="logs/exp_${DATE}"
mkdir -p "$EXP_DIR"

echo -e "\n[1/5] Training HarmonicNet (CTC)..."
nohup python train_comparison_models.py --model harmonic --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_harmonic_ctc.txt" 2>&1
echo "✅ HarmonicNet CTC done. Log: ${EXP_DIR}/log_harmonic_ctc.txt"

echo -e "\n[2/5] Training Standard UNet (CTC)..."
nohup python train_comparison_models.py --model unet --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_unet_ctc.txt" 2>&1
echo "✅ Standard UNet CTC done. Log: ${EXP_DIR}/log_unet_ctc.txt"

echo -e "\n[3/5] Training nnUNet (CTC)..."
nohup python train_comparison_models.py --model nnunet --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_nnunet_ctc.txt" 2>&1
echo "✅ nnUNet CTC done. Log: ${EXP_DIR}/log_nnunet_ctc.txt"

echo -e "\n[4/5] Training Attention UNet (CTC)..."
nohup python train_comparison_models.py --model attention --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_attention_ctc.txt" 2>&1
echo "✅ Attention UNet CTC done. Log: ${EXP_DIR}/log_attention_ctc.txt"

echo -e "\n[5/5] Training TransUNet (CTC)..."
nohup python train_comparison_models.py --model transunet --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_transunet_ctc.txt" 2>&1
echo "✅ TransUNet CTC done. Log: ${EXP_DIR}/log_transunet_ctc.txt"

echo -e "\n=========================================================="
echo "✅ ALL CTC COMPARISON MODELS COMPLETED!"
echo "=========================================================="
