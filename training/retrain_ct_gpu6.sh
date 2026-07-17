#!/bin/bash
# Script untuk Retrain khusus CT menggunakan GPU 6

echo "=========================================================="
echo "🚀 STARTING RETRAIN FOR [CT] DATASET (Mod-Seg-SE2)"
echo "=========================================================="

DATE=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="logs/exp_retrain_ct_${DATE}"
mkdir -p "$EXP_DIR"

echo "📂 Log directory created: $EXP_DIR"
echo "⏳ Training CT on GPU 6..."

CUDA_VISIBLE_DEVICES=6 nohup python -u train_se2_by_dataset.py --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/terminal_ct.txt" 2>&1 &

echo "✅ Proses training CT berjalan di background!"
echo "Gunakan perintah berikut untuk memantau:"
echo "tail -f ${EXP_DIR}/terminal_ct.txt"
echo "=========================================================="
