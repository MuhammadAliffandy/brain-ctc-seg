#!/bin/bash
# Script untuk Retrain khusus CTC menggunakan GPU 7

echo "=========================================================="
echo "🚀 STARTING RETRAIN FOR [CTC] DATASET (Mod-Seg-SE2)"
echo "=========================================================="

DATE=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="logs/exp_retrain_ctc_${DATE}"
mkdir -p "$EXP_DIR"

echo "📂 Log directory created: $EXP_DIR"
echo "⏳ Training CTC on GPU 3..."

CUDA_VISIBLE_DEVICES=3 nohup python -u train_se2_by_dataset.py --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/terminal_ctc.txt" 2>&1 &

echo "✅ Proses training CTC berjalan di background!"
echo "Gunakan perintah berikut untuk memantau:"
echo "tail -f ${EXP_DIR}/terminal_ctc.txt"
echo "=========================================================="
