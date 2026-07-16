#!/bin/bash
# Script untuk Retrain khusus STROKE menggunakan GPU 6

echo "=========================================================="
echo "🚀 STARTING RETRAIN FOR [STROKE] DATASET"
echo "=========================================================="

DATE=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="log_retrain_stroke_${DATE}.txt"

echo "📂 Log file will be saved as: $LOG_FILE"
echo "⏳ Training STROKE on GPU 6..."

# Kita tidak pakai --log_dir karena script stroke tidak punya argparse
# Langsung kita arahkan output (stdout) ke file log
CUDA_VISIBLE_DEVICES=6 nohup python -u train_all_intra.py > "$LOG_FILE" 2>&1 &

echo "✅ Proses training STROKE berjalan di background!"
echo "Gunakan perintah berikut untuk memantau:"
echo "tail -f $LOG_FILE"
echo "=========================================================="
