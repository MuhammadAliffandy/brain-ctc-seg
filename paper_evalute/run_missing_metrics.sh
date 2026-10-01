#!/bin/bash
# run_missing_metrics.sh
# Jalankan script ini untuk menghitung metrik manuscript (HD95, Acc, Prec, Rec) di background

# Masuk ke direktori project yang benar
cd ~/Clara/brain-ctc-seg

echo "🚀 Menjalankan kalkulasi metrics TMI di background..."
echo "Log akan disimpan di: ~/Clara/brain-ctc-seg/paper_evalute/metrics_output.log"

nohup python -u paper_evalute/generate_missing_manuscript_metrics.py > paper_evalute/metrics_output.log 2>&1 &

echo "✅ Proses berjalan! Kakak bisa menutup terminal sekarang."
echo "Gunakan perintah ini untuk memantau progres (jika masih online):"
echo "tail -f paper_evalute/metrics_output.log"
