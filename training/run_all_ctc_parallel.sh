#!/bin/bash
# Menjalankan 6 model (SE2 + 5 Pembanding) secara paralel pada GPU yang berbeda
# untuk dataset CTC.

# Mengambil tanggal hari ini untuk penamaan log (format YYYYMMDD)
DATE=$(date +"%Y%m%d")

echo "=========================================================="
echo "🚀 STARTING PARALLEL TRAINING FOR [CTC] DATASET"
echo "=========================================================="
echo "Memulai eksekusi di background menggunakan nohup..."

# Kita hanya memiliki 4 GPU yang kosong (2, 3, 4, 5).
# DGX diset ke Exclusive Process, jadi 1 GPU hanya bisa 1 proses.
# Solusi: Kita antrekan modelnya!

echo "[GPU 2] Training Attention U-Net..."
(
    # CUDA_VISIBLE_DEVICES=2 python train_se2_by_dataset.py --dataset ctc > "log_se2_ctc_${DATE}.txt" 2>&1
    CUDA_VISIBLE_DEVICES=2 python train_comparison_models.py --model attention --dataset ctc > "log_attention_ctc_${DATE}.txt" 2>&1
) &

echo "[GPU 3] Training HarmonicNet lalu dilanjut TransUNet..."
(
    CUDA_VISIBLE_DEVICES=3 python train_comparison_models.py --model harmonic --dataset ctc > "log_harmonic_ctc_${DATE}.txt" 2>&1
    CUDA_VISIBLE_DEVICES=3 python train_comparison_models.py --model transunet --dataset ctc > "log_transunet_ctc_${DATE}.txt" 2>&1
) &

echo "[GPU 4] Training Standard U-Net (CTC)..."
CUDA_VISIBLE_DEVICES=4 nohup python train_comparison_models.py --model unet --dataset ctc > "log_unet_ctc_${DATE}.txt" 2>&1 &

echo "[GPU 5] Training nnU-Net (CTC)..."
CUDA_VISIBLE_DEVICES=5 nohup python train_comparison_models.py --model nnunet --dataset ctc > "log_nnunet_ctc_${DATE}.txt" 2>&1 &

echo "=========================================================="
echo "✅ SEMUA PROSES TELAH DIJALANKAN DI BACKGROUND!"
echo "Gunakan 'tail -f log_nama_model_ctc_${DATE}.txt' untuk memantau."
echo "Atau jalankan 'nvidia-smi' untuk melihat utilisasi GPU 1-6."
echo "=========================================================="
