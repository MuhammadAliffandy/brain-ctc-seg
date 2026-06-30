#!/bin/bash
# Menjalankan 6 model (SE2 + 5 Pembanding) secara paralel pada GPU yang sedang kosong.
# Sesuai screenshot nvidia-smi terbaru: GPU 1, GPU 3, GPU 6 kosong (0%).
# Kita akan menugaskan 2 model per GPU (karena VRAM 80GB sangat mampu menampung 2 model).

DATE=$(date +"%Y%m%d")

echo "=========================================================="
echo "🚀 STARTING PARALLEL TRAINING FOR [CT] DATASET"
echo "✅ Menggunakan GPU yang kosong: 1, 3, 6 (Masing-masing 2 model)"
echo "=========================================================="
echo "Memulai eksekusi di background menggunakan nohup..."

# ----------------- GPU 1 -----------------
echo "[GPU 1] Training Mod-Seg-SE(2) (CT)..."
CUDA_VISIBLE_DEVICES=1 nohup python train_se2_by_dataset.py --dataset ct > "log_se2_ct_${DATE}.txt" 2>&1 &

echo "[GPU 1] Training HarmonicNet (CT)..."
CUDA_VISIBLE_DEVICES=1 nohup python train_comparison_models.py --model harmonic --dataset ct > "log_harmonic_ct_${DATE}.txt" 2>&1 &


# ----------------- GPU 3 -----------------
echo "[GPU 3] Training Standard U-Net (CT)..."
CUDA_VISIBLE_DEVICES=3 nohup python train_comparison_models.py --model unet --dataset ct > "log_unet_ct_${DATE}.txt" 2>&1 &

echo "[GPU 3] Training nnU-Net (CT)..."
CUDA_VISIBLE_DEVICES=3 nohup python train_comparison_models.py --model nnunet --dataset ct > "log_nnunet_ct_${DATE}.txt" 2>&1 &


# ----------------- GPU 6 -----------------
echo "[GPU 6] Training Attention U-Net (CT)..."
CUDA_VISIBLE_DEVICES=6 nohup python train_comparison_models.py --model attention --dataset ct > "log_attention_ct_${DATE}.txt" 2>&1 &

echo "[GPU 6] Training TransUNet (CT)..."
CUDA_VISIBLE_DEVICES=6 nohup python train_comparison_models.py --model transunet --dataset ct > "log_transunet_ct_${DATE}.txt" 2>&1 &


echo "=========================================================="
echo "✅ SEMUA PROSES TELAH DIJALANKAN DI BACKGROUND!"
echo "Gunakan 'tail -f log_nama_model_ct_${DATE}.txt' untuk memantau."
echo "Atau jalankan 'watch -n 1 nvidia-smi' untuk melihat utilisasi GPU 1, 3, 6."
echo "=========================================================="
