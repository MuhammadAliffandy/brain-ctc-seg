#!/bin/bash
# Menjalankan 6 model (SE2 + 5 Pembanding) secara paralel pada GPU yang berbeda
# untuk dataset CT.

# Mengambil tanggal hari ini untuk penamaan log (format YYYYMMDD)
DATE=$(date +"%Y%m%d")

echo "=========================================================="
echo "🚀 STARTING PARALLEL TRAINING FOR [CT] DATASET"
echo "=========================================================="
echo "Memulai eksekusi di background menggunakan nohup..."

# GPU 1: Mod-Seg-SE(2) [Model Utama]
echo "[GPU 1] Training Mod-Seg-SE(2) (CT)..."
CUDA_VISIBLE_DEVICES=1 nohup python train_se2_by_dataset.py --dataset ct > "log_se2_ct_${DATE}.txt" 2>&1 &

# GPU 2: HarmonicNet
echo "[GPU 2] Training HarmonicNet (CT)..."
CUDA_VISIBLE_DEVICES=2 nohup python train_comparison_models.py --model harmonic --dataset ct > "log_harmonic_ct_${DATE}.txt" 2>&1 &

# GPU 3: Standard U-Net
echo "[GPU 3] Training Standard U-Net (CT)..."
CUDA_VISIBLE_DEVICES=3 nohup python train_comparison_models.py --model unet --dataset ct > "log_unet_ct_${DATE}.txt" 2>&1 &

# GPU 4: nnU-Net
echo "[GPU 4] Training nnU-Net (CT)..."
CUDA_VISIBLE_DEVICES=4 nohup python train_comparison_models.py --model nnunet --dataset ct > "log_nnunet_ct_${DATE}.txt" 2>&1 &

# GPU 5: Attention U-Net
echo "[GPU 5] Training Attention U-Net (CT)..."
CUDA_VISIBLE_DEVICES=5 nohup python train_comparison_models.py --model attention --dataset ct > "log_attention_ct_${DATE}.txt" 2>&1 &

# GPU 6: TransUNet
echo "[GPU 6] Training TransUNet (CT)..."
CUDA_VISIBLE_DEVICES=6 nohup python train_comparison_models.py --model transunet --dataset ct > "log_transunet_ct_${DATE}.txt" 2>&1 &

echo "=========================================================="
echo "✅ SEMUA PROSES TELAH DIJALANKAN DI BACKGROUND!"
echo "Gunakan 'tail -f log_nama_model_ct_${DATE}.txt' untuk memantau."
echo "Atau jalankan 'nvidia-smi' untuk melihat utilisasi GPU 1-6."
echo "=========================================================="
