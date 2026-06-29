#!/bin/bash
# Menjalankan 6 model (SE2 + 5 Pembanding) secara paralel pada GPU yang berbeda
# untuk dataset CTC.

# Mengambil tanggal hari ini untuk penamaan log (format YYYYMMDD)
DATE=$(date +"%Y%m%d")

echo "=========================================================="
echo "🚀 STARTING PARALLEL TRAINING FOR [CTC] DATASET"
echo "=========================================================="
echo "Memulai eksekusi di background menggunakan nohup..."

# GPU 1: Mod-Seg-SE(2) [Model Utama]
echo "[GPU 1] Training Mod-Seg-SE(2) (CTC)..."
CUDA_VISIBLE_DEVICES=1 nohup python train_se2_by_dataset.py --dataset ctc > "log_se2_ctc_${DATE}.txt" 2>&1 &

# GPU 2: HarmonicNet
echo "[GPU 2] Training HarmonicNet (CTC)..."
CUDA_VISIBLE_DEVICES=2 nohup python train_comparison_models.py --model harmonic --dataset ctc > "log_harmonic_ctc_${DATE}.txt" 2>&1 &

# GPU 3: Standard U-Net
echo "[GPU 3] Training Standard U-Net (CTC)..."
CUDA_VISIBLE_DEVICES=3 nohup python train_comparison_models.py --model unet --dataset ctc > "log_unet_ctc_${DATE}.txt" 2>&1 &

# GPU 4: nnU-Net
echo "[GPU 4] Training nnU-Net (CTC)..."
CUDA_VISIBLE_DEVICES=4 nohup python train_comparison_models.py --model nnunet --dataset ctc > "log_nnunet_ctc_${DATE}.txt" 2>&1 &

# GPU 0: Attention U-Net
echo "[GPU 0] Training Attention U-Net (CTC)..."
CUDA_VISIBLE_DEVICES=0 nohup python train_comparison_models.py --model attention --dataset ctc > "log_attention_ctc_${DATE}.txt" 2>&1 &

# GPU 6: TransUNet
echo "[GPU 6] Training TransUNet (CTC)..."
CUDA_VISIBLE_DEVICES=6 nohup python train_comparison_models.py --model transunet --dataset ctc > "log_transunet_ctc_${DATE}.txt" 2>&1 &

echo "=========================================================="
echo "✅ SEMUA PROSES TELAH DIJALANKAN DI BACKGROUND!"
echo "Gunakan 'tail -f log_nama_model_ctc_${DATE}.txt' untuk memantau."
echo "Atau jalankan 'nvidia-smi' untuk melihat utilisasi GPU 1-6."
echo "=========================================================="
