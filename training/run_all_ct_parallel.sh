#!/bin/bash
# Menjalankan 6 model (SE2 + 5 Pembanding) secara paralel pada GPU yang sedang kosong.
# Sesuai screenshot nvidia-smi terbaru: GPU 2, GPU 3, GPU 7 kosong (0%).
# Kita akan menugaskan 2 model per GPU (karena VRAM 80GB sangat mampu menampung 2 model).

DATE=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="logs/exp_${DATE}"
mkdir -p "$EXP_DIR"

echo "=========================================================="
echo "🚀 STARTING PARALLEL TRAINING FOR [CT] DATASET"
echo "✅ Menggunakan GPU yang kosong: 2, 3, 7 (Masing-masing 2 model)"
echo "=========================================================="
echo "Memulai eksekusi di background menggunakan nohup..."

# ----------------- GPU 2 -----------------
# echo "[GPU 2] Training Mod-Seg-SE(2) (CT)..."
# CUDA_VISIBLE_DEVICES=2 nohup python train_se2_by_dataset.py --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/log_se2_ct.txt" 2>&1 &

echo "[GPU 2] Training HarmonicNet (CT)..."
CUDA_VISIBLE_DEVICES=2 nohup python train_comparison_models.py --model harmonic --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/log_harmonic_ct.txt" 2>&1 &


# ----------------- GPU 3 -----------------
echo "[GPU 3] Training Standard U-Net (CT)..."
CUDA_VISIBLE_DEVICES=3 nohup python train_comparison_models.py --model unet --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/log_unet_ct.txt" 2>&1 &

echo "[GPU 3] Training nnU-Net (CT)..."
CUDA_VISIBLE_DEVICES=3 nohup python train_comparison_models.py --model nnunet --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/log_nnunet_ct.txt" 2>&1 &


# ----------------- GPU 7 -----------------
echo "[GPU 7] Training Attention U-Net (CT)..."
CUDA_VISIBLE_DEVICES=7 nohup python train_comparison_models.py --model attention --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/log_attention_ct.txt" 2>&1 &

echo "[GPU 7] Training TransUNet (CT)..."
CUDA_VISIBLE_DEVICES=7 nohup python train_comparison_models.py --model transunet --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/log_transunet_ct.txt" 2>&1 &


echo "=========================================================="
echo "✅ SEMUA PROSES TELAH DIJALANKAN DI BACKGROUND!"
echo "Gunakan 'tail -f ${EXP_DIR}/log_nama_model_ct.txt' untuk memantau."
echo "Atau jalankan 'watch -n 1 nvidia-smi' untuk melihat utilisasi GPU 2, 3, 7."
echo "=========================================================="
