#!/bin/bash
# Menjalankan 6 model K-FOLD secara paralel berdasarkan GPU yang sedang kosong.
# Sesuai screenshot nvidia-smi: GPU 1, 3, 6, 7 kosong (0%), GPU 0 hampir kosong (6%).
# Karena ada 6 model dan 5 GPU kosong, 1 GPU akan menampung 2 model (sangat aman karena VRAM 80GB).

# Cek argument dataset
if [ -z "$1" ]; then
    echo "❌ Error: Harap masukkan dataset (ct atau ctc)!"
    echo "Penggunaan: ./run_all_kfold_parallel.sh ct"
    exit 1
fi

DATASET=$1
FOLDS=5
DATE=$(date +"%Y%m%d")

echo "=========================================================="
echo "🚀 STARTING PARALLEL K-FOLD TRAINING FOR [$DATASET]"
echo "=========================================================="

# GPU 1: Mod-Seg-SE(2) [Model Utama]
echo "[GPU 1] K-Fold Mod-Seg-SE(2)..."
CUDA_VISIBLE_DEVICES=1 nohup python train_kfold.py --model se2 --dataset $DATASET --folds $FOLDS > "log_kfold_se2_${DATASET}_${DATE}.txt" 2>&1 &

# GPU 3: HarmonicNet
echo "[GPU 3] K-Fold HarmonicNet..."
CUDA_VISIBLE_DEVICES=3 nohup python train_kfold.py --model harmonic --dataset $DATASET --folds $FOLDS > "log_kfold_harmonic_${DATASET}_${DATE}.txt" 2>&1 &

# GPU 6: Standard U-Net
echo "[GPU 6] K-Fold Standard U-Net..."
CUDA_VISIBLE_DEVICES=6 nohup python train_kfold.py --model unet --dataset $DATASET --folds $FOLDS > "log_kfold_unet_${DATASET}_${DATE}.txt" 2>&1 &

# GPU 7: nnU-Net
echo "[GPU 7] K-Fold nnU-Net..."
CUDA_VISIBLE_DEVICES=7 nohup python train_kfold.py --model nnunet --dataset $DATASET --folds $FOLDS > "log_kfold_nnunet_${DATASET}_${DATE}.txt" 2>&1 &

# GPU 0: Attention U-Net
echo "[GPU 0] K-Fold Attention U-Net..."
CUDA_VISIBLE_DEVICES=0 nohup python train_kfold.py --model attention --dataset $DATASET --folds $FOLDS > "log_kfold_attention_${DATASET}_${DATE}.txt" 2>&1 &

# GPU 6: TransUNet (Gabung di GPU 6 karena U-Net sangat ringan memakan VRAM)
echo "[GPU 6] K-Fold TransUNet..."
CUDA_VISIBLE_DEVICES=6 nohup python train_kfold.py --model transunet --dataset $DATASET --folds $FOLDS > "log_kfold_transunet_${DATASET}_${DATE}.txt" 2>&1 &

echo "=========================================================="
echo "✅ SEMUA K-FOLD TELAH DIJALANKAN DI BACKGROUND!"
echo "Gunakan 'tail -f log_kfold_nama_model_${DATASET}_${DATE}.txt' untuk memantau."
echo "=========================================================="
