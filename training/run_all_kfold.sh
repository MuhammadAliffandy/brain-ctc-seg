#!/bin/bash

# Pastikan dijalankan dari dalam folder training
cd ~/Clara/brain-ctc-seg/training

echo "=========================================================="
echo "🚀 MEMULAI K-FOLD CROSS VALIDATION UNTUK SEMUA MODEL"
echo "=========================================================="

# Cek argument dataset
if [ -z "$1" ]; then
    echo "❌ Error: Harap masukkan dataset (ct atau ctc)!"
    echo "Penggunaan: ./run_all_kfold.sh ct"
    exit 1
fi

DATASET=$1
FOLDS=5

DATE=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="logs/kfold_exp_${DATE}"
mkdir -p "$EXP_DIR"

# Array model yang akan ditraining
MODELS=("se2" "harmonic" "unet" "nnunet" "attention" "transunet")

for model in "${MODELS[@]}"
do
    echo "▶️ Memulai 5-Fold untuk Model: $model"
    # Menjalankan python script, dan memasukkan output log ke dalam folder experiment
    nohup python train_kfold.py --model "$model" --dataset "$DATASET" --folds "$FOLDS" > "${EXP_DIR}/log_kfold_${model}_${DATASET}.txt" 2>&1 &
    echo "✅ Selesai K-Fold untuk Model: $model"
    echo "----------------------------------------------------------"
done

echo "🎉 SEMUA K-FOLD TRAINING SELESAI!"
echo "Silakan cek file: ~/Clara/brain-ctc-seg/training/saved_models_kfold/master_kfold_results.csv"
