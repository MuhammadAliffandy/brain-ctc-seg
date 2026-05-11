#!/bin/bash

# Pastikan dijalankan dari dalam folder training
cd ~/Clara/brain-ctc-seg/training

echo "=========================================================="
echo "🚀 MEMULAI K-FOLD CROSS VALIDATION UNTUK SEMUA MODEL"
echo "=========================================================="

# Kita set default K=5, dataset=ctc (bisa diubah sesuai kebutuhan)
DATASET="ctc"
FOLDS=5

# Array model yang akan ditraining
MODELS=("se2" "harmonic" "unet" "nnunet" "attention" "transunet")

for model in "${MODELS[@]}"
do
    echo "▶️ Memulai 5-Fold untuk Model: $model"
    # Menjalankan python script, membiarkan output tampil di terminal 
    # (nanti Anda bisa pipe ke file menggunakan nohup saat menjalankan bash ini)
    python train_kfold.py --model "$model" --dataset "$DATASET" --folds "$FOLDS"
    echo "✅ Selesai K-Fold untuk Model: $model"
    echo "----------------------------------------------------------"
done

echo "🎉 SEMUA K-FOLD TRAINING SELESAI!"
echo "Silakan cek file: ~/Clara/brain-ctc-seg/training/saved_models_kfold/master_kfold_results.csv"
