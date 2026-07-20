#!/bin/bash
# Menjalankan Ulang Model yang Gagal (Attention UNet & TransUNet) untuk CTC

# Target GPU (Silakan ganti jika GPU 5 sedang penuh)
TARGET_GPU=5

if [ "$1" != "bg" ]; then
    DATE=$(date +"%Y%m%d_%H%M%S")
    EXP_DIR="logs/exp_ctc_retrain_${DATE}"
    mkdir -p "$EXP_DIR"
    
    echo "=========================================================="
    echo "🚀 RETRAINING FAILED MODELS (ATTENTION & TRANSUNET) [CTC]"
    echo "🎯 Target GPU: $TARGET_GPU"
    echo "📂 Folder Log: $EXP_DIR"
    echo "=========================================================="
    echo "Memulai eksekusi di background. Silakan tutup terminal jika perlu."
    echo "Pantau progres master dengan:"
    echo "tail -f ${EXP_DIR}/master_log_retrain_ctc.txt"
    
    nohup "$0" bg "$EXP_DIR" "$TARGET_GPU" > "${EXP_DIR}/master_log_retrain_ctc.txt" 2>&1 &
    exit 0
fi

EXP_DIR=$2
TARGET_GPU=$3

echo -e "\n[1/2] Retraining Attention UNet (CTC)..."
CUDA_VISIBLE_DEVICES=$TARGET_GPU python train_comparison_models.py --model attention --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_attention_ctc.txt" 2>&1
echo "✅ Attention UNet CTC done. Log: ${EXP_DIR}/log_attention_ctc.txt"

echo -e "\n[2/2] Retraining TransUNet (CTC)..."
CUDA_VISIBLE_DEVICES=$TARGET_GPU python train_comparison_models.py --model transunet --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_transunet_ctc.txt" 2>&1
echo "✅ TransUNet CTC done. Log: ${EXP_DIR}/log_transunet_ctc.txt"

echo -e "\n=========================================================="
echo "✅ RETRAIN CTC MODELS COMPLETED!"
echo "=========================================================="
