#!/bin/bash
# Menjalankan 5 model pembanding secara berurutan untuk dataset CTC
# Diarahkan khusus ke GPU 3 sesuai instruksi terbaru.

# Jika script belum dijalankan di background (tanpa argumen rahasia), jalankan ulang dengan nohup
if [ "$1" != "bg" ]; then
    DATE=$(date +"%Y%m%d_%H%M%S")
    EXP_DIR="logs/exp_ctc_${DATE}"
    mkdir -p "$EXP_DIR"
    
    echo "=========================================================="
    echo "🚀 STARTING COMPARISON MODELS TRAINING FOR [CTC] DATASET"
    echo "🎯 Target GPU: 5"
    echo "📂 Folder Log: $EXP_DIR"
    echo "=========================================================="
    echo "Memulai eksekusi di background. Silakan tutup terminal jika perlu."
    echo "Gunakan perintah ini untuk memantau progres master:"
    echo "tail -f ${EXP_DIR}/master_log_ctc.txt"
    
    nohup "$0" bg "$EXP_DIR" > "${EXP_DIR}/master_log_ctc.txt" 2>&1 &
    exit 0
fi

# ================= BAGIAN BACKGROUND =================
EXP_DIR=$2

echo -e "\n[1/5] Training HarmonicNet (CTC)..."
CUDA_VISIBLE_DEVICES=5 python train_comparison_models.py --model harmonic --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_harmonic_ctc.txt" 2>&1
echo "✅ HarmonicNet CTC done. Log: ${EXP_DIR}/log_harmonic_ctc.txt"

echo -e "\n[2/5] Training Standard UNet (CTC)..."
CUDA_VISIBLE_DEVICES=5 python train_comparison_models.py --model unet --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_unet_ctc.txt" 2>&1
echo "✅ Standard UNet CTC done. Log: ${EXP_DIR}/log_unet_ctc.txt"

echo -e "\n[3/5] Training nnUNet (CTC)..."
CUDA_VISIBLE_DEVICES=5 python train_comparison_models.py --model nnunet --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_nnunet_ctc.txt" 2>&1
echo "✅ nnUNet CTC done. Log: ${EXP_DIR}/log_nnunet_ctc.txt"

echo -e "\n[4/5] Training Attention UNet (CTC)..."
CUDA_VISIBLE_DEVICES=3 python train_comparison_models.py --model attention --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_attention_ctc.txt" 2>&1
echo "✅ Attention UNet CTC done. Log: ${EXP_DIR}/log_attention_ctc.txt"

echo -e "\n[5/5] Training TransUNet (CTC)..."
CUDA_VISIBLE_DEVICES=3 python train_comparison_models.py --model transunet --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_transunet_ctc.txt" 2>&1
echo "✅ TransUNet CTC done. Log: ${EXP_DIR}/log_transunet_ctc.txt"

echo -e "\n=========================================================="
echo "✅ ALL CTC COMPARISON MODELS COMPLETED!"
echo "=========================================================="
