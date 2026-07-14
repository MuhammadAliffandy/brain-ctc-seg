#!/bin/bash
# =====================================================================
# MASTER SCRIPT: RETRAIN ALL MODELS (12 JOBS ACROSS 6 GPUs)
# =====================================================================
# Skrip ini dirancang untuk mendistribusikan 12 proses training ke 
# 6 GPU (GPU 2, 3, 4, 5, 6, 7) agar tidak terjadi tabrakan CUDA_OOM.
# Setiap GPU akan mengeksekusi 1 model CT, lalu dilanjutkan 1 model CTC.

DATE=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="logs/exp_retrain_${DATE}"
mkdir -p "$EXP_DIR"

echo "=========================================================="
echo "🚀 STARTING MASTER RETRAIN (12 MODELS) ACROSS 6 GPUs"
echo "📂 Log directory: $EXP_DIR"
echo "=========================================================="

# ----------------- GPU 2: Mod-Seg-SE(2) -----------------
echo "[GPU 2] Queueing SE2 (CT) -> SE2 (CTC)..."
(
    CUDA_VISIBLE_DEVICES=2 python train_se2_by_dataset.py --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/log_se2_ct.txt" 2>&1
    CUDA_VISIBLE_DEVICES=2 python train_se2_by_dataset.py --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_se2_ctc.txt" 2>&1
) &

# ----------------- GPU 3: HarmonicNet -----------------
echo "[GPU 3] Queueing HarmonicNet (CT) -> HarmonicNet (CTC)..."
(
    CUDA_VISIBLE_DEVICES=3 python train_comparison_models.py --model harmonic --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/log_harmonic_ct.txt" 2>&1
    CUDA_VISIBLE_DEVICES=3 python train_comparison_models.py --model harmonic --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_harmonic_ctc.txt" 2>&1
) &

# ----------------- GPU 4: Standard U-Net -----------------
echo "[GPU 4] Queueing Standard U-Net (CT) -> Standard U-Net (CTC)..."
(
    CUDA_VISIBLE_DEVICES=4 python train_comparison_models.py --model unet --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/log_unet_ct.txt" 2>&1
    CUDA_VISIBLE_DEVICES=4 python train_comparison_models.py --model unet --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_unet_ctc.txt" 2>&1
) &

# ----------------- GPU 5: nnU-Net -----------------
echo "[GPU 5] Queueing nnU-Net (CT) -> nnU-Net (CTC)..."
(
    CUDA_VISIBLE_DEVICES=5 python train_comparison_models.py --model nnunet --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/log_nnunet_ct.txt" 2>&1
    CUDA_VISIBLE_DEVICES=5 python train_comparison_models.py --model nnunet --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_nnunet_ctc.txt" 2>&1
) &

# ----------------- GPU 6: Attention U-Net -----------------
echo "[GPU 6] Queueing Attention U-Net (CT) -> Attention U-Net (CTC)..."
(
    CUDA_VISIBLE_DEVICES=6 python train_comparison_models.py --model attention --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/log_attention_ct.txt" 2>&1
    CUDA_VISIBLE_DEVICES=6 python train_comparison_models.py --model attention --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_attention_ctc.txt" 2>&1
) &

# ----------------- GPU 7: TransUNet -----------------
echo "[GPU 7] Queueing TransUNet (CT) -> TransUNet (CTC)..."
(
    CUDA_VISIBLE_DEVICES=7 python train_comparison_models.py --model transunet --dataset ct --log_dir "$EXP_DIR" > "${EXP_DIR}/log_transunet_ct.txt" 2>&1
    CUDA_VISIBLE_DEVICES=7 python train_comparison_models.py --model transunet --dataset ctc --log_dir "$EXP_DIR" > "${EXP_DIR}/log_transunet_ctc.txt" 2>&1
) &

echo "=========================================================="
echo "✅ SEMUA 12 PROSES TELAH MASUK ANTREAN BACKGROUND!"
echo "Gunakan 'tail -f ${EXP_DIR}/log_se2_ct.txt' untuk memantau progress."
echo "Tinggalkan terminal ini dan biarkan server bekerja semalaman!"
echo "=========================================================="
