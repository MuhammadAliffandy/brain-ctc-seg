#!/bin/bash
# ==============================================================================
# Script: archive_dgx_experiments.sh
# Purpose: Mengumpulkan, memaketkan (.tar.gz), dan menyiapkan pemindahan
#          100% LENGKAP seluruh data komparasi model, bobot model (.pth),
#          training logs, metrik evaluasi CSV, visualisasi, dan naskah dari DGX.
# ==============================================================================

set -e

# Warna output terminal
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$PROJECT_DIR")"

# Folder tujuan penyimpanan file .tar.gz (di luar folder git agar bersih)
BACKUP_DIR="${PARENT_DIR}/backup_brain_ctc_${TIMESTAMP}"
mkdir -p "$BACKUP_DIR"

echo -e "${CYAN}======================================================================${NC}"
echo -e "${GREEN}📦 DGX EXPERIMENT ARCHIVER & EXPORT TOOL (Brain-CTC-Seg)${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Direktori Proyek : ${BLUE}${PROJECT_DIR}${NC}"
echo -e "Direktori Parent : ${BLUE}${PARENT_DIR}${NC}"
echo -e "Direktori Backup : ${YELLOW}${BACKUP_DIR}${NC}"
echo -e "Waktu Eksekusi   : ${TIMESTAMP}"
echo -e "${CYAN}======================================================================${NC}"
echo ""

cd "$PROJECT_DIR"

# ------------------------------------------------------------------------------
# 1. SCANNING ASET LENGKAP (PROJECT + PARENT FOLDER)
# ------------------------------------------------------------------------------
echo -e "${BLUE}[1/6] Memindai seluruh aset model comparison & hasil evaluasi...${NC}"

# Hitung file .pth
PTH_FILES_PROJECT=$(find . -name "*.pth" -not -path "*/kaggle_venv/*" -not -path "*/venv/*" -not -path "*/.venv/*" 2>/dev/null | wc -l)
PTH_FILES_PARENT=$(find "$PARENT_DIR" -maxdepth 2 -name "*.pth" 2>/dev/null | wc -l)
TOTAL_PTH=$((PTH_FILES_PROJECT + PTH_FILES_PARENT))

# Hitung file log
LOG_COUNT=$(find training . -name "*.log" 2>/dev/null | wc -l)

# Hitung file CSV (di proyek dan di parent ~/Clara/)
CSV_PROJECT=$(find . -name "*.csv" -not -path "*/kaggle_venv/*" 2>/dev/null | wc -l)
CSV_PARENT=$(find "$PARENT_DIR" -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l)
TOTAL_CSV=$((CSV_PROJECT + CSV_PARENT))

# Hitung visualisasi gambar
PNG_COUNT=$(find training paper_evalute Paper presentation_assets archive -name "*.png" 2>/dev/null | wc -l)

echo -e "  • Total Model Checkpoints (.pth) : ${GREEN}${TOTAL_PTH}${NC} file"
echo -e "  • Total Training Logs (.log)      : ${GREEN}${LOG_COUNT}${NC} file"
echo -e "  • Total Metrics & Tables (.csv)   : ${GREEN}${TOTAL_CSV}${NC} file"
echo -e "  • Total Figures & Curves (.png)   : ${GREEN}${PNG_COUNT}${NC} file"
echo ""

# ------------------------------------------------------------------------------
# 2. PEMAKETAN 1: SELURUH MODEL WEIGHTS (.PTH) - 6 MODELS + KFOLD + ABLATION
# ------------------------------------------------------------------------------
echo -e "${BLUE}[2/6] Memaketkan SELURUH Bobot Model (Weights .pth)...${NC}"
echo -e "      (Mod-Seg-SE(2), HarmonicNet, nnU-Net, Attention U-Net, TransUNet, Standard U-Net)"
echo -e "      (Termasuk K-Fold Folds 1-5, Ablation Variants, & Best Checkpoints)"

ARCHIVE_MODELS="${BACKUP_DIR}/01_brain_ctc_ALL_model_weights_${TIMESTAMP}.tar.gz"

MODEL_TARGETS=()

# Folder-folder model utama
[ -d "training/saved_models_25D" ] && MODEL_TARGETS+=("training/saved_models_25D")
[ -d "training/saved_models_kfold" ] && MODEL_TARGETS+=("training/saved_models_kfold")
[ -d "training/saved_models_ablation" ] && MODEL_TARGETS+=("training/saved_models_ablation")

# Cari semua folder bernama *saved_models* di dalam training/
for sm_dir in training/saved_models*; do
    if [ -d "$sm_dir" ] && [[ ! " ${MODEL_TARGETS[*]} " =~ " ${sm_dir} " ]]; then
        MODEL_TARGETS+=("$sm_dir")
    fi
done

# Cari semua file .pth di seluruh subdirektori project
while IFS= read -r pth_file; do
    clean_path="${pth_file#./}"
    # Pastikan bukan direktori dan belum tercover dalam MODEL_TARGETS
    is_in_folder=false
    for t in "${MODEL_TARGETS[@]}"; do
        if [[ "$clean_path" == "$t"* ]]; then
            is_in_folder=true
            break
        fi
    done
    if [ "$is_in_folder" = false ]; then
        MODEL_TARGETS+=("$clean_path")
    fi
done < <(find . -name "*.pth" -not -path "*/kaggle_venv/*" -not -path "*/venv/*" -not -path "*/.venv/*")

# Cari file .pth di folder parent (~/Clara) jika ada
if [ -d "$PARENT_DIR" ]; then
    while IFS= read -r p_pth; do
        if [ -f "$p_pth" ]; then
            # Copy sementara ke folder backup agar bisa dipaketkan
            mkdir -p "${BACKUP_DIR}/extra_parent_models"
            cp "$p_pth" "${BACKUP_DIR}/extra_parent_models/"
        fi
    done < <(find "$PARENT_DIR" -maxdepth 1 -name "*.pth" 2>/dev/null)
fi

if [ ${#MODEL_TARGETS[@]} -gt 0 ]; then
    echo -e "  Mengompresi ${#MODEL_TARGETS[@]} direktori/file bobot model ke: ${YELLOW}$(basename "$ARCHIVE_MODELS")${NC}"
    tar -czvf "$ARCHIVE_MODELS" \
        --exclude='__pycache__' \
        "${MODEL_TARGETS[@]}" > /dev/null
    
    # Masukkan extra parent models jika ada
    if [ -d "${BACKUP_DIR}/extra_parent_models" ]; then
        tar -rzvf "$ARCHIVE_MODELS" -C "$BACKUP_DIR" extra_parent_models > /dev/null 2>&1 || true
        rm -rf "${BACKUP_DIR}/extra_parent_models"
    fi

    SIZE_MODELS=$(du -h "$ARCHIVE_MODELS" | cut -f1)
    echo -e "  ${GREEN}✓ Selesai:${NC} $(basename "$ARCHIVE_MODELS") (${SIZE_MODELS})"
else
    echo -e "  ${YELLOW}⚠️ Catatan: Tidak ada file .pth di lokal. Pastikan dijalankan langsung di server DGX.${NC}"
fi
echo ""

# ------------------------------------------------------------------------------
# 3. PEMAKETAN 2: HASIL METRIKS, CSV, LOGS, & VISUALISASI LENGKAP
# ------------------------------------------------------------------------------
echo -e "${BLUE}[3/6] Memaketkan Hasil Evaluasi, CSV Metrics, Logs, & Kurva ROC/Learning Curves...${NC}"
ARCHIVE_RESULTS="${BACKUP_DIR}/02_brain_ctc_ALL_experiment_results_and_logs_${TIMESTAMP}.tar.gz"

RESULTS_TARGETS=()

# Salin juga file CSV evaluasi yang ada di parent ~/Clara/ (misal: comparison_eval_ct.csv, benchmarking_results.csv)
TEMP_PARENT_CSV_DIR="${BACKUP_DIR}/parent_evaluation_csvs"
mkdir -p "$TEMP_PARENT_CSV_DIR"
for p_csv in "$PARENT_DIR"/*.csv; do
    if [ -f "$p_csv" ]; then
        cp "$p_csv" "$TEMP_PARENT_CSV_DIR/"
    fi
done

# Folder-folder hasil evaluasi
[ -d "paper_evalute" ] && RESULTS_TARGETS+=("paper_evalute")
[ -d "data_summaries" ] && RESULTS_TARGETS+=("data_summaries")
[ -d "public_dataset" ] && RESULTS_TARGETS+=("public_dataset")
[ -d "archive" ] && RESULTS_TARGETS+=("archive")
[ -d "Paper/Journal_Figures" ] && RESULTS_TARGETS+=("Paper/Journal_Figures")
[ -d "training/logs_ablation" ] && RESULTS_TARGETS+=("training/logs_ablation")

# Kumpulkan semua file .log, .csv, dan .png di folder training/
while IFS= read -r f; do
    RESULTS_TARGETS+=("${f#./}")
done < <(find training -maxdepth 3 \( -name "*.log" -o -name "*.csv" -o -name "*.png" \) 2>/dev/null)

if [ ${#RESULTS_TARGETS[@]} -gt 0 ]; then
    echo -e "  Mengompresi hasil eksperimen ke: ${YELLOW}$(basename "$ARCHIVE_RESULTS")${NC}"
    tar -czvf "$ARCHIVE_RESULTS" \
        --exclude='__pycache__' \
        "${RESULTS_TARGETS[@]}" > /dev/null

    # Tambahkan parent CSVs jika ada
    if [ "$(ls -A "$TEMP_PARENT_CSV_DIR" 2>/dev/null)" ]; then
        tar -rf "${ARCHIVE_RESULTS%.gz}" -C "$BACKUP_DIR" parent_evaluation_csvs 2>/dev/null || true
        # Gzip ulang jika tar di-append
        gzip -f "${ARCHIVE_RESULTS%.gz}" 2>/dev/null || true
    fi
    rm -rf "$TEMP_PARENT_CSV_DIR"

    SIZE_RESULTS=$(du -h "$ARCHIVE_RESULTS" | cut -f1)
    echo -e "  ${GREEN}✓ Selesai:${NC} $(basename "$ARCHIVE_RESULTS") (${SIZE_RESULTS})"
fi
echo ""

# ------------------------------------------------------------------------------
# 4. PEMAKETAN 3: MANUSCRIPT, PRESENTASI PPTX, LAPORAN DOCX
# ------------------------------------------------------------------------------
echo -e "${BLUE}[4/6] Memaketkan Dokumen Naskah Jurnal, Laporan, & Presentasi Klien...${NC}"
ARCHIVE_DOCS="${BACKUP_DIR}/03_brain_ctc_manuscripts_and_reports_${TIMESTAMP}.tar.gz"

DOC_TARGETS=()
[ -d "Paper" ] && DOC_TARGETS+=("Paper")
[ -d "presentation_assets" ] && DOC_TARGETS+=("presentation_assets")
[ -f "Manuscript_V3_tracked.docx" ] && DOC_TARGETS+=("Manuscript_V3_tracked.docx")
[ -f "Brain_Presentation.pptx" ] && DOC_TARGETS+=("Brain_Presentation.pptx")
[ -f "Brain_CT_Seg_Client_Presentation.pptx" ] && DOC_TARGETS+=("Brain_CT_Seg_Client_Presentation.pptx")
[ -f "CT_SE2_Brain_Architecture_Specification.docx" ] && DOC_TARGETS+=("CT_SE2_Brain_Architecture_Specification.docx")
[ -f "MANUSCRIPT_TODO.md" ] && DOC_TARGETS+=("MANUSCRIPT_TODO.md")
[ -f "presentation_preview.html" ] && DOC_TARGETS+=("presentation_preview.html")

if [ ${#DOC_TARGETS[@]} -gt 0 ]; then
    tar -czvf "$ARCHIVE_DOCS" \
        --exclude='__pycache__' \
        --exclude='Paper/Client_GIFs_Final_25D' \
        "${DOC_TARGETS[@]}" > /dev/null
    SIZE_DOCS=$(du -h "$ARCHIVE_DOCS" | cut -f1)
    echo -e "  ${GREEN}✓ Selesai:${NC} $(basename "$ARCHIVE_DOCS") (${SIZE_DOCS})"
fi
echo ""

# ------------------------------------------------------------------------------
# 5. PEMAKETAN 4: KODE LENGKAP & INTERACTIVE WEB DASHBOARD
# ------------------------------------------------------------------------------
echo -e "${BLUE}[5/6] Memaketkan Source Code Lengkap & Interactive Dashboard...${NC}"
ARCHIVE_CODE="${BACKUP_DIR}/04_brain_ctc_codebase_and_web_${TIMESTAMP}.tar.gz"

tar -czvf "$ARCHIVE_CODE" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pth' \
    --exclude='*.tar.gz' \
    --exclude='*.zip' \
    --exclude='kaggle_venv' \
    --exclude='venv' \
    --exclude='.venv' \
    --exclude='public_dataset/data' \
    --exclude='Paper/Client_GIFs_Final_25D' \
    . > /dev/null
SIZE_CODE=$(du -h "$ARCHIVE_CODE" | cut -f1)
echo -e "  ${GREEN}✓ Selesai:${NC} $(basename "$ARCHIVE_CODE") (${SIZE_CODE})"
echo ""

# ------------------------------------------------------------------------------
# 6. PEMAKETAN 5 (BONUS ALL-IN-ONE MASTER BUNDLE)
# ------------------------------------------------------------------------------
echo -e "${BLUE}[6/6] Membuat 1 Master Bundle Gabungan (ALL-IN-ONE)...${NC}"
ARCHIVE_MASTER="${BACKUP_DIR}/00_BRAIN_CTC_SEG_COMPLETE_MASTER_EXPERIMENTS_${TIMESTAMP}.tar.gz"

# Buat tar gabungan dari folder backup
tar -czvf "$ARCHIVE_MASTER" \
    -C "$BACKUP_DIR" \
    "$(basename "$ARCHIVE_MODELS")" \
    "$(basename "$ARCHIVE_RESULTS")" \
    "$(basename "$ARCHIVE_DOCS")" \
    "$(basename "$ARCHIVE_CODE")" > /dev/null 2>&1 || true

SIZE_MASTER=$(du -h "$ARCHIVE_MASTER" 2>/dev/null | cut -f1 || echo "N/A")
echo -e "  ${GREEN}✓ Master Bundle Selesai:${NC} $(basename "$ARCHIVE_MASTER") (${SIZE_MASTER})"
echo ""

# ------------------------------------------------------------------------------
# CHECKSUM SHA256 VERIFICATION & MANIFEST
# ------------------------------------------------------------------------------
echo -e "${CYAN}======================================================================${NC}"
echo -e "${BLUE}Menghitung Checksum SHA256 untuk verifikasi integritas data...${NC}"
cd "$BACKUP_DIR"
sha256sum *.tar.gz > checksums_sha256.txt
cat checksums_sha256.txt
echo -e "${GREEN}✓ Checksum tersimpan di: ${BACKUP_DIR}/checksums_sha256.txt${NC}"
echo ""

# Buat file ringkasan Manifest detail
cat <<EOF > "${BACKUP_DIR}/BACKUP_MANIFEST.txt"
================================================================================
BRAIN-CTC-SEG COMPLETE EXPERIMENT BACKUP MANIFEST
Generated at : $(date)
Server Host  : $(hostname)
Source Path  : ${PROJECT_DIR}
Parent Path  : ${PARENT_DIR}
================================================================================

PAKET BACKUP YANG DIHASILKAN:
1. 00_BRAIN_CTC_SEG_COMPLETE_MASTER_EXPERIMENTS_${TIMESTAMP}.tar.gz
   -> [ALL-IN-ONE MASTER BUNDLE]: Gabungan seluruh paket 1, 2, 3, dan 4 dalam satu file.

2. 01_brain_ctc_ALL_model_weights_${TIMESTAMP}.tar.gz
   -> SELURUH BOBOT MODEL (.pth):
      • Mod-Seg-SE(2) [OURS - N=8 Equivariant R2Conv] (CT, CTC, All datasets)
      • HarmonicNet (C4)
      • nnU-Net (SOTA Medical Baseline)
      • Attention U-Net
      • TransUNet (Transformer Bottleneck)
      • Standard U-Net
      • 5-Fold Cross Validation Weights (Fold 1 - 5)
      • Ablation Study Checkpoints (*_best.pth, *_latest.pth)
      • Epoch Checkpoints (epoch 100, boundary best)

3. 02_brain_ctc_ALL_experiment_results_and_logs_${TIMESTAMP}.tar.gz
   -> SELURUH HASIL EVALUASI, LOGS, & CSV METRICS:
      • master_kfold_results.csv (Hasil 5-fold semua model)
      • comparison_eval_ct.csv & comparison_eval_ctc.csv
      • benchmarking_results.csv & full_benchmark_results.csv
      • ablation_metrics.csv & lesion_tracking_metrics.csv
      • Seluruh training logs (*.log)
      • Gambar kurva ROC multi-panel, comparative heatmaps, learning curves, GradCAM

4. 03_brain_ctc_manuscripts_and_reports_${TIMESTAMP}.tar.gz
   -> DOKUMEN NASKAH & LAPORAN:
      • Manuscript_V3_tracked.docx (Naskah publikasi tracked changes)
      • Folder Paper/ (LaTeX source, jurnal figures, PDF rujukan)
      • Brain_CT_Seg_Client_Presentation.pptx & Brain_Presentation.pptx
      • CT_SE2_Brain_Architecture_Specification.docx

5. 04_brain_ctc_codebase_and_web_${TIMESTAMP}.tar.gz
   -> KODE SOURCE & WEB DASHBOARD:
      • app.py (Multi-Model Clinical Decision Support Web App)
      • Pipeline training & evaluasi lengkap
      • evaluation web/ (CT-RAD interactive prototype package)

TOTAL UKURAN FOLDER BACKUP:
$(du -sh "${BACKUP_DIR}")
================================================================================
EOF

echo -e "${CYAN}======================================================================${NC}"
echo -e "${GREEN}🎉 100% SELURUH KOMPARASI MODEL, HASIL, & BOBOT BERHASIL DIAMANKAN!${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Direktori Arsip: ${YELLOW}${BACKUP_DIR}${NC}"
ls -lh "$BACKUP_DIR"
echo ""

# ------------------------------------------------------------------------------
# PANDUAN PEMINDAHAN DATA (RCLONE & SCP/RSYNC)
# ------------------------------------------------------------------------------
echo -e "${YELLOW}======================================================================${NC}"
echo -e "${YELLOW}CARA MEMINDAHKAN DATA INI KE GOOGLE DRIVE / LAPTOP:${NC}"
echo -e "${YELLOW}======================================================================${NC}"
echo ""
echo -e "${GREEN}OPSI A (OTOMATIS): Upload ke Google Drive (clara_drive:backup-brain-ctc-seg)${NC}"
echo -e "Jalankan perintah ini di server DGX (atau skrip terpisah):"
echo -e "  ${CYAN}./transfer_backup_to_drive.sh ${BACKUP_DIR}${NC}"
echo ""
echo -e "Perintah manual rclone jika diinginkan:"
echo -e "  ${CYAN}rclone mkdir clara_drive:backup-brain-ctc-seg${NC}"
echo -e "  ${CYAN}rclone copy ${BACKUP_DIR}/ clara_drive:backup-brain-ctc-seg/ -P --transfers 4 --checkers 8${NC}"
echo ""
echo -e "${GREEN}OPSI B: Download Langsung ke Laptop Lokal via rsync / scp${NC}"
echo -e "Jalankan di terminal laptop Anda:"
echo -e "  ${CYAN}rsync -avzP <USER>@<DGX_IP>:${BACKUP_DIR}/ ./backup_brain_ctc_${TIMESTAMP}/${NC}"
echo -e "  atau cukup download file Master All-in-One:"
echo -e "  ${CYAN}scp <USER>@<DGX_IP>:${ARCHIVE_MASTER} ./local_backup/${NC}"
echo ""
echo -e "${GREEN}OPSI C: Validasi Integritas File di Tempat Tujuan${NC}"
echo -e "  ${CYAN}cd ./backup_brain_ctc_${TIMESTAMP} && sha256sum -c checksums_sha256.txt${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo ""

# Opsi interaktif jika terminal dijalankan oleh user secara langsung
if [ -t 0 ]; then
    read -p "Apakah Anda ingin langsung membuat folder 'backup-brain-ctc-seg' dan mengunggah ke Google Drive sekarang? [y/N]: " run_upload
    case "$run_upload" in
        [yY][eE][sS]|[yY])
            echo ""
            "$PROJECT_DIR/transfer_backup_to_drive.sh" "$BACKUP_DIR"
            ;;
        *)
            echo -e "${BLUE}Upload dilewati. Anda dapat menjalankannya nanti dengan: ${CYAN}./transfer_backup_to_drive.sh${NC}"
            ;;
    esac
fi
