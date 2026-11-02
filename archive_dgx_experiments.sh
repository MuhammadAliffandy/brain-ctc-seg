#!/bin/bash
# ==============================================================================
# Script: archive_dgx_experiments.sh
# Purpose: Mengumpulkan, memaketkan (.tar.gz), dan menyiapkan pemindahan
#          seluruh data penting (Model Weights .pth, Training Logs, Hasil Evaluasi
#          CSV/Figures, Manuscript, dan Demo) dari server DGX.
# ==============================================================================

set -e

# Warna output terminal
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Folder tujuan penyimpanan file .tar.gz (di luar folder git agar bersih)
PARENT_DIR="$(dirname "$PROJECT_DIR")"
BACKUP_DIR="${PARENT_DIR}/backup_brain_ctc_${TIMESTAMP}"

mkdir -p "$BACKUP_DIR"

echo -e "${CYAN}======================================================================${NC}"
echo -e "${GREEN}📦 DGX EXPERIMENT ARCHIVER & EXPORT TOOL (Brain-CTC-Seg)${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Direktori Proyek : ${BLUE}${PROJECT_DIR}${NC}"
echo -e "Direktori Backup : ${YELLOW}${BACKUP_DIR}${NC}"
echo -e "Waktu Eksekusi   : ${TIMESTAMP}"
echo ""

cd "$PROJECT_DIR"

# ------------------------------------------------------------------------------
# 1. SCANNING ASET PENTING
# ------------------------------------------------------------------------------
echo -e "${BLUE}[1/5] Memeriksa ketersediaan aset penting...${NC}"

PTH_COUNT=$(find . -name "*.pth" | wc -l)
LOG_COUNT=$(find . -name "*.log" | wc -l)
CSV_COUNT=$(find . -name "*.csv" | wc -l)
PNG_COUNT=$(find training paper_evalute Paper presentation_assets -name "*.png" 2>/dev/null | wc -l)

echo -e "  • Model Checkpoints (.pth) : ${GREEN}${PTH_COUNT}${NC} file ditemukan"
echo -e "  • Training Logs (.log)      : ${GREEN}${LOG_COUNT}${NC} file ditemukan"
echo -e "  • Evaluation CSV (.csv)     : ${GREEN}${CSV_COUNT}${NC} file ditemukan"
echo -e "  • Hasil Gambar/Kurva (.png) : ${GREEN}${PNG_COUNT}${NC} file ditemukan"
echo ""

# ------------------------------------------------------------------------------
# 2. PEMAKETAN 1: MODEL CHECKPOINTS (.PTH) - SANGAT KRUSIAL
# ------------------------------------------------------------------------------
echo -e "${BLUE}[2/5] Memaketkan Model Weights & Checkpoints (.pth)...${NC}"
ARCHIVE_MODELS="${BACKUP_DIR}/01_brain_ctc_model_weights_${TIMESTAMP}.tar.gz"

# Cari folder saved_models_25D dan file .pth lainnya
MODEL_TARGETS=()
if [ -d "training/saved_models_25D" ]; then
    MODEL_TARGETS+=("training/saved_models_25D")
fi

# Cari file .pth di folder training/ atau root
while IFS= read -r pth_file; do
    if [[ ! " ${MODEL_TARGETS[*]} " =~ " ${pth_file} " ]]; then
        MODEL_TARGETS+=("${pth_file#./}")
    fi
done < <(find . -maxdepth 3 -name "*.pth" -not -path "*/kaggle_venv/*" -not -path "*/venv/*" -not -path "*/.venv/*")

if [ ${#MODEL_TARGETS[@]} -gt 0 ]; then
    echo -e "  Mengompresi ${#MODEL_TARGETS[@]} target model weights ke: ${YELLOW}$(basename "$ARCHIVE_MODELS")${NC}"
    tar -czvf "$ARCHIVE_MODELS" \
        --exclude='__pycache__' \
        "${MODEL_TARGETS[@]}" > /dev/null
    SIZE_MODELS=$(du -h "$ARCHIVE_MODELS" | cut -f1)
    echo -e "  ${GREEN}✓ Selesai:${NC} $(basename "$ARCHIVE_MODELS") (${SIZE_MODELS})"
else
    echo -e "  ${YELLOW}⚠️ Tidak ada file .pth yang ditemukan di direktori lokal.${NC}"
fi
echo ""

# ------------------------------------------------------------------------------
# 3. PEMAKETAN 2: HASIL EXPERIMENT, LOGS, CSV, & FIGURE VISUALISASI
# ------------------------------------------------------------------------------
echo -e "${BLUE}[3/5] Memaketkan Hasil Eksperimen, Logs, & Publikasi Metrics...${NC}"
ARCHIVE_RESULTS="${BACKUP_DIR}/02_brain_ctc_experiment_results_and_logs_${TIMESTAMP}.tar.gz"

RESULTS_TARGETS=()
[ -d "paper_evalute" ] && RESULTS_TARGETS+=("paper_evalute")
[ -d "data_summaries" ] && RESULTS_TARGETS+=("data_summaries")
[ -d "Paper/Journal_Figures" ] && RESULTS_TARGETS+=("Paper/Journal_Figures")
[ -d "training/logs_ablation" ] && RESULTS_TARGETS+=("training/logs_ablation")

# Kumpulkan semua file .log dan .csv di folder training/
while IFS= read -r f; do
    RESULTS_TARGETS+=("${f#./}")
done < <(find training -maxdepth 2 \( -name "*.log" -o -name "*.csv" -o -name "*.png" \) 2>/dev/null)

if [ ${#RESULTS_TARGETS[@]} -gt 0 ]; then
    echo -e "  Mengompresi logs, CSV metrics, dan grafik ke: ${YELLOW}$(basename "$ARCHIVE_RESULTS")${NC}"
    tar -czvf "$ARCHIVE_RESULTS" \
        --exclude='__pycache__' \
        "${RESULTS_TARGETS[@]}" > /dev/null
    SIZE_RESULTS=$(du -h "$ARCHIVE_RESULTS" | cut -f1)
    echo -e "  ${GREEN}✓ Selesai:${NC} $(basename "$ARCHIVE_RESULTS") (${SIZE_RESULTS})"
else
    echo -e "  ${YELLOW}⚠️ Target hasil evaluasi tidak ditemukan.${NC}"
fi
echo ""

# ------------------------------------------------------------------------------
# 4. PEMAKETAN 3: MANUSCRIPT, PRESENTASI PPTX, LAPORAN DOCX
# ------------------------------------------------------------------------------
echo -e "${BLUE}[4/5] Memaketkan Dokumen Naskah (Manuscript, PPTX, Docx)...${NC}"
ARCHIVE_DOCS="${BACKUP_DIR}/03_brain_ctc_manuscripts_and_reports_${TIMESTAMP}.tar.gz"

DOC_TARGETS=()
[ -d "Paper" ] && DOC_TARGETS+=("Paper")
[ -d "presentation_assets" ] && DOC_TARGETS+=("presentation_assets")
[ -f "Manuscript_V3_tracked.docx" ] && DOC_TARGETS+=("Manuscript_V3_tracked.docx")
[ -f "Brain_Presentation.pptx" ] && DOC_TARGETS+=("Brain_Presentation.pptx")
[ -f "Brain_CT_Seg_Client_Presentation.pptx" ] && DOC_TARGETS+=("Brain_CT_Seg_Client_Presentation.pptx")
[ -f "CT_SE2_Brain_Architecture_Specification.docx" ] && DOC_TARGETS+=("CT_SE2_Brain_Architecture_Specification.docx")
[ -f "MANUSCRIPT_TODO.md" ] && DOC_TARGETS+=("MANUSCRIPT_TODO.md")

if [ ${#DOC_TARGETS[@]} -gt 0 ]; then
    echo -e "  Mengompresi dokumen dan presentasi ke: ${YELLOW}$(basename "$ARCHIVE_DOCS")${NC}"
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
echo -e "${BLUE}[5/5] Memaketkan Source Code & Interactive Web Application...${NC}"
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
# 6. CHECKSUM SHA256 VERIFICATION
# ------------------------------------------------------------------------------
echo -e "${CYAN}======================================================================${NC}"
echo -e "${BLUE}Menghitung Checksum SHA256 untuk verifikasi integritas data...${NC}"
cd "$BACKUP_DIR"
sha256sum *.tar.gz > checksums_sha256.txt
cat checksums_sha256.txt
echo -e "${GREEN}✓ Checksum tersimpan di: ${BACKUP_DIR}/checksums_sha256.txt${NC}"
echo ""

# Buat file ringkasan Manifest
cat <<EOF > "${BACKUP_DIR}/BACKUP_MANIFEST.txt"
================================================================================
BRAIN-CTC-SEG EXPERIMENT BACKUP MANIFEST
Generated at : $(date)
Server Host  : $(hostname)
Source Path  : ${PROJECT_DIR}
================================================================================

PAKET BACKUP YANG DIHASILKAN:
1. 01_brain_ctc_model_weights_${TIMESTAMP}.tar.gz
   -> Berisi semua bobot model (.pth): Mod-Seg-SE(2), HarmonicNet, nnU-Net, Standard U-Net, Attention U-Net, TransUNet, K-Fold checkpoints.
2. 02_brain_ctc_experiment_results_and_logs_${TIMESTAMP}.tar.gz
   -> Berisi training logs (*.log), metric CSV (Dice, IoU, HD95), dan kurva ROC / learning curve PNG.
3. 03_brain_ctc_manuscripts_and_reports_${TIMESTAMP}.tar.gz
   -> Berisi dokumen Word tracked manuscript, presentasi PPTX, spesifikasi arsitektur.
4. 04_brain_ctc_codebase_and_web_${TIMESTAMP}.tar.gz
   -> Berisi kode lengkap pipeline training, evaluasi, dan dashboard CT-RAD streamlit.

TOTAL UKURAN:
$(du -sh "${BACKUP_DIR}")
================================================================================
EOF

echo -e "${CYAN}======================================================================${NC}"
echo -e "${GREEN}🎉 PROSES PEMAKETAN SELESAI DENGAN SUKSES!${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Semua arsip tersimpan rapi di: ${YELLOW}${BACKUP_DIR}${NC}"
ls -lh "$BACKUP_DIR"
echo ""

# ------------------------------------------------------------------------------
# PANDUAN PEMINDAHAN DATA (RCLONE & SCP/RSYNC)
# ------------------------------------------------------------------------------
echo -e "${YELLOW}CARA MEMINDAHKAN HASIL BACKUP INI:${NC}"
echo ""
echo -e "${GREEN}OPSI 1: Upload Langsung ke Google Drive via rclone (Paling Cepat dari Server)${NC}"
echo -e "Jalankan perintah ini di server:"
echo -e "  ${CYAN}rclone copy ${BACKUP_DIR}/ clara_drive:Backup_Brain_CTC_Seg_${TIMESTAMP}/ -P${NC}"
echo ""
echo -e "${GREEN}OPSI 2: Download Langsung ke Laptop/PC Lokal (via SCP / Rsync)${NC}"
echo -e "Jalankan perintah ini di terminal Laptop/PC Anda (bukan di server):"
echo -e "  ${CYAN}rsync -avzP <USER>@<DGX_IP>:${BACKUP_DIR}/ ./backup_brain_ctc_${TIMESTAMP}/${NC}"
echo -e "  atau:"
echo -e "  ${CYAN}scp -r <USER>@<DGX_IP>:${BACKUP_DIR} ./local_backup/${NC}"
echo ""
echo -e "${GREEN}OPSI 3: Verifikasi Integritas File Setelah Didownload${NC}"
echo -e "Di komputer tujuan, jalankan:"
echo -e "  ${CYAN}cd ./backup_brain_ctc_${TIMESTAMP} && sha256sum -c checksums_sha256.txt${NC}"
echo -e "${CYAN}======================================================================${NC}"
