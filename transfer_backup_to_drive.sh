#!/bin/bash
# ==============================================================================
# Script: transfer_backup_to_drive.sh
# Purpose: Membuat folder 'backup-brain-ctc-seg' di Google Drive (clara_drive:)
#          dan mentransfer seluruh arsip .tar.gz hasil backup secara otomatis.
# ==============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$PROJECT_DIR")"

echo -e "${CYAN}======================================================================${NC}"
echo -e "${GREEN}☁️  GOOGLE DRIVE BACKUP TRANSFER TOOL (rclone)${NC}"
echo -e "${CYAN}======================================================================${NC}"

# 1. Cari folder backup terbaru jika tidak dispesifikasikan via argumen
if [ -n "$1" ]; then
    TARGET_BACKUP="$1"
else
    # Cari direktori backup_brain_ctc_* terbaru di parent folder
    TARGET_BACKUP=$(find "$PARENT_DIR" -maxdepth 1 -type d -name "backup_brain_ctc_*" | sort -r | head -n 1)
fi

if [ -z "$TARGET_BACKUP" ] || [ ! -d "$TARGET_BACKUP" ]; then
    echo -e "${RED}❌ Error: Folder backup tidak ditemukan!${NC}"
    echo -e "   Silakan jalankan ${YELLOW}./archive_dgx_experiments.sh${NC} terlebih dahulu"
    echo -e "   atau masukkan path manual: ${CYAN}./transfer_backup_to_drive.sh /path/to/backup_folder${NC}"
    exit 1
fi

echo -e "Sumber Data Lokal : ${YELLOW}${TARGET_BACKUP}${NC}"
echo -e "Target Google Drive: ${BLUE}clara_drive:backup-brain-ctc-seg/${NC}"
echo ""

# 2. Periksa ketersediaan tool rclone
if ! command -v rclone &> /dev/null; then
    echo -e "${RED}❌ rclone belum terinstall di environment ini.${NC}"
    echo -e "   Menginstall rclone via conda..."
    conda install -c conda-forge rclone -y
fi

# 3. Buat direktori 'backup-brain-ctc-seg' di Google Drive
echo -e "${BLUE}[1/3] Membuat folder 'backup-brain-ctc-seg' di Google Drive...${NC}"
rclone mkdir clara_drive:backup-brain-ctc-seg

# Jika mount lokal new_drive aktif di parent folder, buat juga foldernya
if [ -d "${PARENT_DIR}/new_drive" ]; then
    mkdir -p "${PARENT_DIR}/new_drive/backup-brain-ctc-seg" 2>/dev/null || true
fi
echo -e "  ${GREEN}✓ Folder 'backup-brain-ctc-seg' berhasil dibuat dan diverifikasi di Google Drive.${NC}"
echo ""

# 4. Transfer file backup ke Google Drive
echo -e "${BLUE}[2/3] Mengunggah seluruh file backup ke clara_drive:backup-brain-ctc-seg/...${NC}"
echo -e "      (Kecepatan koneksi datacenter DGX, progress real-time)..."
echo ""

rclone copy "${TARGET_BACKUP}/" clara_drive:backup-brain-ctc-seg/ \
    -P \
    --transfers 4 \
    --checkers 8 \
    --drive-chunk-size 64M \
    --stats 2s

echo ""
echo -e "${GREEN}✓ Proses upload selesai!${NC}"
echo ""

# 5. Verifikasi isi di Google Drive
echo -e "${BLUE}[3/3] Memeriksa daftar file yang tersimpan di Google Drive...${NC}"
echo -e "${CYAN}----------------------------------------------------------------------${NC}"
rclone ls clara_drive:backup-brain-ctc-seg/
echo -e "${CYAN}----------------------------------------------------------------------${NC}"

echo ""
echo -e "${CYAN}======================================================================${NC}"
echo -e "${GREEN}🎉 SUKSES: Seluruh data model & hasil eksperimen telah aman di Drive!${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Lokasi Drive : ${YELLOW}clara_drive:backup-brain-ctc-seg/${NC}"
echo -e "Lokasi Mount : ${BLUE}${PARENT_DIR}/new_drive/backup-brain-ctc-seg/${NC}"
echo ""
