#!/bin/bash
# ==============================================================================
# Script: transfer_backup_to_drive.sh
# Purpose: Membuat folder 'backup-brain-ctc-seg' di Google Drive (xxdrive:)
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
echo -e "${GREEN}☁️  GOOGLE DRIVE BACKUP TRANSFER TOOL (rclone -> xxdrive)${NC}"
echo -e "${CYAN}======================================================================${NC}"

# 1. Periksa ketersediaan tool rclone
if ! command -v rclone &> /dev/null; then
    echo -e "${RED}❌ rclone belum terinstall di environment ini.${NC}"
    echo -e "   Menginstall rclone via conda..."
    conda install -c conda-forge rclone -y
fi

# 2. Deteksi remote yang tersedia
AVAILABLE_REMOTES=$(rclone listremotes 2>/dev/null || true)

# Prioritaskan xxdrive: jika tersedia atau jika diberikan via argumen
DRIVE_REMOTE="xxdrive"
TARGET_BACKUP=""

if [ -n "$1" ]; then
    if [[ "$1" == *":"* ]] || [[ "$AVAILABLE_REMOTES" == *"$1:"* ]]; then
        DRIVE_REMOTE="${1%:}"
        TARGET_BACKUP="$2"
    else
        TARGET_BACKUP="$1"
    fi
fi

# Jika argumen tidak menentukan remote, cek apakah xxdrive ada
if ! echo "$AVAILABLE_REMOTES" | grep -q "${DRIVE_REMOTE}:"; then
    if echo "$AVAILABLE_REMOTES" | grep -q "clara_drive:"; then
        DRIVE_REMOTE="clara_drive"
    fi
fi

# 3. Cari folder backup terbaru jika belum diset
if [ -z "$TARGET_BACKUP" ]; then
    TARGET_BACKUP=$(find "$PARENT_DIR" -maxdepth 1 -type d -name "backup_brain_ctc_*" | sort -r | head -n 1)
fi

if [ -z "$TARGET_BACKUP" ] || [ ! -d "$TARGET_BACKUP" ]; then
    echo -e "${RED}❌ Error: Folder backup tidak ditemukan!${NC}"
    echo -e "   Silakan jalankan ${YELLOW}./archive_dgx_experiments.sh${NC} terlebih dahulu"
    echo -e "   atau masukkan path manual: ${CYAN}./transfer_backup_to_drive.sh ${DRIVE_REMOTE} /path/to/backup_folder${NC}"
    exit 1
fi

echo -e "Sumber Data Lokal : ${YELLOW}${TARGET_BACKUP}${NC}"
echo -e "Target Remote     : ${GREEN}${DRIVE_REMOTE}:${NC}"
echo -e "Folder Tujuan     : ${BLUE}${DRIVE_REMOTE}:backup-brain-ctc-seg/${NC}"
echo ""

# 4. Buat direktori 'backup-brain-ctc-seg' di Google Drive target
echo -e "${BLUE}[1/3] Membuat folder 'backup-brain-ctc-seg' di ${DRIVE_REMOTE}...${NC}"
rclone mkdir "${DRIVE_REMOTE}:backup-brain-ctc-seg"
echo -e "  ${GREEN}✓ Folder 'backup-brain-ctc-seg' siap di ${DRIVE_REMOTE}:${NC}"
echo ""

# 5. Transfer file backup ke Google Drive
echo -e "${BLUE}[2/3] Mengunggah seluruh file backup ke ${DRIVE_REMOTE}:backup-brain-ctc-seg/...${NC}"
echo -e "      (Kecepatan koneksi datacenter DGX, progress real-time)..."
echo ""

rclone copy "${TARGET_BACKUP}/" "${DRIVE_REMOTE}:backup-brain-ctc-seg/" \
    -P \
    --transfers 4 \
    --checkers 8 \
    --drive-chunk-size 64M \
    --stats 2s

echo ""
echo -e "${GREEN}✓ Proses upload selesai!${NC}"
echo ""

# 6. Verifikasi isi di Google Drive
echo -e "${BLUE}[3/3] Memeriksa daftar file yang tersimpan di ${DRIVE_REMOTE}:backup-brain-ctc-seg/...${NC}"
echo -e "${CYAN}----------------------------------------------------------------------${NC}"
rclone ls "${DRIVE_REMOTE}:backup-brain-ctc-seg/"
echo -e "${CYAN}----------------------------------------------------------------------${NC}"

echo ""
echo -e "${CYAN}======================================================================${NC}"
echo -e "${GREEN}🎉 SUKSES: Seluruh data model & hasil eksperimen aman di ${DRIVE_REMOTE}!${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Lokasi Drive : ${YELLOW}${DRIVE_REMOTE}:backup-brain-ctc-seg/${NC}"
echo ""
