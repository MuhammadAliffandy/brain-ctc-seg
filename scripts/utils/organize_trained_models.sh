#!/bin/bash
# Script untuk mengumpulkan semua model .pth yang telah di-training ke dalam satu folder rapi

echo "=========================================================="
echo "📂 MENGELOMPOKKAN SEMUA MODEL (.pth) KE FOLDER FINAL"
echo "=========================================================="

TARGET_DIR="Final_Trained_Models"

# Buat folder utama dan subfolder
mkdir -p "${TARGET_DIR}/CT"
mkdir -p "${TARGET_DIR}/CTC"
mkdir -p "${TARGET_DIR}/Stroke"
mkdir -p "${TARGET_DIR}/Hemorrhage"

echo "✅ Membuat direktori ${TARGET_DIR}..."

# 1. CT Models
echo "Mengkopi model CT..."
cp training/saved_models_25D/*_ct_best.pth "${TARGET_DIR}/CT/" 2>/dev/null

# 2. CTC Models
echo "Mengkopi model CTC..."
cp training/saved_models_25D/*_ctc_best.pth "${TARGET_DIR}/CTC/" 2>/dev/null

# 3. Stroke Models (Kaggle Stroke)
echo "Mengkopi model Stroke..."
# Ambil model kaggle_best.pth (tapi kecualikan kaggle_hemorrhage_best.pth jika ada)
for file in public_dataset/saved_models/*_kaggle_best.pth; do
    if [ -f "$file" ]; then
        cp "$file" "${TARGET_DIR}/Stroke/"
    fi
done

# 4. Hemorrhage Models (Kaggle Hemorrhage)
echo "Mengkopi model Hemorrhage..."
cp public_dataset/saved_models/*_kaggle_hemorrhage_best.pth "${TARGET_DIR}/Hemorrhage/" 2>/dev/null

echo "=========================================================="
echo "🎉 SELESAI! Semua model telah dikelompokkan ke dalam folder:"
echo "$(pwd)/${TARGET_DIR}/"
echo "=========================================================="
tree "${TARGET_DIR}" || ls -R "${TARGET_DIR}"
