# Brain CTC Seg - Model Training & Evaluation Pipeline

Repositori ini memuat *source code* untuk riset segmentasi pendarahan otak (Brain Hemorrhage) dan jaringan otak (Brain Targeting) menggunakan CT dan CTC (*Computed Tomography with Contrast*). Pendekatan utama kami menggunakan arsitektur **Mod-Seg-SE(2)** (Group-equivariant CNN) untuk menjaga konsistensi geometris (rotasi/translasi) pada representasi 2.5D slices.

---

## 📂 Struktur Direktori Utama

### 1. `paper_evalute/` (Evaluation & Profiling)
Direktori ini berisi semua skrip untuk *profiling* dataset, menjalankan evaluasi akhir, dan membuat visualisasi metrik untuk kebutuhan penulisan jurnal (Paper).

*   `dataset_profiler.py`
    *   **Fungsi:** Menghitung jumlah aktual *slices* CT vs CTC dari file ZIP di Google Drive dan membandingkannya dengan *workspace* lokal.
    *   **Kapan dipakai:** Sebelum memulai *training* untuk memastikan tidak ada *missing data* atau proses ekstrak yang terputus di *server*.
*   `evaluate_trained_models.py`
    *   **Fungsi:** Melakukan *inference* dan evaluasi metrik (Dice, IoU, Precision, Recall, Accuracy) dari semua model kompetitor terhadap Test Set (split 15%). Menghasilkan file CSV (`comparison_eval_ct.csv` atau `comparison_eval_ctc.csv`).
    *   **Fitur Spesial:** Memiliki argumen `--dataset ct|ctc|all` untuk mengevaluasi model pada domain data tertentu saja.
*   `check_public_dataset.py`
    *   **Fungsi:** Menginspeksi dataset publik eksternal untuk validasi eksternal (External Evaluation) model.
*   `comparative_figure.py` / `comparative_figure_heatmap.py`
    *   **Fungsi:** Membuat *figure* / gambar hasil segmentasi (Overlay, Heatmap) untuk di- *copy-paste* langsung ke dalam file LaTeX paper jurnal.

### 2. `training/` (Model Training)
Direktori ini berisi skrip yang memuat arsitektur model dan *training loop* dari *scratch*.

*   `train.py`
    *   **Fungsi:** Skrip *training* orisinal untuk melatih arsitektur SE(2) menggunakan *Loss* kombinasi (Focal + Dice + Edge Boundary Loss) secara general pada *dataset* gabungan.
*   `train_se2_by_dataset.py`
    *   **Fungsi:** Versi modifikasi dari `train.py` yang bisa melatih model SE(2) secara spesifik **hanya** pada data `CT` atau **hanya** pada data `CTC`.
    *   **Cara pakai:** `python train_se2_by_dataset.py --dataset ct`
*   `train_comparison_models.py`
    *   **Fungsi:** Skrip terpusat untuk melatih model-model pembanding (baseline) dari nol menggunakan pipeline augmentasi dan *loss* yang persis sama. Model yang didukung:
        *   `harmonic` (HarmonicNet C4)
        *   `unet` (Standard U-Net)
        *   `nnunet` (nnU-Net)
        *   `attention` (Attention U-Net)
        *   `transunet` (TransUNet)
    *   **Cara pakai:** `python train_comparison_models.py --model unet --dataset ct`

---

## 🚀 Workflow Eksperimen Jurnal (CT vs CTC Separated)

Klien meminta evaluasi perbandingan model dipisah berdasarkan modalitas pengakuisisian data (CT biasa vs CTC/kontras). Berikut urutan eksekusi eksperimen:

**Step 1: Data Profiling & Extraction**
```bash
python paper_evalute/dataset_profiler.py
```
*(Memastikan 7999 slices diekstrak secara utuh ke `~/Clara/local_ct_workspace_full`)*

**Step 2: Training Model Utama (SE2)**
```bash
python training/train_se2_by_dataset.py --dataset ct
python training/train_se2_by_dataset.py --dataset ctc
```

**Step 3: Training Model Pembanding (Competitors)**
Anda dapat menjalankan *training* secara manual satu per satu untuk setiap model:
```bash
# Model HarmonicNet (C4)
python training/train_comparison_models.py --model harmonic --dataset ct
python training/train_comparison_models.py --model harmonic --dataset ctc

# Model Standard U-Net
python training/train_comparison_models.py --model unet --dataset ct
python training/train_comparison_models.py --model unet --dataset ctc

# Model nnU-Net
python training/train_comparison_models.py --model nnunet --dataset ct
python training/train_comparison_models.py --model nnunet --dataset ctc

# Model Attention U-Net
python training/train_comparison_models.py --model attention --dataset ct
python training/train_comparison_models.py --model attention --dataset ctc

# Model TransUNet
python training/train_comparison_models.py --model transunet --dataset ct
python training/train_comparison_models.py --model transunet --dataset ctc
```

**Alternatif Otomatis (Background Execution)**
Untuk menjalankan semua model di atas secara berurutan dan terhindar dari putusnya koneksi SSH, gunakan *shell scripts* yang sudah disediakan. Skrip-skrip ini sudah otomatis menggunakan `nohup` di dalamnya:
```bash
# Untuk semua eksperimen CT
cd training && ./run_all_ct_models.sh

# Untuk semua eksperimen CTC
cd training && ./run_all_ctc_models.sh
```

**Step 3.5: 5-Fold Cross Validation Training**
Untuk memastikan *robustness* model dengan evaluasi 5-Fold Cross Validation (akan memproses seluruh model dan me-generate `master_kfold_results.csv`):
```bash
cd training && ./run_all_kfold.sh
```

**Step 4: Evaluasi dan Hasilkan Laporan CSV**
```bash
python paper_evalute/evaluate_trained_models.py --dataset ct
python paper_evalute/evaluate_trained_models.py --dataset ctc
```
*Hasil akan tersimpan sebagai CSV di directory home Anda.*
