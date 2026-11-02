# 📦 Panduan Lengkap: Backup Data & Memindahkan Model Comparison dari Server DGX

Dokumen ini berisi panduan langkah demi langkah (*Standard Operating Procedure*) untuk mengarsipkan (**tar.gz**) seluruh aset penting di proyek `brain-ctc-seg`, meliputi bobot model terlatih (**model weights .pth**), log eksperimen, kurva evaluasi, file CSV metrics, naskah jurnal/manuscript, serta cara memindahkannya ke Google Drive atau mengunduhnya ke laptop lokal.

---

## 🎯 1. Daftar Aset Penting yang Harus Dipindahkan

Berdasarkan arsitektur proyek dan benchmark yang telah dijalankan, berikut komponen penting yang **tidak ada di Git** (karena terkena `.gitignore`) dan wajib diselamatkan:

| Komponen | Lokasi di Server | Keterangan |
| :--- | :--- | :--- |
| **Model Weights (.pth)** | `training/saved_models_25D/` & `training/*.pth` | Bobot 6 model terlatih: **Mod-Seg-SE(2)**, **HarmonicNet (C4)**, **nnU-Net**, **Attention U-Net**, **TransUNet**, **Standard U-Net**, model checkpoint per-epoch, K-Fold (Fold 1-5), dan model Ablation. |
| **Training & Evaluation Logs** | `training/*.log` & `training/logs_ablation/` | Log riwayat epoch, loss curve, convergence, time per epoch. |
| **Metrics & Hasil CSV** | `training/*.csv` & `paper_evalute/*.csv` | Tabel master K-Fold (`master_kfold_results.csv`), perbandingan HD95, Dice, IoU, sensitivity, specificity, lesion tracking. |
| **Grafik & Gambar Publikasi** | `paper_evalute/*.png`, `Paper/Journal_Figures/` | Multi-panel ROC curve, comparative heatmaps, ablation curves, GradCAM ensemble visual. |
| **Manuscript & Naskah** | `Manuscript_V3_tracked.docx`, `Paper/` | Naskah final dengan tracked changes, dokumen spesifikasi arsitektur, PPTX presentasi klien. |
| **Web Dashboard & Code** | `app.py`, `evaluation web/`, `.agents/` | Kode source lengkap sistem multi-model decision support dashboard. |

---

## ⚡ 2. Cara Cepat: 1 Perintah Otomatis di DGX Server

Kami telah menyediakan skrip otomatis [`archive_dgx_experiments.sh`](file:///Users/aliffandy/Documents/PukulEnam/brain-ctc-seg/archive_dgx_experiments.sh) yang akan:
1. Memindai semua file penting.
2. Membaginya menjadi **4 paket `.tar.gz` modular** (agar tidak menjadi 1 file raksasa yang rentan corrupt saat transfer).
3. Menghasilkan file **SHA-256 Checksum** (`checksums_sha256.txt`) untuk memvalidasi integritas data.
4. Menyimpan arsip di luar folder git (di direktori parent: `/raid/D13K48009/Clara/backup_brain_ctc_<timestamp>`).

### Langkah Eksekusi di Terminal Server:

```bash
# 1. Masuk ke direktori proyek dan pastikan repo terbaru
cd /raid/D13K48009/Clara/brain-ctc-seg
git pull origin main

# 2. Jalankan skrip pemaketan backup
./archive_dgx_experiments.sh
```

### Hasil Paket yang Dihasilkan:
```text
backup_brain_ctc_20261102_XXXXXX/
├── 01_brain_ctc_model_weights_XXXXXX.tar.gz          <- Seluruh model .pth
├── 02_brain_ctc_experiment_results_and_logs_XXXXXX.tar.gz <- CSV, Log, Grafik
├── 03_brain_ctc_manuscripts_and_reports_XXXXXX.tar.gz <- Word, PPTX, Naskah
├── 04_brain_ctc_codebase_and_web_XXXXXX.tar.gz       <- Source code & Web
├── checksums_sha256.txt                              <- Hash validasi
└── BACKUP_MANIFEST.txt                               <- Catatan detail isi paket
```

---

## 🚚 3. Cara Memindahkan / Mentransfer File Hasil Backup

Ada 2 metode utama yang dapat Anda pilih:

### 🟢 METODE A: Upload Langsung ke Google Drive via `rclone` (SANGAT DIREKOMENDASIKAN)
*Keuntungan: Sangat cepat (100–300 MB/s menggunakan bandwidth datacenter DGX), tidak membebani memori laptop Anda, dan langsung aman di cloud.*

Koneksi `rclone` dengan remote `clara_drive:` sudah terkonfigurasi di server DGX. Cukup jalankan perintah berikut di terminal DGX:

```bash
# Ganti <FOLDER_BACKUP> dengan nama folder hasil eksekusi langkah 2
cd /raid/D13K48009/Clara

rclone copy ./backup_brain_ctc_XXXXXXXX_XXXXXX/ clara_drive:Backup_Brain_CTC_Seg/ -P
```

*(Flag `-P` akan menampilkan progress bar kecepatan dan persentase upload secara real-time).*

---

### 💻 METODE B: Download Langsung ke Laptop / Komputer Lokal via `rsync` / `scp`
Buka terminal di **Laptop/Komputer Lokal Anda** (bukan di server), lalu jalankan:

```bash
# Menggunakan rsync (mendukung resume jika koneksi terputus):
rsync -avzP <USER>@<IP_SERVER_DGX>:/raid/D13K48009/Clara/backup_brain_ctc_*/ ./DGX_Brain_Backup/

# Atau menggunakan SCP:
scp -r <USER>@<IP_SERVER_DGX>:/raid/D13K48009/Clara/backup_brain_ctc_* ./DGX_Brain_Backup/
```

---

## 🛠️ 4. Cara Manual (Jika Ingin Tar File Tertentu Saja)

Jika Anda hanya ingin mengambil bagian tertentu secara manual tanpa skrip:

### 1. Hanya Model Weights (.pth):
```bash
cd /raid/D13K48009/Clara/brain-ctc-seg
tar -czvf /raid/D13K48009/Clara/all_models_weights.tar.gz training/saved_models_25D/
```

### 2. Hanya Log & CSV Hasil Eksperimen:
```bash
cd /raid/D13K48009/Clara/brain-ctc-seg
tar -czvf /raid/D13K48009/Clara/all_results_and_logs.tar.gz \
    paper_evalute/ \
    data_summaries/ \
    training/*.log \
    training/*.csv
```

### 3. Hanya Naskah Manuscript & Presentasi:
```bash
cd /raid/D13K48009/Clara/brain-ctc-seg
tar -czvf /raid/D13K48009/Clara/all_manuscripts.tar.gz \
    Manuscript_V3_tracked.docx \
    Brain_Presentation.pptx \
    Brain_CT_Seg_Client_Presentation.pptx \
    Paper/
```

---

## ✅ 5. Verifikasi Integritas File (Anti-Corrupt)

Setelah file berhasil didownload di komputer baru atau Google Drive:
```bash
cd DGX_Brain_Backup/
sha256sum -c checksums_sha256.txt
```
Jika muncul status `OK` untuk seluruh file `.tar.gz`, artinya data 100% utuh tanpa ada byte yang rusak selama proses pemindahan.
