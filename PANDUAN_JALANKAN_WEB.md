# 🧠 Panduan Menjalankan Web Dashboard Brain CT Agreement & Ngrok

Dokumen ini berisi catatan lengkap perintah (*command cheat-sheet*) untuk menjalankan aplikasi web Streamlit dan menghubungkannya ke Ngrok (atau Cloudflare Tunnel).

---

## ⚡ Perintah Utama (Harian / Daily Commands)

Berikut 2 perintah persis yang biasa kita gunakan:

### 1️⃣ Terminal 1: Nyalakan Web
```bash
cd /raid/D13K48009/Clara/brain-ctc-seg
./run_agreement_web.sh 8501
```

### 2️⃣ Terminal 2: Nyalakan Ngrok
```bash
ngrok http 8501 --config ~/.config/ngrok/ngrok2.yml --host-header="localhost:8501"
```

---

## 🚀 1. Persiapan Awal di Server (DGX)

Buka terminal dan masuk ke direktori proyek, lalu pastikan kode sudah paling baru:

```bash
cd /raid/D13K48009/Clara/brain-ctc-seg
git pull origin main
```

---

## 🖥️ 2. Menjalankan Web Dashboard (Streamlit)

Buka **Terminal 1** (atau gunakan `tmux` / `screen`):

### Cara Cepat (Rekomendasi):
```bash
./run_agreement_web.sh 8501
```

### Jika Ingin Memilih Nomor GPU Tertentu (Misal GPU 5):
```bash
CUDA_VISIBLE_DEVICES=5 ./run_agreement_web.sh 8501
```

### Perintah Manual (Alternatif Langsung Streamlit):
```bash
streamlit run app.py --server.port 8501
```

> **Catatan:** File `.streamlit/config.toml` sudah otomatis mengatur `enableCORS = false` dan `enableXsrfProtection = false` agar koneksi WebSocket Ngrok lancar dan tidak ditolak.

---

## 🌐 3. Menghubungkan ke Ngrok (Akses Publik / Klien)

Buka **Terminal 2** (tab baru):

### Cara Cepat (Pakai Script Helper):
```bash
./start_ngrok.sh 8501
```

### Cara Manual (Ngrok CLI Bawaan):
```bash
ngrok http 8501 --host-header="localhost:8501"
```

Setelah dijalankan, Ngrok akan memunculkan URL publik HTTPS seperti:
```text
Forwarding   https://xxxx-xxxx-xxxx.ngrok-free.dev -> http://localhost:8501
```
Bagikan URL tersebut ke klien atau buka di browser Anda.

---

## 👥 4. Menjalankan 2 Link / Akun Ngrok Sekaligus (Multi-Tunnel)

Jika Anda ingin membuka 2 aplikasi sekaligus dengan link berbeda:

### Terminal 2 (Akun Pertama — Port 8501):
```bash
ngrok http 8501 --host-header="localhost:8501"
```

### Terminal 3 (Akun Kedua — Port 8502):
```bash
ngrok http 8502 --config ~/.config/ngrok/ngrok2.yml --host-header="localhost:8502"
```

*(Token akun kedua cukup didaftarkan sekali saja dengan perintah: `ngrok config add-authtoken TOKEN_BARU --config ~/.config/ngrok/ngrok2.yml`)*

---

## ⚡ 5. Alternatif Cloudflare Tunnel (100% Gratis & Bebas Batasan Akun)

Jika ingin membuka link publik instan tanpa batasan akun:

```bash
# Download binary (cukup sekali):
curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared

# Jalankan tunnel:
./cloudflared tunnel --url http://localhost:8501
```

---

## 🛠️ 6. Troubleshooting (Perintah Darurat)

### Cek Port Mana yang Sedang Berjalan:
```bash
lsof -i :8501 -i :8502
# atau
netstat -tuln | grep -E "8501|8502"
```

### Hentikan Streamlit yang Macet di Background:
```bash
pkill -f streamlit
```

### Hentikan Ngrok yang Masih Nyangkut:
```bash
pkill -f ngrok
```

### Jika Muncul Error "NumPy 1.x cannot be run in NumPy 2.x" (ESCNN Crash):
Library steerable `escnn` (`lie_learn`) wajib menggunakan NumPy versi 1.x. Jalankan perintah ini:
```bash
pip install "numpy<2"
```
*(Script `./run_agreement_web.sh` terbaru sekarang sudah otomatis mengecek dan memperbaiki ini).*
