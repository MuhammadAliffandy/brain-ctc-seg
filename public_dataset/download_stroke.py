import os
import subprocess

def main():
    print("🚀 Memulai proses download Dataset Stroke...")
    
    # Target direktori download lokal (di dalam project kita)
    target_dir = os.path.expanduser("~/Clara/brain-ctc-seg/public_dataset/brain-stroke-ct-dataset")
    os.makedirs(target_dir, exist_ok=True)
    
    # Mengecek apakah file kaggle.json ada (syarat wajib dari Kaggle)
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json_path):
        print("\n❌ ERROR: Kredensial Kaggle tidak ditemukan!")
        print("Silakan ikuti langkah ini:")
        print("1. Buka kaggle.com -> Account -> Create New API Token")
        print("2. Upload file kaggle.json tersebut ke DGX")
        print("3. Taruh di ~/.kaggle/kaggle.json")
        print("4. Jalankan: chmod 600 ~/.kaggle/kaggle.json")
        return
        
    print(f"✅ Kredensial Kaggle ditemukan di {kaggle_json_path}")
    print(f"📥 Mengunduh dataset ke: {target_dir}")
    
    try:
        # Menjalankan perintah CLI resmi Kaggle
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", "ozguraslank/brain-stroke-ct-dataset", 
            "-p", target_dir, 
            "--unzip"
        ]
        
        # Eksekusi dengan output langsung ke terminal
        subprocess.run(cmd, check=True)
        print("\n✅ Dataset Stroke berhasil didownload dan di-ekstrak!")
        
    except FileNotFoundError:
        print("\n❌ Kaggle CLI belum di-install. Menginstall via pip...")
        subprocess.run(["pip", "install", "kaggle"], check=True)
        print("✅ Instalasi selesai. Silakan jalankan ulang script ini!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Gagal men-download dataset. Error: {e}")
        print("💡 TIPS: Jika muncul error 403 Forbidden, artinya Kakak harus login ke web Kaggle,")
        print("cari dataset 'ozcangundes/brain-stroke-ct-dataset', dan klik 'I Accept' / 'Download' untuk pertama kali.")

if __name__ == "__main__":
    main()
