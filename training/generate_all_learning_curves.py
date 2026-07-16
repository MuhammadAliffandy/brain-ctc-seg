import os
import glob
import subprocess

def main():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    LOG_DIRS = [
        os.path.join(PROJECT_ROOT, "logs"),
        os.path.join(PROJECT_ROOT, "..", "public_dataset")
    ]
    OUT_DIR = os.path.join(PROJECT_ROOT, "learning_curves")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    all_txts = []
    for d in LOG_DIRS:
        if os.path.isdir(d):
            print(f"🔍 Mencari file log di: {d}...")
            all_txts.extend(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True))
    
    if not all_txts:
        print("❌ Tidak ada file log (.txt) yang ditemukan.")
        return
        
    print(f"✅ Ditemukan {len(all_txts)} file log. Mulai memplot learning curve...\n")
    
    success_count = 0
    fail_count = 0
    
    plot_script = os.path.join(PROJECT_ROOT, "..", "paper_evalute", "plot_learning_curve.py")
    
    for log_path in all_txts:
        base_name = os.path.splitext(os.path.basename(log_path))[0]
        out_png = os.path.join(OUT_DIR, f"{base_name}_learning_curve.png")
        
        # Buat judul yang rapi
        title = base_name.replace("log_", "").replace("_", " ").upper()
        
        print(f"⏳ Memproses: {base_name}...")
        
        try:
            # Panggil plot_learning_curve.py
            cmd = [
                "python", plot_script,
                "--log", log_path,
                "--out", out_png,
                "--title", title
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ Tersimpan -> {out_png}")
                success_count += 1
            else:
                print(f"   ❌ Gagal memproses {base_name}. Mungkin bukan log training yang valid.")
                print(f"      Error: {result.stderr.strip()}")
                fail_count += 1
        except Exception as e:
            print(f"   ❌ Error eksekusi skrip: {e}")
            fail_count += 1
            
    print(f"\n=======================================================")
    print(f"🎉 SELESAI! Berhasil: {success_count} | Gagal (Bukan Log Training): {fail_count}")
    print(f"📂 Semua gambar (PNG) disimpan di: {OUT_DIR}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
