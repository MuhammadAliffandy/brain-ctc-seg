import os
import glob
import subprocess

def find_latest_model(prefix):
    """Mencari folder terbaru dengan prefix tertentu dan mengambil model terbarunya."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    
    # Cari semua folder yang berawalan prefix
    folders = glob.glob(os.path.join(logs_dir, f"{prefix}*"))
    if not folders:
        return None
        
    # Urutkan berdasarkan waktu modifikasi (terbaru di akhir)
    latest_folder = sorted(folders, key=os.path.getmtime)[-1]
    
    # Cari model .pth di dalamnya
    models = glob.glob(os.path.join(latest_folder, "model_epoch_*.pth"))
    if not models:
        return None
        
    # Ambil model dengan epoch terbesar
    def get_epoch(p):
        try:
            return int(os.path.basename(p).split('_')[-1].split('.')[0])
        except:
            return 0
            
    best_model = sorted(models, key=get_epoch)[-1]
    return best_model

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inf_script = os.path.join(script_dir, "inference_figure.py")
    
    datasets = {
        "CTC": "exp_retrain_ctc_",
        "CT": "exp_retrain_ct_"
    }
    
    for ds_name, prefix in datasets.items():
        print(f"\n=========================================")
        print(f"🚀 Memproses Visualisasi untuk Dataset: {ds_name}")
        
        best_model = find_latest_model(prefix)
        if not best_model:
            print(f"❌ Tidak menemukan model (.pth) terbaru untuk {ds_name}!")
            continue
            
        print(f"✅ Model ditemukan: {best_model}")
        
        out_dir = os.path.join(script_dir, "Inference_Figures", ds_name)
        os.makedirs(out_dir, exist_ok=True)
        
        cmd = [
            "python", inf_script,
            "--model", best_model,
            "--out", out_dir
        ]
        
        print(f"⏳ Menjalankan inference_figure.py...")
        subprocess.run(cmd)
        
    print("\n✅✅ PROSES VISUALISASI SELESAI ✅✅")

if __name__ == "__main__":
    main()
