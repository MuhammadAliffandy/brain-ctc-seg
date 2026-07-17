import os
import glob
import re
import pandas as pd

def parse_best_from_terminal(log_path):
    # Parse format: Ep  42 [CT] Loss 0.1234 | Dice 0.7890 | IoU 0.6543 | Prec 0.8123 | Rec 0.7654 | LR 1.00e-04
    pattern = re.compile(
        r"Ep\s+(\d+).*?\|\s*Dice\s+([\d.]+)\s*\|\s*IoU\s+([\d.]+)"
    )
    best_dice = 0.0
    best_iou = 0.0
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                dice = float(m.group(2))
                iou = float(m.group(3))
                if dice > best_dice:
                    best_dice = dice
                if iou > best_iou:
                    best_iou = iou
    return best_dice, best_iou

def get_latest_terminal_log(prefix):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    folders = glob.glob(os.path.join(logs_dir, f"{prefix}*"))
    if not folders:
        return None
    latest_folder = sorted(folders, key=os.path.getmtime)[-1]
    
    # Check for terminal.txt or terminal_ct.txt
    txts = glob.glob(os.path.join(latest_folder, "terminal*.txt"))
    if not txts:
        return None
    return txts[0]

def update_csv(csv_name, log_prefix):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Temukan CSV di root proyek
    csv_path = os.path.join(project_root, csv_name)
    if not os.path.exists(csv_path):
        print(f"⚠️  Tidak menemukan {csv_path} di DGX. Lewati.")
        return

    log_path = get_latest_terminal_log(log_prefix)
    if not log_path:
        print(f"⚠️  Tidak menemukan log terbaru untuk {log_prefix}. Lewati.")
        return

    best_dice, best_iou = parse_best_from_terminal(log_path)
    if best_dice == 0.0:
        print(f"⚠️  Log {log_path} belum memiliki metrik Dice. Lewati.")
        return

    print(f"✅ Ditemukan skor baru dari {log_path} -> Best Dice: {best_dice:.4f}, Best IoU: {best_iou:.4f}")

    # Update CSV
    df = pd.read_csv(csv_path)
    # Cari baris yang mengandung "Mod-Seg-SE(2)"
    mask = df['Model'].str.contains('Mod-Seg-SE\(2\)', na=False)
    if mask.any():
        df.loc[mask, 'Best Dice'] = best_dice
        df.loc[mask, 'Best IoU'] = best_iou
        df.to_csv(csv_path, index=False)
        print(f"✅ Berhasil mengupdate skor Mod-Seg-SE(2) di {csv_name}!")
    else:
        print(f"⚠️  Tidak menemukan baris Mod-Seg-SE(2) di {csv_name}.")

def main():
    print("🔄 Mengupdate CSV dengan hasil training TERBARU untuk grafik ROC...")
    update_csv("ct_summary.csv", "exp_retrain_ct_")
    update_csv("ctc_summary.csv", "exp_retrain_ctc_")
    
    # Jalankan generator ROC
    print("\n🎨 Memulai proses pembuatan grafik 2x2 ROC...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roc_script = os.path.join(project_root, "paper_evalute", "plot_combined_2x2_roc.py")
    
    import subprocess
    subprocess.run(["python", roc_script])

if __name__ == "__main__":
    main()
