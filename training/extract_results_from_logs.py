"""
extract_results_from_logs.py
=============================
Cari log training terbaru dari folder logs/, parse metrik terbaik (Best Dice & IoU),
lalu simpan ke CSV summary.

Format log yang dipahami:
    Ep  42 | Loss 0.1234 | Dice 0.7890 | IoU 0.6543 | Prec 0.8123 | Rec 0.7654 | LR 1.00e-04
    Ep  42 [CT] Loss 0.1234 | Dice 0.7890 | IoU 0.6543 | Prec 0.8123 | Rec 0.7654 | LR 1.00e-04

Usage:
    python extract_results_from_logs.py
    python extract_results_from_logs.py --log_dir logs/exp_20260702_123456
    python extract_results_from_logs.py --out results_summary.csv
    python extract_results_from_logs.py --dataset ct
    python extract_results_from_logs.py --last_n 10   # ambil rata-rata N epoch terakhir
"""

import os
import re
import glob
import argparse
import datetime
import pandas as pd

# ================================================================
# CONFIG — Model display names
# ================================================================
MODEL_DISPLAY = {
    "se2":        "Mod-Seg-SE(2) [OURS]",
    "harmonic":   "HarmonicNet (C4)",
    "nnunet":     "nnU-Net",
    "attention":  "Attention U-Net",
    "transunet":  "TransUNet",
    "unet":       "Standard U-Net",
}

DATASET_DISPLAY = {
    "ct":  "CT (Non-Contrast)",
    "ctc": "CTC (CTC + CTW)",
    "all": "All Combined",
}


# ================================================================
# PARSE SINGLE LOG FILE
# ================================================================
def parse_log(log_path: str):
    """
    Parse satu file log training, return list per epoch.
    """
    # Support 2 format:
    # Format A (train_comparison_models): "Ep  42 | Loss ..."
    # Format B (train_se2_by_dataset):    "Ep  42 [CT] Loss ..."
    pattern = re.compile(
        r"Ep\s+(\d+)"                          # epoch
        r"(?:\s+\[.*?\])?"                     # optional [DATASET] tag
        r"\s*[|\s]+Loss\s+([\d.]+)"            # loss
        r"\s*\|\s*Dice\s+([\d.]+)"             # dice
        r"\s*\|\s*IoU\s+([\d.]+)"              # iou
        r"\s*\|\s*Prec\s+([\d.]+)"             # precision
        r"\s*\|\s*Rec\s+([\d.]+)"              # recall
        r"(?:\s*\|\s*LR\s+([\d.e+-]+))?"       # lr (optional)
    )

    records = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                records.append({
                    'epoch': int(m.group(1)),
                    'loss':  float(m.group(2)),
                    'dice':  float(m.group(3)),
                    'iou':   float(m.group(4)),
                    'prec':  float(m.group(5)),
                    'rec':   float(m.group(6)),
                    'lr':    float(m.group(7)) if m.group(7) else None,
                })
    return records


# ================================================================
# EXTRACT BEST + FINAL METRICS FROM RECORDS
# ================================================================
def summarize(records: list, last_n: int = 5):
    """
    Dari list epoch records, return dict metrik ringkasan.
    - best_*  : nilai terbaik selama training
    - final_* : rata-rata N epoch terakhir (lebih stabil dari epoch akhir saja)
    """
    if not records:
        return None

    dices  = [r['dice'] for r in records]
    ious   = [r['iou']  for r in records]
    precs  = [r['prec'] for r in records]
    recs   = [r['rec']  for r in records]
    losses = [r['loss'] for r in records]

    best_dice_idx = dices.index(max(dices))
    best_iou_idx  = ious.index(max(ious))

    # Final = rata-rata N epoch terakhir
    tail = records[-last_n:]
    n    = len(tail)

    return {
        'Best Dice':         round(max(dices), 4),
        'Best Dice Epoch':   records[best_dice_idx]['epoch'],
        'Best IoU':          round(max(ious),  4),
        'Best IoU Epoch':    records[best_iou_idx]['epoch'],
        'Best Prec':         round(max(precs), 4),
        'Best Rec':          round(max(recs),  4),

        f'Final Dice (avg last {last_n})': round(sum(r['dice'] for r in tail) / n, 4),
        f'Final IoU  (avg last {last_n})': round(sum(r['iou']  for r in tail) / n, 4),
        f'Final Prec (avg last {last_n})': round(sum(r['prec'] for r in tail) / n, 4),
        f'Final Rec  (avg last {last_n})': round(sum(r['rec']  for r in tail) / n, 4),

        'Min Loss':            round(min(losses), 4),
        'Last Epoch':          records[-1]['epoch'],
        'Total Epochs Parsed': len(records),
    }


# ================================================================
# FIND LATEST LOGS PER (model, dataset)
# ================================================================
def find_latest_logs(log_dir: str, dataset_filter: str = None):
    """
    Cari semua file .txt di log_dir (rekursif).
    Nama file: training_{model}_{dataset}_{timestamp}.txt
    Kembalikan dict: {(model, dataset): path_terbaru}
    """
    pattern = re.compile(
        r"training_([a-z0-9]+)_([a-z]+)_(\d{8}_\d{6})\.txt$",
        re.IGNORECASE
    )

    found = {}  # (model, dataset) -> (timestamp_str, path)

    # Cari rekursif di log_dir
    all_txts = glob.glob(os.path.join(log_dir, "**", "*.txt"), recursive=True)
    all_txts += glob.glob(os.path.join(log_dir, "*.txt"))
    all_txts  = list(set(all_txts))

    for path in all_txts:
        fname = os.path.basename(path)
        m = pattern.match(fname)
        if not m:
            continue
        model_key   = m.group(1).lower()
        dataset_key = m.group(2).lower()
        timestamp   = m.group(3)

        if dataset_filter and dataset_key != dataset_filter.lower():
            continue

        key = (model_key, dataset_key)
        if key not in found or timestamp > found[key][0]:
            found[key] = (timestamp, path)

    return {k: v[1] for k, v in found.items()}


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract best metrics dari training logs → CSV summary"
    )
    parser.add_argument(
        '--log_dir', type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'),
        help="Root folder tempat log .txt disimpan (default: ./logs)"
    )
    parser.add_argument(
        '--out', type=str, default=None,
        help="Output CSV path (default: results_summary_{timestamp}.csv di log_dir)"
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help="Filter dataset: ct | ctc | all (default: semua)"
    )
    parser.add_argument(
        '--last_n', type=int, default=5,
        help="Rata-rata N epoch terakhir untuk metrik 'Final' (default: 5)"
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help="Tampilkan detail semua metrik per log"
    )
    args = parser.parse_args()

    log_dir = os.path.expanduser(args.log_dir)

    if not os.path.isdir(log_dir):
        # Fallback: cari di direktori parent (training/)
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        if os.path.isdir(alt):
            log_dir = alt
            print(f"⚠️  log_dir tidak ditemukan, fallback ke: {log_dir}")
        else:
            print(f"❌ Folder log tidak ditemukan: {log_dir}")
            print("   Jalankan training dulu, atau tentukan --log_dir secara manual.")
            return

    print(f"\n{'='*65}")
    print(f"  📂 Scanning logs di  : {log_dir}")
    print(f"  Filter dataset       : {args.dataset or 'semua'}")
    print(f"  Final avg last N     : {args.last_n} epochs")
    print(f"{'='*65}\n")

    latest = find_latest_logs(log_dir, dataset_filter=args.dataset)

    if not latest:
        print("❌ Tidak ada file log yang cocok ditemukan.")
        print("   Pastikan format nama file: training_{model}_{dataset}_{YYYYMMDD_HHMMSS}.txt")
        return

    rows = []
    for (model_key, dataset_key), log_path in sorted(latest.items()):
        model_name   = MODEL_DISPLAY.get(model_key, model_key.upper())
        dataset_name = DATASET_DISPLAY.get(dataset_key, dataset_key.upper())

        print(f"  🔍 [{dataset_name}] {model_name}")
        print(f"      Log : {os.path.basename(log_path)}")

        records = parse_log(log_path)

        if not records:
            print(f"      ⚠️  Tidak ada epoch metric ditemukan — skip.\n")
            continue

        summary = summarize(records, last_n=args.last_n)

        print(f"      ✅  {summary['Total Epochs Parsed']} epochs | "
              f"Best Dice={summary['Best Dice']:.4f} (ep {summary['Best Dice Epoch']}) | "
              f"Best IoU={summary['Best IoU']:.4f} (ep {summary['Best IoU Epoch']})")

        if args.verbose:
            for k, v in summary.items():
                print(f"         {k}: {v}")
        print()

        row = {
            'Model':    model_name,
            'Dataset':  dataset_name,
            'Log File': os.path.basename(log_path),
            'Log Path': log_path,
        }
        row.update(summary)
        rows.append(row)

    if not rows:
        print("❌ Tidak ada data yang berhasil di-parse.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(['Dataset', 'Best Dice'], ascending=[True, False])

    # Output path
    if args.out:
        out_path = os.path.expanduser(args.out)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(log_dir, f"results_summary_{ts}.csv")

    df.to_csv(out_path, index=False)

    # ── Pretty print summary table ──
    print(f"\n{'='*65}")
    print("  📊 HASIL RINGKASAN (sorted by Best Dice)")
    print(f"{'='*65}")

    cols = ['Model', 'Dataset', 'Best Dice', 'Best IoU',
            'Best Prec', 'Best Rec', 'Best Dice Epoch', 'Last Epoch']
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))

    print(f"\n{'='*65}")
    print(f"  💾 CSV tersimpan → {out_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
