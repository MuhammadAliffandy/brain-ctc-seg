"""
evaluate_trained_models.py
==========================
Quick evaluation of already-trained comparison models.
Loads weights, runs inference on val split, prints Dice/IoU/Prec/Rec.

Usage (DGX):
    python ~/Clara/brain-ctc-seg/paper_evalute/evaluate_trained_models.py
"""

import os, re, sys
import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np, pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from escnn import gspaces
import escnn.nn as enn


# ================================================================
# ARCHITECTURES (same as train_comparison_models.py)
# ================================================================

# ── HarmonicNet (C4) ──
class _EqConv(nn.Module):
    def __init__(self, a, b, m=None):
        super().__init__()
        if m is None: m = b
        self.seq = enn.SequentialModule(
            enn.R2Conv(a,m,3,padding=1,bias=False), enn.InnerBatchNorm(m), enn.ReLU(m,inplace=True),
            enn.R2Conv(m,b,3,padding=1,bias=False), enn.InnerBatchNorm(b), enn.ReLU(b,inplace=True),
        )
    def forward(self, x): return self.seq(x)

class _EqDown(nn.Module):
    def __init__(self, a, b):
        super().__init__(); self.pool=enn.PointwiseMaxPool(a,2); self.conv=_EqConv(a,b)
    def forward(self, x): return self.conv(self.pool(x))

class _EqUp(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.up=enn.R2Upsampling(a,scale_factor=2,mode='bilinear',align_corners=True)
        self.conv=_EqConv(a+b,b)
    def forward(self, x1, x2): return self.conv(enn.tensor_directsum([x2, self.up(x1)]))

class HarmonicNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, N=4, base_channels=32):
        super().__init__()
        self.act = gspaces.rot2dOnR2(N=N)
        c = base_channels
        def ft(n): return enn.FieldType(self.act, n * [self.act.regular_repr])
        self.fin = enn.FieldType(self.act, n_channels * [self.act.trivial_repr])
        f1,f2,f3,f4,f5 = ft(c),ft(c*2),ft(c*4),ft(c*8),ft(c*16)
        self.inc = _EqConv(self.fin,f1)
        self.d1,self.d2,self.d3,self.d4 = _EqDown(f1,f2),_EqDown(f2,f3),_EqDown(f3,f4),_EqDown(f4,f5)
        self.u1,self.u2,self.u3,self.u4 = _EqUp(f5,f4),_EqUp(f4,f3),_EqUp(f3,f2),_EqUp(f2,f1)
        self.outc = enn.R2Conv(f1, enn.FieldType(self.act, n_classes*[self.act.trivial_repr]), 1)
    def forward(self, x):
        g=enn.GeometricTensor(x,self.fin)
        x1=self.inc(g); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3); x5=self.d4(x4)
        x=self.u1(x5,x4); x=self.u2(x,x3); x=self.u3(x,x2); x=self.u4(x,x1)
        return self.outc(x).tensor


# ── Standard U-Net ──
class _DC(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True),
            nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True),
        )
    def forward(self, x): return self.seq(x)

class StandardUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc=_DC(n_channels,64)
        self.d1=nn.Sequential(nn.MaxPool2d(2),_DC(64,128))
        self.d2=nn.Sequential(nn.MaxPool2d(2),_DC(128,256))
        self.d3=nn.Sequential(nn.MaxPool2d(2),_DC(256,512))
        self.u1=nn.ConvTranspose2d(512,256,2,stride=2); self.c1=_DC(512,256)
        self.u2=nn.ConvTranspose2d(256,128,2,stride=2); self.c2=_DC(256,128)
        self.u3=nn.ConvTranspose2d(128,64, 2,stride=2); self.c3=_DC(128,64)
        self.out=nn.Conv2d(64,n_classes,1)
    def _pc(self, x, s):
        dy=s.size(2)-x.size(2); dx=s.size(3)-x.size(3)
        return torch.cat([s, F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])], 1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        x=self.c1(self._pc(self.u1(x4),x3))
        x=self.c2(self._pc(self.u2(x),x2))
        x=self.c3(self._pc(self.u3(x),x1))
        return self.out(x)


# ================================================================
# DATASET
# ================================================================
class CTBrain25DDataset(Dataset):
    def __init__(self, dataframe, root_dir):
        self.root_dir=root_dir; self.patient_slices={}; self.all_samples=[]
        pc='Patient_Folder' if 'Patient_Folder' in dataframe.columns else 'Patient'
        for p in dataframe[pc].unique():
            pd_=os.path.join(root_dir,p)
            if not os.path.exists(pd_): continue
            imgs=sorted([f for f in os.listdir(pd_) if f.endswith('_img.npy')],
                        key=lambda x: int(re.findall(r'\d+',x)[-1]) if re.findall(r'\d+',x) else 0)
            pairs=[(os.path.join(pd_,n), os.path.join(pd_,n).replace('_img.npy','_mask.npy'))
                   for n in imgs if os.path.exists(os.path.join(pd_,n).replace('_img.npy','_mask.npy'))]
            if pairs:
                self.patient_slices[p]=pairs
                for i in range(len(pairs)): self.all_samples.append((p,i))

    def __len__(self): return len(self.all_samples)

    def __getitem__(self, idx):
        p,si=self.all_samples[idx]; sl=self.patient_slices[p]
        pp=max(0,si-1); nx=min(len(sl)-1,si+1)
        i0=np.load(sl[pp][0]).astype(np.float32)
        i1=np.load(sl[si][0]).astype(np.float32)
        i2=np.load(sl[nx][0]).astype(np.float32)
        m =np.load(sl[si][1]).astype(np.uint8)
        if m.max()>1: m=(m>0).astype(np.uint8)
        img=torch.from_numpy(np.stack([i0,i1,i2],-1)).permute(2,0,1)
        mask=torch.from_numpy(m).long()
        return img, mask


# ================================================================
# EVALUATION
# ================================================================
def evaluate(model, loader, device, name):
    model.eval(); tp=fp=fn=0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"  {name}", ncols=80):
            imgs=imgs.to(device,non_blocking=True)
            masks=masks.to(device,non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits=model(imgs)
            preds=torch.argmax(F.softmax(logits,1),1)
            pf=preds.view(-1); mf=masks.view(-1)
            tp+=((pf==1)&(mf==1)).sum().item()
            fp+=((pf==1)&(mf==0)).sum().item()
            fn+=((pf==0)&(mf==1)).sum().item()
    eps=1e-7
    total=tp+fp+fn
    acc=(tp)/(total+eps)   # approximate (no TN tracked, but gives relative measure)
    prec=tp/(tp+fp+eps)
    rec =tp/(tp+fn+eps)
    f1  =(2*tp)/(2*tp+fp+fn+eps)
    iou =tp/(tp+fp+fn+eps)
    return {'Accuracy':round(acc,4),'Precision':round(prec,4),'Recall':round(rec,4),'F1 (Dice)':round(f1,4),'IoU':round(iou,4)}


# ================================================================
# MAIN
# ================================================================
def main():
    CSV_REPORT  = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH   = os.path.expanduser("~/Clara/local_ct_workspace")
    SAVE_DIR    = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D")

    # ─── Model registry: display name → (class, weight filename) ───
    MODELS = [
        ("HarmonicNet (C4)",   HarmonicNet,   "harmonic_net_epoch_100.pth"),
        ("Standard U-Net",     StandardUNet,  "standard_unet_epoch_100.pth"),
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print(f"  📊 EVALUATION — Trained Comparison Models")
    print(f"  Device: {device} | Split: 15% val | Metric: Dice / IoU")
    print(f"{'='*65}\n")

    if not os.path.exists(CSV_REPORT):
        print(f"❌ CSV not found: {CSV_REPORT}"); sys.exit(1)

    df       = pd.read_csv(CSV_REPORT)
    train_df = df.sample(frac=0.85, random_state=42)
    val_df   = df.drop(train_df.index)
    print(f"  Val patients : {len(val_df)}")

    val_set    = CTBrain25DDataset(val_df, DATA_PATH)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"  Val slices   : {len(val_set)}\n")

    all_results = []
    for display_name, ModelClass, weight_file in MODELS:
        weight_path = os.path.join(SAVE_DIR, weight_file)
        print(f"{'─'*65}")
        print(f"  Model : {display_name}")
        print(f"  Weights: {weight_path}")

        if not os.path.exists(weight_path):
            print(f"  ⚠️  Weight file not found — skipping\n")
            continue

        model = ModelClass(n_channels=3, n_classes=2).to(device)
        model.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True), strict=False
        )
        print(f"  ✅ Weights loaded\n")

        metrics = evaluate(model, val_loader, device, display_name)
        all_results.append({"Model": display_name, **metrics})

        print(f"\n  Results → Dice: {metrics['F1 (Dice)']} | IoU: {metrics['IoU']} | "
              f"Precision: {metrics['Precision']} | Recall: {metrics['Recall']}\n")
        del model; torch.cuda.empty_cache()

    # ─── Summary table ───
    if all_results:
        df_res = pd.DataFrame(all_results).sort_values('F1 (Dice)', ascending=False)
        print(f"\n{'='*65}")
        print("  SUMMARY TABLE")
        print(f"{'='*65}")
        print(df_res.to_string(index=False))
        print(f"{'='*65}\n")

        out_csv = os.path.expanduser("~/Clara/comparison_eval_results.csv")
        df_res.to_csv(out_csv, index=False)
        print(f"  💾 Saved to: {out_csv}\n")


if __name__ == "__main__":
    main()
