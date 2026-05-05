"""
evaluate_trained_models.py
==========================
Evaluates trained models on CT or CTC validation split separately.

Usage (DGX):
    python ~/Clara/brain-ctc-seg/paper_evalute/evaluate_trained_models.py --dataset ct
    python ~/Clara/brain-ctc-seg/paper_evalute/evaluate_trained_models.py --dataset ctc
    python ~/Clara/brain-ctc-seg/paper_evalute/evaluate_trained_models.py --dataset all  # legacy combined
"""

import os, re, sys, argparse
import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np, pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from escnn import gspaces
import escnn.nn as enn


# ================================================================
# ARCHITECTURES
# ================================================================

# ── SE2_CNNET — EXACT copy from training/train.py ──
# Layer names MUST be identical to produce matching checkpoint keys.
class DoubleEquivariantConv(nn.Module):
    def __init__(self, in_type, out_type, mid_type=None):
        super().__init__()
        if not mid_type: mid_type = out_type
        self.double_conv = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(mid_type), enn.ReLU(mid_type, inplace=True),
            enn.R2Conv(mid_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type), enn.ReLU(out_type, inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_type, out_type):
        super().__init__()
        self.pool = enn.PointwiseMaxPool(in_type, kernel_size=2)
        self.conv = DoubleEquivariantConv(in_type, out_type)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_type, out_type):
        super().__init__()
        self.up   = enn.R2Upsampling(in_type, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleEquivariantConv(in_type + out_type, out_type)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x  = enn.tensor_directsum([x2, x1])
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_type, n_classes):
        super().__init__()
        gspace   = in_type.gspace
        out_type = enn.FieldType(gspace, n_classes * [gspace.trivial_repr])
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=1)
    def forward(self, x): return self.conv(x)

class SE2_CNNET(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, N=8, base_channels=24):
        super().__init__()
        self.r2_act = gspaces.rot2dOnR2(N=N)
        c = base_channels
        self.feat_type_in = enn.FieldType(self.r2_act, n_channels * [self.r2_act.trivial_repr])
        self.feat_type_1  = enn.FieldType(self.r2_act, c      * [self.r2_act.regular_repr])
        self.feat_type_2  = enn.FieldType(self.r2_act, (c*2)  * [self.r2_act.regular_repr])
        self.feat_type_3  = enn.FieldType(self.r2_act, (c*4)  * [self.r2_act.regular_repr])
        self.feat_type_4  = enn.FieldType(self.r2_act, (c*8)  * [self.r2_act.regular_repr])
        self.feat_type_5  = enn.FieldType(self.r2_act, (c*16) * [self.r2_act.regular_repr])

        self.inc   = DoubleEquivariantConv(self.feat_type_in, self.feat_type_1)
        self.down1 = Down(self.feat_type_1, self.feat_type_2)
        self.down2 = Down(self.feat_type_2, self.feat_type_3)
        self.down3 = Down(self.feat_type_3, self.feat_type_4)
        self.down4 = Down(self.feat_type_4, self.feat_type_5)
        self.up1   = Up(self.feat_type_5, self.feat_type_4)
        self.up2   = Up(self.feat_type_4, self.feat_type_3)
        self.up3   = Up(self.feat_type_3, self.feat_type_2)
        self.up4   = Up(self.feat_type_2, self.feat_type_1)
        self.outc  = OutConv(self.feat_type_1, n_classes)

    def forward(self, x):
        x_geom = enn.GeometricTensor(x, self.feat_type_in)
        x1 = self.inc(x_geom)
        x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        x  = self.up1(x5, x4); x = self.up2(x, x3)
        x  = self.up3(x, x2);  x = self.up4(x, x1)
        return self.outc(x).tensor


def load_se2_weights(model, path, device):
    """Load SE2 weights — exact same architecture as train.py, no remapping needed."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    sample = list(ckpt.keys())[:3]
    print(f"  🔑 Key sample: {sample}")
    result = model.load_state_dict(ckpt, strict=False)
    if result.missing_keys:
        print(f"  ⚠️  Missing  : {result.missing_keys[:4]}")
    if result.unexpected_keys:
        print(f"  ⚠️  Unexpected: {result.unexpected_keys[:4]}")
    if not result.missing_keys and not result.unexpected_keys:
        print(f"  ✅ All weights loaded perfectly (0 missing, 0 unexpected)")
    return model


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


# ── nnU-Net (InstanceNorm + LeakyReLU) ──
class _NNC(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(i,o,3,padding=1,bias=False),nn.InstanceNorm2d(o),nn.LeakyReLU(0.01,True),
            nn.Conv2d(o,o,3,padding=1,bias=False),nn.InstanceNorm2d(o),nn.LeakyReLU(0.01,True),
        )
    def forward(self, x): return self.seq(x)

class nnUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        ch=[32,64,128,256,320]
        self.enc=nn.ModuleList([_NNC(n_channels if i==0 else ch[i-1],ch[i]) for i in range(5)])
        self.pool=nn.MaxPool2d(2)
        self.ups=nn.ModuleList([nn.ConvTranspose2d(ch[i],ch[i-1],2,stride=2) for i in range(4,0,-1)])
        self.dec=nn.ModuleList([_NNC(ch[i-1]*2,ch[i-1]) for i in range(4,0,-1)])
        self.out=nn.Conv2d(ch[0],n_classes,1)
    def forward(self, x):
        skips=[]
        for i,enc in enumerate(self.enc):
            x=enc(x)
            if i<4: skips.append(x); x=self.pool(x)
        for up,dec,skip in zip(self.ups,self.dec,reversed(skips)):
            x=up(x)
            dy=skip.size(2)-x.size(2); dx=skip.size(3)-x.size(3)
            x=dec(torch.cat([skip,F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])],1))
        return self.out(x)


# ── Attention U-Net ──
class _AttnGate(nn.Module):
    def __init__(self, g, x, mid):
        super().__init__()
        self.Wg=nn.Conv2d(g,mid,1); self.Wx=nn.Conv2d(x,mid,1)
        self.psi=nn.Sequential(nn.Conv2d(mid,1,1),nn.Sigmoid())
    def forward(self, g, x):
        a=self.psi(F.relu(self.Wg(g)+self.Wx(x),True))
        return x*F.interpolate(a,size=x.shape[2:],mode='nearest')

class AttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc=_DC(n_channels,64)
        self.d1=nn.Sequential(nn.MaxPool2d(2),_DC(64,128))
        self.d2=nn.Sequential(nn.MaxPool2d(2),_DC(128,256))
        self.d3=nn.Sequential(nn.MaxPool2d(2),_DC(256,512))
        self.u1=nn.ConvTranspose2d(512,256,2,stride=2); self.a1=_AttnGate(256,256,128); self.c1=_DC(512,256)
        self.u2=nn.ConvTranspose2d(256,128,2,stride=2); self.a2=_AttnGate(128,128,64);  self.c2=_DC(256,128)
        self.u3=nn.ConvTranspose2d(128,64, 2,stride=2); self.a3=_AttnGate(64,64,32);    self.c3=_DC(128,64)
        self.out=nn.Conv2d(64,n_classes,1)
    def _pc(self, x, s):
        dy=s.size(2)-x.size(2); dx=s.size(3)-x.size(3)
        return torch.cat([s,F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])],1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        u=self.u1(x4); x=self.c1(self._pc(u,self.a1(u,x3)))
        u=self.u2(x);  x=self.c2(self._pc(u,self.a2(u,x2)))
        u=self.u3(x);  x=self.c3(self._pc(u,self.a3(u,x1)))
        return self.out(x)


# ── TransUNet (Transformer bottleneck) ──
class _TransBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.n1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.n2=nn.LayerNorm(dim); self.mlp=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def forward(self, x):
        B,C,H,W=x.shape; t=x.flatten(2).transpose(1,2)
        t=t+self.attn(self.n1(t),self.n1(t),self.n1(t))[0]
        t=t+self.mlp(self.n2(t))
        return t.transpose(1,2).reshape(B,C,H,W)

class TransUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc=_DC(n_channels,64)
        self.d1=nn.Sequential(nn.MaxPool2d(2),_DC(64,128))
        self.d2=nn.Sequential(nn.MaxPool2d(2),_DC(128,256))
        self.d3=nn.Sequential(nn.MaxPool2d(2),_DC(256,512))
        self.trans=nn.Sequential(_TransBlock(512),_TransBlock(512))
        self.u1=nn.ConvTranspose2d(512,256,2,stride=2); self.c1=_DC(512,256)
        self.u2=nn.ConvTranspose2d(256,128,2,stride=2); self.c2=_DC(256,128)
        self.u3=nn.ConvTranspose2d(128,64, 2,stride=2); self.c3=_DC(128,64)
        self.out=nn.Conv2d(64,n_classes,1)
    def _pc(self, x, s):
        dy=s.size(2)-x.size(2); dx=s.size(3)-x.size(3)
        return torch.cat([s,F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])],1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        x4=self.trans(x4)
        x=self.c1(self._pc(self.u1(x4),x3))
        x=self.c2(self._pc(self.u2(x),x2))
        x=self.c3(self._pc(self.u3(x),x1))
        return self.out(x)


# ================================================================
# DATASET A — NO resize (for SE2_CNNET trained via train.py)
# ================================================================
class CTBrain25DDatasetNoResize(Dataset):
    """Exact mirror of train.py CTBrain25DDataset — NO resize, raw values."""
    def __init__(self, dataframe, root_dir):
        self.patient_slices={}; self.all_samples=[]
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
        # No resize — same as train.py validation pipeline
        img = np.stack([i0, i1, i2], axis=-1)   # [H, W, 3]
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(m).long()


# ================================================================
# DATASET B — WITH 256x256 resize (for models in train_comparison_models.py)
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

        # Stack 2.5D — NO normalization (training used raw .npy values)
        img    = np.stack([i0, i1, i2], axis=-1)  # [H, W, 3]
        img_t  = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)   # [1,3,H,W]
        mask_t = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        # Resize to 256×256 — same as training pipeline in benchmarking.py
        img_t  = F.interpolate(img_t,  size=(256, 256), mode='bilinear', align_corners=False)
        mask_t = F.interpolate(mask_t, size=(256, 256), mode='nearest')

        return img_t.squeeze(0), mask_t.squeeze(0).squeeze(0).long()


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
# FILTER HELPER
# ================================================================
def filter_df_by_dataset(df, dataset_key, patient_col='Patient_Folder'):
    if dataset_key == 'ct':
        mask = df[patient_col].str.startswith('CT_')
    elif dataset_key == 'ctc':
        mask = df[patient_col].str.startswith('CTC_') | df[patient_col].str.startswith('CTW_')
    else:
        mask = pd.Series([True] * len(df), index=df.index)
    return df[mask]


# ================================================================
# MAIN
# ================================================================
def main(dataset_key: str = 'all'):
    CSV_REPORT  = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH   = os.path.expanduser("~/Clara/local_ct_workspace")
    SAVE_DIR    = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D")

    # ─── Model registry ─────────────────────────────────────────────────────
    # Weights now have dataset suffix: *_ct_best.pth / *_ctc_best.pth
    # For 'all' mode, fall back to old epoch_100 naming (backward compat)
    ds = dataset_key  # short alias
    MODELS = [
        # (display_name, ModelClass, weight_filename, use_se2_loader)
        ("Mod-Seg-SE(2) [OURS]", SE2_CNNET,    f"se2_unet_{ds}_best.pth",        True),
        ("HarmonicNet (C4)",     HarmonicNet,   f"harmonic_net_{ds}_best.pth",    False),
        ("nnU-Net",              nnUNet,        f"nn_unet_{ds}_best.pth",         False),
        ("Attention U-Net",      AttentionUNet, f"attention_unet_{ds}_best.pth",  False),
        ("TransUNet",            TransUNet,     f"trans_unet_{ds}_best.pth",      False),
        ("Standard U-Net",       StandardUNet,  f"standard_unet_{ds}_best.pth",   False),
    ]
    # Fallback weight names for backward compatibility (old 'all' combined run)
    FALLBACK = {
        "Mod-Seg-SE(2) [OURS]": ["se2_unet_epoch_100.pth", "se2_unet_best_25D_Boundary.pth"],
        "HarmonicNet (C4)":     ["harmonic_net_epoch_100.pth"],
        "nnU-Net":              ["nn_unet_epoch_100.pth"],
        "Attention U-Net":      ["attention_unet_epoch_100.pth"],
        "TransUNet":            ["trans_unet_epoch_100.pth"],
        "Standard U-Net":       ["standard_unet_epoch_100.pth"],
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print(f"  📊 EVALUATION — Dataset: {dataset_key.upper()} | Split: 15% val")
    print(f"  Device: {device} | Metric: Dice / IoU")
    print(f"{'='*65}\n")

    if not os.path.exists(CSV_REPORT):
        print(f"❌ CSV not found: {CSV_REPORT}"); sys.exit(1)

    df = pd.read_csv(CSV_REPORT)
    pc = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df = filter_df_by_dataset(df, dataset_key, pc)
    print(f"  Dataset '{dataset_key}': {len(df)} patients total")
    if len(df) == 0:
        print("❌ No patients match this dataset type."); sys.exit(1)

    train_df = df.sample(frac=0.85, random_state=42)
    val_df   = df.drop(train_df.index)
    print(f"  Val patients : {len(val_df)}")

    # Dataset A — no resize — for SE2_CNNET (train.py pipeline)
    val_set_native = CTBrain25DDatasetNoResize(val_df, DATA_PATH)
    val_loader_native = DataLoader(val_set_native, batch_size=8, shuffle=False,
                                   num_workers=4, pin_memory=True, persistent_workers=True)

    # Dataset B — resize to 256x256 — for models in train_comparison_models.py
    val_set_256 = CTBrain25DDataset(val_df, DATA_PATH)
    val_loader_256 = DataLoader(val_set_256, batch_size=8, shuffle=False,
                                num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"  Val slices (native) : {len(val_set_native)}")
    print(f"  Val slices (256px)  : {len(val_set_256)}\n")

    all_results = []
    for entry in MODELS:
        display_name, ModelClass, weight_file, use_se2_loader = entry

        # Choose correct val loader based on training pipeline
        val_loader = val_loader_native if use_se2_loader else val_loader_256

        # Try primary weight path, then fallbacks
        candidates = [weight_file] + FALLBACK.get(display_name, [])
        weight_path = None
        for wf in candidates:
            p = os.path.join(SAVE_DIR, wf)
            if os.path.exists(p): weight_path = p; break

        print(f"{'─'*65}")
        print(f"  Model  : {display_name}")
        print(f"  Weights: {weight_path or 'NOT FOUND'}")

        if weight_path is None:
            print(f"  ⚠️  Weight file not found — skipping\n")
            continue

        model = ModelClass(n_channels=3, n_classes=2).to(device)
        if use_se2_loader:
            model = load_se2_weights(model, weight_path, device)
        else:
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
        print(f"  SUMMARY TABLE — Dataset: {dataset_key.upper()}")
        print(f"{'='*65}")
        print(df_res.to_string(index=False))
        print(f"{'='*65}\n")

        out_csv = os.path.expanduser(f"~/Clara/comparison_eval_{dataset_key}.csv")
        df_res.to_csv(out_csv, index=False)
        print(f"  💾 Saved to: {out_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all models on CT or CTC val split")
    parser.add_argument('--dataset', default='all', choices=['ct', 'ctc', 'all'],
                        help="Dataset type to evaluate on: 'ct', 'ctc', or 'all' (combined)")
    args = parser.parse_args()
    main(args.dataset)
