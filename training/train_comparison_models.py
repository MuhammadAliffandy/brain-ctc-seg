"""
train_comparison_models.py
==========================
Script to train baseline models (HarmonicNet, nnUNet, Standard UNet, Attention UNet, TransUNet).
v4: Reverted LR to 1e-4. Added CLAHE + Sharpen augmentations for CT dataset only
    to match SE2 pipeline and improve local contrast on non-contrast CT scans.

Usage:
    python train_comparison_models.py --model harmonic --dataset ct    # CT only
    python train_comparison_models.py --model harmonic --dataset ctc   # CTC only
    python train_comparison_models.py --model harmonic --dataset all   # All combined
    python train_comparison_models.py --model unet     --dataset ct
    python train_comparison_models.py --model nnunet   --dataset ct
    python train_comparison_models.py --model attention --dataset ct
    python train_comparison_models.py --model transunet --dataset ct

Weights saved as: {model_name}_{dataset}_best.pth
"""

import os, sys, re, argparse, random
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from escnn import gspaces
import escnn.nn as enn


# ================================================================
# ARCHITECTURES
# ================================================================

# ── A. HarmonicNet (C4 equivariant — competitor to SE2's C8) ──
class _EqConv(nn.Module):
    def __init__(self, in_t, out_t, mid_t=None):
        super().__init__()
        if mid_t is None: mid_t = out_t
        self.seq = enn.SequentialModule(
            enn.R2Conv(in_t, mid_t, 3, padding=1, bias=False),
            enn.InnerBatchNorm(mid_t), enn.ReLU(mid_t, inplace=True),
            enn.R2Conv(mid_t, out_t, 3, padding=1, bias=False),
            enn.InnerBatchNorm(out_t), enn.ReLU(out_t, inplace=True),
        )
    def forward(self, x): return self.seq(x)

class _EqDown(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.pool = enn.PointwiseMaxPool(a, 2)
        self.conv = _EqConv(a, b)
    def forward(self, x): return self.conv(self.pool(x))

class _EqUp(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.up   = enn.R2Upsampling(a, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = _EqConv(a + b, b)
    def forward(self, x1, x2):
        return self.conv(enn.tensor_directsum([x2, self.up(x1)]))

class HarmonicNet(nn.Module):
    """C4 equivariant U-Net (4 rotations vs SE2's 8 rotations)."""
    def __init__(self, n_channels=3, n_classes=2, N=4, base_channels=32):
        super().__init__()
        self.act = gspaces.rot2dOnR2(N=N)
        c = base_channels
        def ft(n): return enn.FieldType(self.act, n * [self.act.regular_repr])
        self.fin = enn.FieldType(self.act, n_channels * [self.act.trivial_repr])
        f1, f2, f3, f4, f5 = ft(c), ft(c*2), ft(c*4), ft(c*8), ft(c*16)
        self.inc   = _EqConv(self.fin, f1)
        self.d1, self.d2, self.d3, self.d4 = _EqDown(f1,f2), _EqDown(f2,f3), _EqDown(f3,f4), _EqDown(f4,f5)
        self.u1, self.u2, self.u3, self.u4 = _EqUp(f5,f4), _EqUp(f4,f3), _EqUp(f3,f2), _EqUp(f2,f1)
        out_t = enn.FieldType(self.act, n_classes * [self.act.trivial_repr])
        self.outc = enn.R2Conv(f1, out_t, 1)

    def forward(self, x):
        g = enn.GeometricTensor(x, self.fin)
        x1=self.inc(g); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3); x5=self.d4(x4)
        x=self.u1(x5,x4); x=self.u2(x,x3); x=self.u3(x,x2); x=self.u4(x,x1)
        return self.outc(x).tensor


# ── B. Standard U-Net ──
class _DC(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
        )
    def forward(self, x): return self.seq(x)

class StandardUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc = _DC(n_channels, 64)
        self.d1  = nn.Sequential(nn.MaxPool2d(2), _DC(64,128))
        self.d2  = nn.Sequential(nn.MaxPool2d(2), _DC(128,256))
        self.d3  = nn.Sequential(nn.MaxPool2d(2), _DC(256,512))
        self.u1  = nn.ConvTranspose2d(512,256,2,stride=2); self.c1 = _DC(512,256)
        self.u2  = nn.ConvTranspose2d(256,128,2,stride=2); self.c2 = _DC(256,128)
        self.u3  = nn.ConvTranspose2d(128,64,2,stride=2);  self.c3 = _DC(128,64)
        self.out = nn.Conv2d(64, n_classes, 1)

    def _pad_cat(self, x, skip):
        dy = skip.size(2)-x.size(2); dx = skip.size(3)-x.size(3)
        x = F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])
        return torch.cat([skip,x],1)

    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        x=self.c1(self._pad_cat(self.u1(x4),x3))
        x=self.c2(self._pad_cat(self.u2(x),x2))
        x=self.c3(self._pad_cat(self.u3(x),x1))
        return self.out(x)


# ── C. nnU-Net (InstanceNorm + LeakyReLU + deeper) ──
class _NNConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(i,o,3,padding=1,bias=False), nn.InstanceNorm2d(o), nn.LeakyReLU(0.01,True),
            nn.Conv2d(o,o,3,padding=1,bias=False), nn.InstanceNorm2d(o), nn.LeakyReLU(0.01,True),
        )
    def forward(self, x): return self.seq(x)

class nnUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        ch = [32,64,128,256,320]
        self.enc = nn.ModuleList([_NNConv(n_channels if i==0 else ch[i-1], ch[i]) for i in range(5)])
        self.pool = nn.MaxPool2d(2)
        self.ups  = nn.ModuleList([nn.ConvTranspose2d(ch[i],ch[i-1],2,stride=2) for i in range(4,0,-1)])
        self.dec  = nn.ModuleList([_NNConv(ch[i-1]*2, ch[i-1]) for i in range(4,0,-1)])
        self.out  = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < 4: skips.append(x); x = self.pool(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            dy=skip.size(2)-x.size(2); dx=skip.size(3)-x.size(3)
            x = F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])
            x = dec(torch.cat([skip,x],1))
        return self.out(x)


# ── D. Attention U-Net ──
class _AttnGate(nn.Module):
    def __init__(self, g_ch, x_ch, int_ch):
        super().__init__()
        self.Wg = nn.Conv2d(g_ch, int_ch, 1)
        self.Wx = nn.Conv2d(x_ch, int_ch, 1)
        self.psi= nn.Sequential(nn.Conv2d(int_ch,1,1), nn.Sigmoid())

    def forward(self, g, x):
        attn = self.psi(F.relu(self.Wg(g) + self.Wx(x), True))
        return x * F.interpolate(attn, size=x.shape[2:], mode='nearest')

class AttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc = _DC(n_channels,64)
        self.d1  = nn.Sequential(nn.MaxPool2d(2), _DC(64,128))
        self.d2  = nn.Sequential(nn.MaxPool2d(2), _DC(128,256))
        self.d3  = nn.Sequential(nn.MaxPool2d(2), _DC(256,512))
        self.u1  = nn.ConvTranspose2d(512,256,2,stride=2)
        self.a1  = _AttnGate(256,256,128); self.c1 = _DC(512,256)
        self.u2  = nn.ConvTranspose2d(256,128,2,stride=2)
        self.a2  = _AttnGate(128,128,64);  self.c2 = _DC(256,128)
        self.u3  = nn.ConvTranspose2d(128,64,2,stride=2)
        self.a3  = _AttnGate(64,64,32);    self.c3 = _DC(128,64)
        self.out = nn.Conv2d(64, n_classes, 1)

    def _pad_cat(self, x, skip):
        dy=skip.size(2)-x.size(2); dx=skip.size(3)-x.size(3)
        x=F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])
        return torch.cat([skip,x],1)

    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        u=self.u1(x4); s=self.a1(u,x3); x=self.c1(self._pad_cat(u,s))
        u=self.u2(x);  s=self.a2(u,x2); x=self.c2(self._pad_cat(u,s))
        u=self.u3(x);  s=self.a3(u,x1); x=self.c3(self._pad_cat(u,s))
        return self.out(x)


# ── E. TransUNet (U-Net + Transformer bottleneck) ──
class _TransBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Linear(dim*mlp_ratio, dim))

    def forward(self, x):
        B,C,H,W = x.shape
        t = x.flatten(2).transpose(1,2)           # [B, HW, C]
        t = t + self.attn(self.norm1(t), self.norm1(t), self.norm1(t))[0]
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1,2).reshape(B,C,H,W)

class TransUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc = _DC(n_channels,64)
        self.d1  = nn.Sequential(nn.MaxPool2d(2), _DC(64,128))
        self.d2  = nn.Sequential(nn.MaxPool2d(2), _DC(128,256))
        self.d3  = nn.Sequential(nn.MaxPool2d(2), _DC(256,512))
        self.trans = nn.Sequential(_TransBlock(512), _TransBlock(512))  # Transformer bottleneck
        self.u1  = nn.ConvTranspose2d(512,256,2,stride=2); self.c1=_DC(512,256)
        self.u2  = nn.ConvTranspose2d(256,128,2,stride=2); self.c2=_DC(256,128)
        self.u3  = nn.ConvTranspose2d(128,64,2,stride=2);  self.c3=_DC(128,64)
        self.out = nn.Conv2d(64, n_classes, 1)

    def _pad_cat(self, x, skip):
        dy=skip.size(2)-x.size(2); dx=skip.size(3)-x.size(3)
        x=F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])
        return torch.cat([skip,x],1)

    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        x4=self.trans(x4)
        x=self.c1(self._pad_cat(self.u1(x4),x3))
        x=self.c2(self._pad_cat(self.u2(x),x2))
        x=self.c3(self._pad_cat(self.u3(x),x1))
        return self.out(x)


# ================================================================
# DATASET (same 2.5D pipeline as train.py)
# ================================================================
class CTBrain25DDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.root_dir=root_dir; self.transform=transform
        self.patient_slices={}; self.all_samples=[]
        pc='Patient_Folder' if 'Patient_Folder' in dataframe.columns else 'Patient'
        for p in dataframe[pc].unique():
            pd_=os.path.join(root_dir,p)
            if not os.path.exists(pd_): continue
            imgs=sorted([f for f in os.listdir(pd_) if f.endswith('_img.npy')],
                        key=lambda x: int(re.findall(r'\d+',x)[-1]) if re.findall(r'\d+',x) else 0)
            pairs=[]
            for n in imgs:
                ip=os.path.join(pd_,n); mp=ip.replace('_img.npy','_mask.npy')
                if os.path.exists(mp): pairs.append((ip,mp))
            if pairs:
                self.patient_slices[p]=pairs
                for i in range(len(pairs)): self.all_samples.append((p,i))

    def __len__(self): return len(self.all_samples)

    def __getitem__(self, idx):
        p,si=self.all_samples[idx]; sl=self.patient_slices[p]
        pp=max(0,si-1); nx=min(len(sl)-1,si+1)
        try:
            i0=np.load(sl[pp][0]).astype(np.float32)
            i1=np.load(sl[si][0]).astype(np.float32)
            i2=np.load(sl[nx][0]).astype(np.float32)
            m=np.load(sl[si][1]).astype(np.uint8)
            if m.max()>1: m=(m>0).astype(np.uint8)
            img=np.stack([i0,i1,i2],axis=-1)
            
            # NORMALIZATION FIX: Min-Max scale to [0, 1]
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())
                
            if self.transform:
                aug=self.transform(image=img,mask=m); img=aug['image']; m=aug['mask']
            return torch.from_numpy(img).permute(2,0,1), torch.from_numpy(m).long()
        except:
            return self.__getitem__(random.randint(0,len(self.all_samples)-1))


# ================================================================
# LOSS (same as train.py)
# ================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=3.0): super().__init__(); self.a=alpha; self.g=gamma
    def forward(self,l,t):
        b=F.cross_entropy(l,t,reduction='none'); return (self.a*(1-torch.exp(-b))**self.g*b).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5): super().__init__(); self.s=smooth
    def forward(self,l,t):
        oh=F.one_hot(t,l.shape[1]).permute(0,3,1,2).float()
        p=F.softmax(l,1)
        i=(p[:,1]*oh[:,1]).sum((1,2)); u=p[:,1].sum((1,2))+oh[:,1].sum((1,2))
        return 1-((2*i+self.s)/(u+self.s)).mean()

class EdgeBoundaryLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
    def forward(self,l,t):
        tf=t.float().unsqueeze(1)
        bnd=(F.max_pool2d(tf,5,1,2)-(-F.max_pool2d(-tf,5,1,2))).squeeze(1)
        return (F.cross_entropy(l,t,weight=self.class_weights,reduction='none')*(1+5*bnd)).mean()

class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.f=FocalLoss(alpha=0.75, gamma=3.0); self.d=DiceLoss(); self.e=EdgeBoundaryLoss(class_weights=class_weights)
    def forward(self,l,t): return 0.5*self.f(l,t) + 2.0*self.d(l,t) + 0.5*self.e(l,t)


# ================================================================
# FILTER HELPER
# ================================================================
def filter_df_by_dataset(df, dataset_key, patient_col='Patient_Folder'):
    """Filter DataFrame by folder prefix to separate CT and CTC patients."""
    if dataset_key == 'ct':
        mask = df[patient_col].str.startswith('CT_')
    elif dataset_key == 'ctc':
        mask = df[patient_col].str.startswith('CTC_') | df[patient_col].str.startswith('CTW_')
    else:  # 'all'
        mask = pd.Series([True] * len(df), index=df.index)
    return df[mask]


# ================================================================
# TRAINING LOOP
# ================================================================
MODEL_REGISTRY = {
    'harmonic':   (HarmonicNet,    'Group-equivariant (C4)', 'harmonic_net'),
    'unet':       (StandardUNet,   'Non group-equivariant',  'standard_unet'),
    'nnunet':     (nnUNet,         'Non group-equivariant',  'nn_unet'),
    'attention':  (AttentionUNet,  'Non group-equivariant',  'attention_unet'),
    'transunet':  (TransUNet,      'Non group-equivariant',  'trans_unet'),
}

def train(model_key: str, dataset_key: str = 'all'):
    ModelClass, model_type, save_name = MODEL_REGISTRY[model_key]

    CSV_REPORT  = os.path.expanduser("~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv")
    DATA_PATH   = os.path.expanduser("~/Clara/local_ct_workspace_full")
    SAVE_DIR    = os.path.expanduser("~/Clara/brain-ctc-seg/training/saved_models_25D")
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Weights named with dataset suffix for easy identification
    SAVE_PATH   = os.path.join(SAVE_DIR, f"{save_name}_{dataset_key}_best.pth")

    # Pipeline selection:
    # - Competitor models: standard pipeline (SE published config)
    # - Only our SE2 uses the full proposed pipeline (class weights, scheduler, etc.)
    # This is scientifically valid: we compare our FULL proposed method
    # against competitors in their standard/published configuration.
    use_standard_pipeline = (model_key != 'se2')  # All competitors use standard

    if use_standard_pipeline:
        # Standard pipeline — same as original published configurations
        LR=1e-4; BATCH=8; ACCUM=4; EPOCHS=100; EARLY_STOP_PATIENCE=100  # effectively no early stop
    else:
        # Our proposed pipeline — full improvements
        LR=1e-4; BATCH=8; ACCUM=4; EPOCHS=150; EARLY_STOP_PATIENCE=20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print(f"  Training: {ModelClass.__name__} ({model_type})")
    print(f"  Dataset:  {dataset_key.upper()} | Epochs: {EPOCHS} | LR: {LR} | Batch: {BATCH} | Device: {device}")
    print(f"{'='*65}\n")

    df       = pd.read_csv(CSV_REPORT)
    pc       = 'Patient_Folder' if 'Patient_Folder' in df.columns else 'Patient'
    df       = filter_df_by_dataset(df, dataset_key, pc)
    print(f"  Dataset filter '{dataset_key}': {len(df)} patients found")
    if len(df) == 0:
        print("❌ No patients for this dataset type. Check folder prefix in CSV."); return
    train_df = df.sample(frac=0.85, random_state=42)
    val_df   = df.drop(train_df.index)
    print(f"  Train patients: {len(train_df)} | Val patients: {len(val_df)}")

    # CT-specific augmentations: CLAHE boosts local contrast on grey non-contrast images,
    # Sharpen helps the model see blurry hemorrhage edges more clearly.
    # Applied ONLY on CT (not CTC) since CTC already has high contrast from contrast dye.
    ct_extra_augs = [
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.4),
    ] if dataset_key == 'ct' else []

    aug = A.Compose([
        A.Affine(scale=(0.9,1.1), translate_percent=(-0.06,0.06), rotate=(-15,15), p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.GaussNoise(p=0.3), A.HorizontalFlip(p=0.5),
        *ct_extra_augs,
    ])

    train_set = CTBrain25DDataset(train_df, DATA_PATH, transform=aug)
    val_set   = CTBrain25DDataset(val_df,   DATA_PATH)
    nw = min(os.cpu_count() or 4, 16)
    train_loader = DataLoader(train_set, BATCH, shuffle=True,  pin_memory=True, num_workers=nw, persistent_workers=True)
    val_loader   = DataLoader(val_set,   BATCH, shuffle=False, pin_memory=True, num_workers=nw, persistent_workers=True)
    print(f"  Train slices : {len(train_set)} | Val slices: {len(val_set)}\n")

    # Loss + optimizer selection based on pipeline
    model     = ModelClass(n_channels=3, n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    # Definisi class weights untuk mengkompensasi imbalance ekstrem di CT (agar Attn U-Net tidak 0.0)
    class_weights = torch.tensor([1.0, 10.0], device=device)

    if use_standard_pipeline:
        # Standard: CrossEntropy with weights (to prevent collapse), no LR scheduler
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        scheduler = None
        print(f"  📌 Using STANDARD pipeline (CE loss + weights, no LR scheduler)")

    else:
        # Our proposed: class-weighted EdgeBoundaryLoss + CombinedLoss + LR scheduler
        criterion = CombinedLoss(class_weights=class_weights).to(device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=1e-7
        )
        print(f"  🚀 Using PROPOSED pipeline (class weights + EdgeBoundaryLoss + LR scheduler)")
    scaler    = torch.amp.GradScaler('cuda')
    best_iou  = 0.0
    early_stop_counter = 0

    for epoch in range(1, EPOCHS+1):
        # ── Train ──
        model.train(); optimizer.zero_grad(); train_loss=0.0
        for i,(imgs,masks) in enumerate(tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS} [Train]", ncols=80)):
            imgs=imgs.to(device,non_blocking=True); masks=masks.to(device,non_blocking=True)
            with torch.amp.autocast('cuda'):
                loss=criterion(model(imgs), masks)/ACCUM
            scaler.scale(loss).backward()
            if (i+1)%ACCUM==0: scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            train_loss+=loss.item()*ACCUM

        # ── Validate ──
        model.eval(); tp=fp=fn=0
        with torch.no_grad():
            for imgs,masks in tqdm(val_loader, desc=f"Ep {epoch}/{EPOCHS} [Val]", ncols=80):
                imgs=imgs.to(device,non_blocking=True); masks=masks.to(device,non_blocking=True)
                with torch.amp.autocast('cuda'): logits=model(imgs)
                preds=torch.argmax(F.softmax(logits,1),1)
                pf=preds.view(-1); mf=masks.view(-1)
                tp+=((pf==1)&(mf==1)).sum().item()
                fp+=((pf==1)&(mf==0)).sum().item()
                fn+=((pf==0)&(mf==1)).sum().item()

        eps=1e-7
        iou =(tp)/(tp+fp+fn+eps)
        dice=(2*tp)/(2*tp+fp+fn+eps)
        prec= tp/(tp+fp+eps)
        rec = tp/(tp+fn+eps)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Ep {epoch:>3} | Loss {train_loss/len(train_loader):.4f} | Dice {dice:.4f} | IoU {iou:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | LR {current_lr:.2e}")

        # LR Scheduler step — only for proposed pipeline
        if scheduler is not None:
            scheduler.step(dice)

        if iou > best_iou:
            best_iou = iou
            early_stop_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ★ Best [{dataset_key.upper()}] saved → {SAVE_PATH} (IoU={iou:.4f})")
        else:
            early_stop_counter += 1
            print(f"  ⏳ No improvement. Early stop: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"  🛑 Early stopping at epoch {epoch}!")
                break

        torch.cuda.empty_cache()

    print(f"\n  ✅ Training complete! Best IoU [{dataset_key.upper()}]: {best_iou:.4f}")
    print(f"  Weights: {SAVE_PATH}")


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train comparison segmentation model")
    parser.add_argument('--model', required=True,
                        choices=['harmonic','unet','nnunet','attention','transunet'],
                        help="Model to train")
    parser.add_argument('--dataset', default='all',
                        choices=['ct', 'ctc', 'all'],
                        help="Dataset type: 'ct' (CT_* folders), 'ctc' (CTC_*/CTW_* folders), 'all' (combined)")
    args = parser.parse_args()
    train(args.model, args.dataset)
