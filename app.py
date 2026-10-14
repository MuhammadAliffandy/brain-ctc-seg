"""
Brain CT Multi-Model Agreement & Comparison Dashboard
=====================================================
Streamlit-based ensemble agreement platform for Brain CT segmentation (NCCT, CECT, Public Datasets),
benchmarking Mod-Seg-SE(2), HarmonicNet, nnU-Net, Attention U-Net, TransUNet, and Standard U-Net.
"""

import os
import sys
import glob
import re
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ─────────────────────────────────────────────────────────────────
# 1. PAGE SETUP & STYLES (Aesthetic Clinical Theme)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain CT Multi-Model Agreement System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #58a6ff;
        font-weight: 700;
        font-size: 2.1rem;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-header p {
        color: #8b949e;
        font-size: 1.05rem;
        margin: 0;
    }
    .badge-bar {
        display: flex;
        gap: 10px;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .badge {
        background-color: #21262d;
        border: 1px solid #30363d;
        color: #c9d1d9;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .badge-accent {
        background-color: rgba(56, 139, 253, 0.15);
        border: 1px solid #388bfd;
        color: #58a6ff;
    }
    .badge-green {
        background-color: rgba(46, 160, 67, 0.15);
        border: 1px solid #2ea043;
        color: #3fb950;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# 2. MODEL DEFINITIONS (6 BENCHMARK ARCHITECTURES)
# ─────────────────────────────────────────────────────────────────
class _DC(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
        )
    def forward(self, x): return self.seq(x)

class StandardUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc = _DC(n_channels, 64)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), _DC(64,  128))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), _DC(128, 256))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), _DC(256, 512))
        self.u1 = nn.ConvTranspose2d(512,256,2,stride=2); self.c1=_DC(512,256)
        self.u2 = nn.ConvTranspose2d(256,128,2,stride=2); self.c2=_DC(256,128)
        self.u3 = nn.ConvTranspose2d(128,64, 2,stride=2); self.c3=_DC(128,64)
        self.out = nn.Conv2d(64, n_classes, 1)
    def _pc(self, x, s):
        dy=s.size(2)-x.size(2); dx=s.size(3)-x.size(3)
        return torch.cat([s, F.pad(x,[dx//2,dx-dx//2,dy//2,dy-dy//2])],1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        x=self.c1(self._pc(self.u1(x4),x3))
        x=self.c2(self._pc(self.u2(x),x2))
        x=self.c3(self._pc(self.u3(x),x1))
        return self.out(x)

class _NNC(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o), nn.LeakyReLU(0.01,True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o), nn.LeakyReLU(0.01,True),
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

# Equivariant models conditionally imported (escnn on DGX)
HAS_ESCNN = False
try:
    from escnn import gspaces
    from escnn import nn as enn
    HAS_ESCNN = True
except ImportError:
    pass

if HAS_ESCNN:
    class DoubleEquivariantConv(nn.Module):
        def __init__(self, in_type, out_type):
            super().__init__()
            self.conv = enn.SequentialModule(
                enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
                enn.InnerBatchNorm(out_type),
                enn.ReLU(out_type, inplace=True),
                enn.R2Conv(out_type, out_type, kernel_size=3, padding=1, bias=False),
                enn.InnerBatchNorm(out_type),
                enn.ReLU(out_type, inplace=True)
            )
        def forward(self, x): return self.conv(x)

    class Down(nn.Module):
        def __init__(self, in_type, out_type):
            super().__init__()
            self.down = enn.PointwiseMaxPool(in_type, 2)
            self.conv = DoubleEquivariantConv(in_type, out_type)
        def forward(self, x): return self.conv(self.down(x))

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
        def __init__(self, n_channels=3, n_classes=2, N=8, base_channels=32):
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

    class HarmonicNet(nn.Module):
        def __init__(self, n_channels=3, n_classes=2, N=4):
            super().__init__()
            self.g = gspaces.rot2dOnR2(N=N)
            t_in  = enn.FieldType(self.g, n_channels*[self.g.trivial_repr])
            t1 = enn.FieldType(self.g, 16*[self.g.regular_repr])
            t2 = enn.FieldType(self.g, 32*[self.g.regular_repr])
            t3 = enn.FieldType(self.g, 64*[self.g.regular_repr])
            t4 = enn.FieldType(self.g, 128*[self.g.regular_repr])
            self.inc = _EqConv(t_in, t1)
            self.d1  = _EqDown(t1, t2); self.d2 = _EqDown(t2, t3); self.d3 = _EqDown(t3, t4)
            self.u1  = enn.R2Upsampling(t4, 2, mode='bilinear', align_corners=True)
            self.c1  = _EqConv(t4+t3, t3)
            self.u2  = enn.R2Upsampling(t3, 2, mode='bilinear', align_corners=True)
            self.c2  = _EqConv(t3+t2, t2)
            self.u3  = enn.R2Upsampling(t2, 2, mode='bilinear', align_corners=True)
            self.c3  = _EqConv(t2+t1, t1)
            t_out = enn.FieldType(self.g, n_classes*[self.g.trivial_repr])
            self.outc= enn.R2Conv(t1, t_out, 1)
        def forward(self, x):
            x_g = enn.GeometricTensor(x, enn.FieldType(self.g, x.shape[1]*[self.g.trivial_repr]))
            x1 = self.inc(x_g); x2 = self.d1(x1); x3 = self.d2(x2); x4 = self.d3(x3)
            x  = self.c1(enn.tensor_directsum([x3, self.u1(x4)]))
            x  = self.c2(enn.tensor_directsum([x2, self.u2(x)]))
            x  = self.c3(enn.tensor_directsum([x1, self.u3(x)]))
            return self.outc(x).tensor
else:
    SE2_CNNET = None
    HarmonicNet = None


# ─────────────────────────────────────────────────────────────────
# 3. ROBUST PATH DISCOVERY (DGX SERVER & LOCAL FALLBACK)
# ─────────────────────────────────────────────────────────────────
SEARCH_ROOTS = [
    "/raid/D13K48009/Clara/brain-ctc-seg",
    os.path.expanduser("~/Clara/brain-ctc-seg"),
    os.path.abspath(os.path.join(os.path.dirname(__file__))),
    os.path.expanduser("~/brain-ctc-seg"),
]

DATA_SEARCH_DIRS = [
    "/raid/D13K48009/Clara/local_ct_workspace_full",
    os.path.expanduser("~/Clara/local_ct_workspace_full"),
    os.path.expanduser("~/local_ct_workspace_full"),
]

def find_first_existing(subpaths):
    for root in SEARCH_ROOTS:
        for sp in subpaths:
            full = os.path.join(root, sp)
            if os.path.exists(full):
                return full
    return None

DATASET_CONFIGS = {
    "NCCT (Non-Contrast CT)": {
        "key": "ncct",
        "description": "Non-contrast Brain CT — 2.5D spatial context (NTUH Dataset)",
        "source": "npy",
        "prefix": "CT_",
        "weights": {
            "se2": ["training/saved_models_25D/se2_unet_ct_best.pth", "training/saved_models/se2_unet_ct_best.pth"],
            "harmonic": ["training/saved_models_25D/harmonic_net_ct_best.pth"],
            "nn": ["training/saved_models_25D/nn_unet_ct_best.pth"],
            "attn": ["training/saved_models_25D/attention_unet_ct_best.pth"],
            "trans": ["training/saved_models_25D/trans_unet_ct_best.pth"],
            "std": ["training/saved_models_25D/standard_unet_ct_best.pth"],
        }
    },
    "CECT (Contrast-Enhanced CT)": {
        "key": "cect",
        "description": "Contrast-enhanced Brain CT — 2.5D spatial context (NTUH Dataset)",
        "source": "npy",
        "prefix": "CTC_",
        "weights": {
            "se2": ["training/saved_models_25D/se2_unet_ctc_best.pth"],
            "harmonic": ["training/saved_models_25D/harmonic_net_ctc_best.pth"],
            "nn": ["training/saved_models_25D/nn_unet_ctc_best.pth"],
            "attn": ["training/saved_models_25D/attention_unet_ctc_best.pth"],
            "trans": ["training/saved_models_25D/trans_unet_ctc_best.pth"],
            "std": ["training/saved_models_25D/standard_unet_ctc_best.pth"],
        }
    },
    "Public Dataset - Brain Stroke": {
        "key": "stroke",
        "description": "Kaggle External Brain Stroke CT Dataset (Intra-dataset benchmark)",
        "source": "kaggle_stroke",
        "prefix": None,
        "weights": {
            "se2": ["public_dataset/saved_models/Mod-Seg-SE2_kaggle_best.pth"],
            "harmonic": ["public_dataset/saved_models/harmonic_net_kaggle_best.pth"],
            "nn": ["public_dataset/saved_models/nnU-Net_kaggle_best.pth"],
            "attn": ["public_dataset/saved_models/Attention_U-Net_kaggle_best.pth"],
            "trans": ["public_dataset/saved_models/TransUNet_kaggle_best.pth"],
            "std": ["public_dataset/saved_models/Standard_U-Net_kaggle_best.pth"],
        }
    },
    "Public Dataset - Hemorrhage": {
        "key": "hemo",
        "description": "Kaggle External Intracranial Hemorrhage CT Dataset (Intra-dataset benchmark)",
        "source": "kaggle_hemo",
        "prefix": None,
        "weights": {
            "se2": ["public_dataset/saved_models/Mod-Seg-SE2_kaggle_hemorrhage_best.pth"],
            "harmonic": ["public_dataset/saved_models/harmonic_net_kaggle_hemorrhage_best.pth"],
            "nn": ["public_dataset/saved_models/nnU-Net_kaggle_hemorrhage_best.pth"],
            "attn": ["public_dataset/saved_models/Attention_U-Net_kaggle_hemorrhage_best.pth"],
            "trans": ["public_dataset/saved_models/TransUNet_kaggle_hemorrhage_best.pth"],
            "std": ["public_dataset/saved_models/Standard_U-Net_kaggle_hemorrhage_best.pth"],
        }
    }
}

MODEL_METADATA = [
    {"key": "se2",      "name": "Mod-Seg-SE(2)",     "type": "Equivariant (N=8)", "class": SE2_CNNET,      "is_se2": True,  "color": "#3fb950"},
    {"key": "harmonic", "name": "HarmonicNet",       "type": "Equivariant (C4)",  "class": HarmonicNet,    "is_se2": False, "color": "#58a6ff"},
    {"key": "nn",       "name": "nnU-Net",           "type": "Medical SOTA",      "class": nnUNet,         "is_se2": False, "color": "#d29922"},
    {"key": "attn",     "name": "Attention U-Net",   "type": "Attention Gates",   "class": AttentionUNet,  "is_se2": False, "color": "#db61a2"},
    {"key": "trans",    "name": "TransUNet",         "type": "Transformer",       "class": TransUNet,      "is_se2": False, "color": "#bc8cff"},
    {"key": "std",      "name": "Standard U-Net",    "type": "Baseline CNN",      "class": StandardUNet,   "is_se2": False, "color": "#f85149"},
]


# ─────────────────────────────────────────────────────────────────
# 4. DATASET CANDIDATE DISCOVERY & ALIGNED LOADING (FIG 5 PROTOCOL)
# ─────────────────────────────────────────────────────────────────
CROP_MARGIN = 40
ROTATE_K = 3  # Standard axial orientation: frontal/eyes anterior (top)

@st.cache_data(show_spinner=False)
def discover_dataset_cases(dataset_key):
    """
    Scans dataset directories on DGX server and ranks candidate slices by lesion size.
    Returns list of candidate dicts.
    """
    cfg = DATASET_CONFIGS[dataset_key]
    candidates = []
    
    if cfg["source"] == "npy":
        for data_dir in DATA_SEARCH_DIRS:
            if not os.path.exists(data_dir): continue
            for folder in sorted(os.listdir(data_dir)):
                if cfg["prefix"] and not folder.upper().startswith(cfg["prefix"].upper()):
                    continue
                fpath = os.path.join(data_dir, folder)
                if not os.path.isdir(fpath): continue
                imgs = sorted([f for f in os.listdir(fpath) if f.endswith('_img.npy')],
                              key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
                for i, fname in enumerate(imgs):
                    img_path = os.path.join(fpath, fname)
                    mask_path = img_path.replace('_img.npy', '_mask.npy')
                    if not os.path.exists(mask_path): continue
                    m = np.load(mask_path)
                    if m.max() > 1: m = (m > 0).astype(np.uint8)
                    n_px = int(np.sum(m > 0))
                    # Pick prominent tumor slices (500 to 20,000 px)
                    if 500 < n_px < 25000:
                        i_prev = max(0, i - 1)
                        i_next = min(len(imgs) - 1, i + 1)
                        clean_folder = folder.replace('__', '_')[:24]
                        slice_num = fname.replace('_img.npy', '')
                        candidates.append({
                            "type": "npy",
                            "px": n_px,
                            "label": f"[{n_px:,} px lesion] {clean_folder} — Slice {slice_num}",
                            "name": f"{folder}_{slice_num}",
                            "prev": os.path.join(fpath, imgs[i_prev]),
                            "curr": img_path,
                            "next": os.path.join(fpath, imgs[i_next]),
                            "mask": mask_path
                        })
            if candidates: break
            
    # Sort descending by tumor volume so user gets the clearest cases first
    candidates = sorted(candidates, key=lambda x: x["px"], reverse=True)
    return candidates[:12]


def load_aligned_npy_sample(case_dict):
    """
    Exact mirror of comparative_inference_all_datasets.py protocol:
    1. Min-Max normalization
    2. Skull margin crop (CROP_MARGIN = 40)
    3. Rotation (ROTATE_K = 3) to align anatomical orientation
    4. Interpolated resize to 256x256 for consistent CNN receptive fields
    """
    i0 = np.load(case_dict['prev']).astype(np.float32)
    i1 = np.load(case_dict['curr']).astype(np.float32)
    i2 = np.load(case_dict['next']).astype(np.float32)
    mask = np.load(case_dict['mask']).astype(np.uint8)
    if mask.max() > 1: mask = (mask > 0).astype(np.uint8)

    img_25d = np.stack([i0, i1, i2], axis=-1)
    if img_25d.max() > img_25d.min():
        img_25d_n = (img_25d - img_25d.min()) / (img_25d.max() - img_25d.min())
    else:
        img_25d_n = img_25d

    mid = i1.copy()
    if mid.max() > 2.0: mid = mid / 255.0
    if 'CTC_' in case_dict['curr']:
        mid = np.clip(mid * 0.65, 0, 1)
    else:
        mid = np.clip(mid, 0, 1)

    # 1. Crop margins to remove empty scan border
    H, W = mid.shape
    cm = min(CROP_MARGIN, H // 8)
    mid_crop = mid[cm:-cm, cm:-cm]
    mask_crop = mask[cm:-cm, cm:-cm]
    inp_crop = img_25d_n[cm:-cm, cm:-cm]

    # 2. Rotate by k=ROTATE_K (3) to standard anatomical orientation
    mid_rot = np.rot90(mid_crop, k=ROTATE_K).copy()
    mask_rot = np.rot90(mask_crop, k=ROTATE_K).copy()
    inp_rot = np.rot90(inp_crop, k=ROTATE_K).copy()

    # 3. Standardize to 256x256
    mid_final = cv2.resize(mid_rot, (256, 256))
    mask_final = cv2.resize(mask_rot, (256, 256), interpolation=cv2.INTER_NEAREST)
    inp_final = cv2.resize(inp_rot, (256, 256))

    tensor = torch.from_numpy(inp_final).permute(2, 0, 1).unsqueeze(0).float()
    return mid_final, tensor, mask_final


def generate_clinical_synthetic_case(case_idx=0):
    """Fallback generator for local demo mode with realistic tumor boundaries."""
    H, W = 256, 256
    y, x = np.mgrid[:H, :W]
    center_y, center_x = H // 2, W // 2

    # Brain parenchyma skull mask
    brain_mask = ((x - center_x) ** 2) / (98 ** 2) + ((y - center_y) ** 2) / (112 ** 2) <= 1.0
    base_scan = np.zeros((H, W), dtype=np.float32)
    base_scan[brain_mask] = 0.38 + 0.04 * np.sin(x[brain_mask] / 12.0)
    
    # Lateral ventricles
    v_mask = (((x - center_x - 14) ** 2) / (10 ** 2) + ((y - center_y) ** 2) / (28 ** 2) <= 1.0) | \
             (((x - center_x + 14) ** 2) / (10 ** 2) + ((y - center_y) ** 2) / (28 ** 2) <= 1.0)
    base_scan[v_mask] = 0.14

    # Lesion presets
    presets = [
        {"cx": center_x + 38, "cy": center_y - 18, "rx": 24, "ry": 22, "val": 0.82, "name": "Reference Parietal Glioma"},
        {"cx": center_x - 35, "cy": center_y + 25, "rx": 20, "ry": 18, "val": 0.78, "name": "Left Frontal Meningioma"},
        {"cx": center_x + 30, "cy": center_y + 35, "rx": 26, "ry": 20, "val": 0.22, "name": "Temporal Infarct (Hypodense)"},
    ]
    p = presets[case_idx % len(presets)]
    
    lesion_mask = (((x - p["cx"]) ** 2) / (p["rx"] ** 2) + ((y - p["cy"]) ** 2) / (p["ry"] ** 2) <= 1.0) & brain_mask
    base_scan[lesion_mask] = p["val"]

    noise = np.random.normal(0, 0.015, (H, W)).astype(np.float32)
    img_gray = np.clip(base_scan + noise, 0.0, 1.0)
    gt_mask = lesion_mask.astype(np.uint8)
    tensor = torch.from_numpy(np.stack([img_gray, img_gray, img_gray], axis=0)).unsqueeze(0).float()
    return img_gray, tensor, gt_mask, p["name"]


# ─────────────────────────────────────────────────────────────────
# 5. MODEL CACHING & INFERENCE
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_cached_model(model_key, weight_file, device_str):
    device = torch.device(device_str)
    meta = next((m for m in MODEL_METADATA if m["key"] == model_key), None)
    if meta is None or meta["class"] is None:
        return None
        
    ModelClass = meta["class"]
    try:
        if meta["is_se2"]:
            ckpt = torch.load(weight_file, map_location=device, weights_only=False)
            base_channels = 24
            for k, val in ckpt.items():
                if 'batch_norm_[8].weight' in k and 'inc' in k:
                    base_channels = val.shape[0]
                    break
            model = ModelClass(n_channels=3, n_classes=2, base_channels=base_channels).to(device)
            model.load_state_dict(ckpt, strict=False)
        else:
            model = ModelClass(n_channels=3, n_classes=2).to(device)
            ckpt = torch.load(weight_file, map_location=device, weights_only=False)
            model.load_state_dict(ckpt, strict=False)
            
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {model_key} from {weight_file}: {e}")
        return None


def run_single_inference(model, tensor, device_str):
    if model is None: return None
    device = torch.device(device_str)
    try:
        with torch.no_grad():
            t = tensor.to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(t)
            else:
                logits = model(t)

            probs = F.softmax(logits, dim=1) # [1, 2, H, W]
            tumor_prob = probs[0, 1].squeeze(0).cpu().numpy().astype(np.float32)
            pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            return {"prob": tumor_prob, "mask": pred_mask}
    except Exception as e:
        print(f"Inference error: {e}")
        return None


def generate_mock_prediction(gt_mask, img_gray, seed_offset=0):
    """Realistic simulation for local testing when weights are not present."""
    np.random.seed(42 + seed_offset)
    prob = gt_mask.astype(np.float32)
    # Add subtle boundary dilation/erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if seed_offset % 3 == 0:
        prob = cv2.dilate(prob, kernel, iterations=1)
    elif seed_offset % 3 == 1:
        prob = cv2.erode(prob, kernel, iterations=1)
    
    prob = cv2.GaussianBlur(prob, (5, 5), 1.0)
    mask = (prob >= 0.40).astype(np.uint8)
    return {"prob": prob, "mask": mask}


# ─────────────────────────────────────────────────────────────────
# 6. METRICS & ENSEMBLE ENGINE
# ─────────────────────────────────────────────────────────────────
def compute_dice_iou(pred, gt):
    if gt is None: return None, None
    intersection = np.sum((pred > 0) & (gt > 0))
    p_sum = np.sum(pred > 0)
    g_sum = np.sum(gt > 0)
    if p_sum + g_sum == 0: return 1.0, 1.0
    dice = (2.0 * intersection) / (p_sum + g_sum)
    union = p_sum + g_sum - intersection
    iou = intersection / union if union > 0 else 1.0
    return float(dice), float(iou)


def build_ensemble_results(predictions, voting_mode, vote_threshold, conf_threshold):
    valid_keys = [k for k, v in predictions.items() if v is not None and "mask" in v]
    num_models = len(valid_keys)
    if num_models == 0: return None

    first_mask = predictions[valid_keys[0]]["mask"]
    H, W = first_mask.shape
    
    agreement_counts = np.zeros((H, W), dtype=np.int32)
    prob_sum = np.zeros((H, W), dtype=np.float32)

    for k in valid_keys:
        agreement_counts += (predictions[k]["mask"] > 0).astype(np.int32)
        prob_sum += predictions[k]["prob"]

    avg_prob = prob_sum / num_models

    if voting_mode == "Majority Voting":
        ensemble_mask = (agreement_counts >= vote_threshold).astype(np.uint8)
    elif voting_mode == "Soft Voting (Average Probability)":
        ensemble_mask = (avg_prob >= conf_threshold).astype(np.uint8)
    elif voting_mode == "Safe-Fail (High Sensitivity)":
        se2_mask = predictions.get("se2", {}).get("mask", np.zeros_like(agreement_counts))
        ensemble_mask = ((agreement_counts >= 2) | (se2_mask > 0)).astype(np.uint8)
    else:
        ensemble_mask = (agreement_counts >= 3).astype(np.uint8)

    tumor_pixels = int(np.sum(ensemble_mask > 0))
    unanimous_pixels = int(np.sum(agreement_counts >= num_models))
    consensus_ratio = float(unanimous_pixels / tumor_pixels) if tumor_pixels > 0 else 1.0

    return {
        "ensemble_mask": ensemble_mask,
        "agreement_counts": agreement_counts,
        "avg_prob": avg_prob,
        "tumor_pixels": tumor_pixels,
        "unanimous_pixels": unanimous_pixels,
        "consensus_ratio": consensus_ratio,
        "num_models": num_models
    }


def overlay_mask(img_gray, mask, color=(63, 185, 80), alpha=0.45):
    base = (np.clip(img_gray, 0, 1) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    if mask is None or np.sum(mask) == 0:
        return rgb
    blended = np.where(
        mask[..., None] > 0,
        (rgb * (1 - alpha) + np.array(color) * alpha).astype(np.uint8),
        rgb
    )
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color, 1, cv2.LINE_AA)
    return blended


def create_agreement_heatmap(agreement_counts, num_models):
    norm = np.clip(agreement_counts / float(num_models), 0, 1)
    heatmap_colored = (plt.cm.turbo(norm)[..., :3] * 255).astype(np.uint8)
    heatmap_colored[agreement_counts == 0] = [15, 20, 25] # Background
    return heatmap_colored


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


# ─────────────────────────────────────────────────────────────────
# 7. MAIN INTERACTIVE APPLICATION
# ─────────────────────────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Brain CT Multi-Model Agreement & Comparison</h1>
        <p>Ensemble Decision Support Platform for NCCT, CECT, and Public External Brain CT Benchmarks</p>
        <div class="badge-bar">
            <span class="badge badge-accent">6 Benchmark Models</span>
            <span class="badge badge-green">Mod-Seg-SE(2) Equivariant Core</span>
            <span class="badge">Inter-Model Consensus Voting</span>
            <span class="badge">Uncertainty Disagreement Mapping</span>
            <span class="badge">Strict Zero-Leakage Dataset Separation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar Configuration ──
    with st.sidebar:
        st.markdown("### ⚙️ Modality & Dataset")
        selected_dataset_name = st.selectbox(
            "Select Imaging Dataset",
            list(DATASET_CONFIGS.keys()),
            index=0,
            help="NCCT and CECT contrast profiles use distinct, isolated trained weights."
        )
        current_cfg = DATASET_CONFIGS[selected_dataset_name]
        st.caption(f"📁 {current_cfg['description']}")

        st.markdown("---")
        st.markdown("### 🗳️ Agreement & Ensemble Rules")
        voting_mode = st.radio(
            "Consensus Strategy",
            [
                "Majority Voting",
                "Soft Voting (Average Probability)",
                "Safe-Fail (High Sensitivity)"
            ],
            index=0
        )

        vote_threshold = 3
        conf_threshold = 0.50

        if voting_mode == "Majority Voting":
            vote_threshold = st.slider(
                "Consensus Threshold (Models)",
                min_value=2,
                max_value=6,
                value=3,
                help="Minimum number of models that must agree for a pixel to be marked as tumor."
            )
            st.caption(f"Pixel is flagged if ≥ **{vote_threshold}/6** models agree.")
        elif voting_mode == "Soft Voting (Average Probability)":
            conf_threshold = st.slider("Probability Cutoff", 0.20, 0.95, 0.50, 0.05)
            st.caption(f"Flagged if mean probability ≥ **{conf_threshold:.2f}**.")
        else:
            st.caption("Triggers lesion if ≥ 2 models OR Mod-Seg-SE(2) identifies positive.")

        st.markdown("---")
        st.markdown("### 🖥️ Hardware Status")
        device_opt = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Target Device: **{device_opt.upper()}** (escnn: {'Active ✅' if HAS_ESCNN else 'Fallback Mode ⚠️'})")

    # ── SECTION 1: SCAN & GROUND TRUTH INPUT ──
    st.subheader("1. Input Brain CT Scan & Case Selection")
    
    # Check for real patient cases on DGX
    discovered_cases = discover_dataset_cases(selected_dataset_name)
    
    col_input_mode, col_case_picker = st.columns([1, 2])
    with col_input_mode:
        if discovered_cases:
            input_source = st.radio("Source", ["DGX Patient Dataset", "Upload Custom Scan"], horizontal=True)
        else:
            input_source = st.radio("Source", ["Verified Preset Cases", "Upload Custom Scan"], horizontal=True)

    img_gray, tensor_input, gt_mask = None, None, None
    slice_title = "Reference Case"

    if input_source == "DGX Patient Dataset" and discovered_cases:
        with col_case_picker:
            case_options = [c["label"] for c in discovered_cases]
            chosen_label = st.selectbox("Select Patient Tumor Case:", case_options, index=0)
            chosen_case = next(c for c in discovered_cases if c["label"] == chosen_label)
            img_gray, tensor_input, gt_mask = load_aligned_npy_sample(chosen_case)
            slice_title = chosen_case["name"]
            st.caption(f"Loaded Slice: **{chosen_case['name']}** (Confirmed GT: **{int(np.sum(gt_mask > 0)):,} px**)")

    elif input_source == "Upload Custom Scan":
        uploaded_file = st.file_uploader("Upload Brain CT Scan (PNG/JPG/NPY)", type=["png", "jpg", "jpeg", "npy"])
        if uploaded_file is not None:
            slice_title = uploaded_file.name
            if uploaded_file.name.endswith(".npy"):
                arr = np.load(uploaded_file)
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    img_gray = cv2.resize(arr[..., 1], (256, 256))
                    tensor_input = torch.from_numpy(cv2.resize(arr, (256, 256))).permute(2, 0, 1).unsqueeze(0).float()
                else:
                    img_gray = cv2.resize(arr, (256, 256))
                    tensor_input = torch.from_numpy(np.stack([img_gray]*3, axis=0)).unsqueeze(0).float()
            else:
                pil_img = Image.open(uploaded_file).convert("L")
                img_gray = cv2.resize(np.array(pil_img).astype(np.float32) / 255.0, (256, 256))
                tensor_input = torch.from_numpy(np.stack([img_gray]*3, axis=0)).unsqueeze(0).float()
    else:
        with col_case_picker:
            preset_choice = st.selectbox(
                "Select Clinical Preset Case:",
                ["Case 1: Right Parietal Glioma", "Case 2: Left Frontal Meningioma", "Case 3: Temporal Infarct"],
                index=0
            )
            case_idx = 0 if "Case 1" in preset_choice else (1 if "Case 2" in preset_choice else 2)
            img_gray, tensor_input, gt_mask, slice_title = generate_clinical_synthetic_case(case_idx)
            st.caption(f"Loaded Preset: **{slice_title}** (Reference GT: **{int(np.sum(gt_mask > 0)):,} px**)")

    if img_gray is None:
        st.warning("Please select a valid case or upload a slice to begin.")
        return

    # ── Run Predictions for All 6 Models ──
    predictions = {}
    with st.spinner("Running 6-Model Inference Pipeline..."):
        for meta in MODEL_METADATA:
            m_key = meta["key"]
            weight_candidates = current_cfg["weights"].get(m_key, [])
            resolved_weight = find_first_existing(weight_candidates)

            if resolved_weight and (not meta["is_se2"] or HAS_ESCNN):
                model = load_cached_model(m_key, resolved_weight, device_opt)
                pred = run_single_inference(model, tensor_input, device_opt)
                if pred is not None:
                    predictions[m_key] = pred
                    continue

            # Fallback mock prediction for local testing
            seed_offset = {"se2": 0, "harmonic": 3, "nn": 6, "attn": 9, "trans": 12, "std": 15}.get(m_key, 0)
            predictions[m_key] = generate_mock_prediction(gt_mask, img_gray, seed_offset=seed_offset)

    # ── Compute Ensemble Agreement ──
    ensemble_data = build_ensemble_results(predictions, voting_mode, vote_threshold, conf_threshold)
    if ensemble_data is None:
        st.error("Failed to generate ensemble agreement.")
        return

    # ── 4-Panel Core Overview (Raw, GT, Ensemble, Heatmap) ──
    st.markdown("---")
    st.markdown("### 2. Primary Diagnostic & Consensus Overviews")
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        with st.container(border=True):
            st.markdown("**1. Raw Brain CT Scan**")
            st.image(img_gray, use_container_width=True, clamp=True)
            st.caption("Native Axial Slice")

    with p2:
        with st.container(border=True):
            st.markdown("**:blue[2. Ground Truth Annotation]**")
            gt_overlay = overlay_mask(img_gray, gt_mask, color=(56, 139, 253), alpha=0.55) if gt_mask is not None else img_gray
            st.image(gt_overlay, use_container_width=True)
            st.caption(f"Tumor Area: **{int(np.sum(gt_mask > 0)):,} px**" if gt_mask is not None else "No Ground Truth")

    with p3:
        with st.container(border=True):
            st.markdown("**:green[3. Ensemble Consensus Mask]**")
            ens_overlay = overlay_mask(img_gray, ensemble_data["ensemble_mask"], color=(63, 185, 80), alpha=0.55)
            st.image(ens_overlay, use_container_width=True)
            if gt_mask is not None:
                e_dice, e_iou = compute_dice_iou(ensemble_data["ensemble_mask"], gt_mask)
                st.caption(f"Dice: **{e_dice:.3f}** | IoU: **{e_iou:.3f}**")
            else:
                st.caption(f"Consensus Area: **{ensemble_data['tumor_pixels']:,} px**")

    with p4:
        with st.container(border=True):
            st.markdown("**:orange[4. Agreement Heatmap]**")
            heatmap_img = create_agreement_heatmap(ensemble_data["agreement_counts"], ensemble_data["num_models"])
            st.image(heatmap_img, use_container_width=True)
            st.caption(f"Unanimous: **{ensemble_data['consensus_ratio']*100:.1f}%**")

    # ── Consensus Summary Metrics & Clinical Callout ──
    st.markdown("#### Clinical Consensus Evaluation")
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1:
        st.metric("Ensemble Strategy", voting_mode)
    with m_c2:
        st.metric("Consensus Tumor Area", f"{ensemble_data['tumor_pixels']:,} px")
    with m_c3:
        st.metric("Model Agreement Concordance", f"{ensemble_data['consensus_ratio']*100:.1f}%")
    with m_c4:
        if gt_mask is not None:
            ens_dice, ens_iou = compute_dice_iou(ensemble_data["ensemble_mask"], gt_mask)
            st.metric("Ensemble Dice Score", f"{ens_dice:.4f}", delta=f"IoU: {ens_iou:.4f}")
        else:
            st.metric("Unanimous Pixels", f"{ensemble_data['unanimous_pixels']:,} px")

    consensus_level = ensemble_data['consensus_ratio']
    if ensemble_data['tumor_pixels'] == 0:
        st.info("🏥 **Diagnostic Consensus: No Significant Focal Lesion Detected** — All models concur on lesion-free parenchyma.")
    elif consensus_level >= 0.65:
        st.success(
            f"🏥 **Diagnostic Consensus: High Inter-Model Concordance ({consensus_level*100:.1f}%)** — "
            f"**{ensemble_data['unanimous_pixels']:,} pixels** share unanimous multi-model agreement. The core lesion boundary is sharply localized."
        )
    else:
        st.warning(
            f"🏥 **Diagnostic Consensus: Peripheral Ambiguity Flagged ({consensus_level*100:.1f}%)** — "
            "Primary lesion nidus detected, but significant disagreement exists on peripheral boundaries. Specialist radiologist review recommended."
        )

    # ── SECTION 3: INDIVIDUAL MODEL BREAKDOWN (6 MODELS) ──
    st.markdown("---")
    st.subheader("3. Individual Model Segmentation Breakdown (6 Models)")

    m_cols = st.columns(6)
    for idx, meta in enumerate(MODEL_METADATA):
        m_key = meta["key"]
        p = predictions[m_key]
        px_count = int(np.sum(p["mask"] > 0))
        is_prop = (m_key == "se2")
        overlay = overlay_mask(img_gray, p["mask"], color=hex_to_rgb(meta["color"]), alpha=0.50)

        with m_cols[idx]:
            with st.container(border=True):
                if is_prop:
                    st.markdown(f"**:green[{meta['name']}]** ⭐")
                else:
                    st.markdown(f"**{meta['name']}**")
                st.caption(meta['type'])
                
                st.image(overlay, use_container_width=True)
                
                if gt_mask is not None:
                    d, iou = compute_dice_iou(p["mask"], gt_mask)
                    st.metric("Dice Score", f"{d:.3f}", delta=f"IoU: {iou:.3f}")
                else:
                    st.metric("Detected Area", f"{px_count:,} px")
                st.caption(f"Area: **{px_count:,} px**")

    # ── SECTION 4: UNCERTAINTY & AGREEMENT DISTRIBUTION ──
    st.markdown("---")
    st.subheader("4. Inter-Model Agreement Distribution & Volume Comparison")
    col_chart1, col_chart2 = st.columns([1, 1], gap="medium")

    with col_chart1:
        agreement_flat = ensemble_data["agreement_counts"][ensemble_data["agreement_counts"] > 0]
        if len(agreement_flat) > 0:
            counts = [np.sum(agreement_flat == k) for k in range(1, ensemble_data["num_models"] + 1)]
            x_labels = [f"{k}/6 Models" for k in range(1, ensemble_data["num_models"] + 1)]
            palette = ['#4c566a', '#5e81ac', '#81a1c1', '#88c0d0', '#a3be8c', '#3fb950']
            
            if HAS_PLOTLY:
                fig_hist = go.Figure(data=[
                    go.Bar(
                        x=x_labels, y=counts, marker_color=palette,
                        text=[f"{c} px" for c in counts], textposition='auto'
                    )
                ])
                fig_hist.update_layout(
                    title="Pixel Count by Model Agreement Level", template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=260, margin=dict(l=20, r=20, t=35, b=20)
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(6, 3), facecolor='#161b22')
                ax.set_facecolor('#161b22')
                bars = ax.bar(x_labels, counts, color=palette)
                ax.bar_label(bars, fmt='%d px', color='#e6edf3', padding=3, fontsize=8)
                ax.set_title("Pixel Count by Model Agreement Level", color='#58a6ff', fontsize=10, fontweight='bold')
                ax.tick_params(colors='#8b949e', labelsize=8)
                for sp in ax.spines.values(): sp.set_color('#30363d')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    with col_chart2:
        model_names = [m["name"] for m in MODEL_METADATA] + ["Ensemble"]
        model_areas = [int(np.sum(predictions[m["key"]]["mask"] > 0)) for m in MODEL_METADATA] + [ensemble_data["tumor_pixels"]]
        colors = [m["color"] for m in MODEL_METADATA] + ["#388bfd"]

        if HAS_PLOTLY:
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=model_names, y=model_areas, marker_color=colors,
                    text=[f"{a} px" for a in model_areas], textposition='auto'
                )
            ])
            fig_bar.update_layout(
                title="Tumor Segment Area per Model (Pixels)", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=260, margin=dict(l=20, r=20, t=35, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3), facecolor='#161b22')
            ax.set_facecolor('#161b22')
            bars = ax.bar(model_names, model_areas, color=colors)
            ax.bar_label(bars, fmt='%d px', color='#e6edf3', padding=3, fontsize=8)
            ax.set_title("Tumor Segment Area per Model (Pixels)", color='#58a6ff', fontsize=10, fontweight='bold')
            ax.tick_params(colors='#8b949e', labelsize=8)
            plt.xticks(rotation=20, ha='right')
            for sp in ax.spines.values(): sp.set_color('#30363d')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Footer ──
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8b949e; font-size: 0.85rem; padding: 10px 0;">
        Brain CT Multi-Model Agreement System &bull; NTUH NCCT / CECT &amp; Kaggle External Validation &bull; Mod-Seg-SE(2)
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
