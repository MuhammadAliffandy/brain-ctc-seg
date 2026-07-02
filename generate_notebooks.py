import json
import os

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=2)

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + '\n' for line in text.split('\n')]}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in text.split('\n')]}

# -----------------------------------------
# 1. SE(2) Notebook
# -----------------------------------------
se2_cells = [
    md("# Mod-Seg-SE(2) Architecture (Ours)\nThis notebook contains the implementation of the proposed Group-Equivariant SE(2) Convolutional Neural Network for hemorrhage segmentation.\nIt uses `escnn` to achieve translation and discrete rotation (N=8) equivariance."),
    code("""import torch
import torch.nn as nn
from escnn import gspaces
import escnn.nn as enn

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
        return self.outc(x).tensor"""),
    md("### Model Initialization\nHere is how to initialize the proposed model:"),
    code("""model = SE2_CNNET(n_channels=3, n_classes=2, N=8, base_channels=32)
print("SE(2) Model instantiated successfully!")""")
]

# -----------------------------------------
# 2. HarmonicNet Notebook
# -----------------------------------------
harmonic_cells = [
    md("# HarmonicNet (C4) Architecture\nThis notebook contains the implementation of the HarmonicNet baseline.\nIt is an equivariant U-Net but relies on a standard C4 discrete group (4 rotations) instead of SE(2) N=8."),
    code("""import torch
import torch.nn as nn
from escnn import gspaces
import escnn.nn as enn

class _EqConv(nn.Module):
    def __init__(self, in_t, out_t, mid_t=None):
        super().__init__()
        if mid_t is None: mid_t = out_t
        self.seq = enn.SequentialModule(
            enn.R2Conv(in_t, mid_t, 3, padding=1, bias=False),
            enn.InnerBatchNorm(mid_t), enn.ReLU(mid_t, inplace=True),
            enn.R2Conv(mid_t, out_t, 3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type=out_t), enn.ReLU(out_t, inplace=True),
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
        return self.outc(x).tensor"""),
    md("### Model Initialization\nHere is how to initialize HarmonicNet:"),
    code("""model = HarmonicNet(n_channels=3, n_classes=2, N=4, base_channels=32)
print("HarmonicNet instantiated successfully!")""")
]

# -----------------------------------------
# 3. nnU-Net Notebook
# -----------------------------------------
nnunet_cells = [
    md("# nnU-Net Baseline Architecture\nThis notebook contains the implementation of the nnU-Net medical standard baseline.\nIt uses Instance Normalization and LeakyReLU inside the convolutional blocks."),
    code("""import torch
import torch.nn as nn
import torch.nn.functional as F

class _NNConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o), nn.LeakyReLU(0.01, True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o), nn.LeakyReLU(0.01, True),
        )
    def forward(self, x): return self.seq(x)

class nnUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        ch = [32, 64, 128, 256, 320]
        self.enc = nn.ModuleList([_NNConv(n_channels if i==0 else ch[i-1], ch[i]) for i in range(5)])
        self.pool = nn.MaxPool2d(2)
        self.ups  = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i-1], 2, stride=2) for i in range(4, 0, -1)])
        self.dec  = nn.ModuleList([_NNConv(ch[i-1]*2, ch[i-1]) for i in range(4, 0, -1)])
        self.out  = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < 4: skips.append(x); x = self.pool(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            dy = skip.size(2) - x.size(2); dx = skip.size(3) - x.size(3)
            x = F.pad(x, [dx//2, dx - dx//2, dy//2, dy - dy//2])
            x = dec(torch.cat([skip, x], 1))
        return self.out(x)"""),
    md("### Model Initialization\nHere is how to initialize nnU-Net:"),
    code("""model = nnUNet(n_channels=3, n_classes=2)
print("nnU-Net instantiated successfully!")""")
]

# Generate the files
out_dir = os.path.expanduser("~/Documents/PukulEnam/brain-ctc-seg/notebooks")
os.makedirs(out_dir, exist_ok=True)

create_notebook(os.path.join(out_dir, "Architecture_SE2.ipynb"), se2_cells)
create_notebook(os.path.join(out_dir, "Architecture_HarmonicNet.ipynb"), harmonic_cells)
create_notebook(os.path.join(out_dir, "Architecture_nnUNet.ipynb"), nnunet_cells)

print("Notebooks successfully generated in ~/Documents/PukulEnam/brain-ctc-seg/notebooks/")
