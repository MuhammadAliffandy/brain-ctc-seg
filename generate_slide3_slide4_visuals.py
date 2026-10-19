import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont

os.makedirs("presentation_assets/slide_cards", exist_ok=True)

# Generate synthetic realistic Brain CT scan
def get_brain_ct_volume():
    H, W = 256, 256
    y, x = np.mgrid[:H, :W]
    cx, cy = 128, 128

    brain_mask = ((x - cx) ** 2) / (96 ** 2) + ((y - cy) ** 2) / (110 ** 2) <= 1.0
    skull_outer = ((x - cx) ** 2) / (102 ** 2) + ((y - cy) ** 2) / (116 ** 2) <= 1.0
    skull_bone = skull_outer & (~brain_mask)

    slices = []
    # 3 slices: prev, curr, next
    for s_idx, z_shift in enumerate([-1, 0, 1]):
        img = np.zeros((H, W), dtype=np.float32)
        # Brain parenchyma
        img[brain_mask] = 0.38 + 0.03 * np.sin((x[brain_mask] + z_shift*4) / 14.0) + np.random.normal(0, 0.015, size=img[brain_mask].shape)
        
        # Ventricles
        v_mask = (((x - cx - 14) ** 2) / (10 ** 2) + ((y - cy) ** 2) / ((28 - abs(z_shift)*3) ** 2) <= 1.0) | \
                 (((x - cx + 14) ** 2) / (10 ** 2) + ((y - cy) ** 2) / ((28 - abs(z_shift)*3) ** 2) <= 1.0)
        img[v_mask] = 0.12 + np.random.normal(0, 0.01, size=img[v_mask].shape)
        img[skull_bone] = 0.95
        
        # Frontal-parietal Tumor
        tx, ty = cx + 36, cy - 22
        rx, ry = 22 - abs(z_shift)*2, 20 - abs(z_shift)*2
        lesion = (((x - tx) ** 2) / (rx ** 2) + ((y - ty) ** 2) / (ry ** 2) <= 1.0) & brain_mask
        img[lesion] = 0.78 + np.random.normal(0, 0.02, size=img[lesion].shape)
        slices.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
        if z_shift == 0:
            center_lesion = lesion
            center_coords = (tx, ty, rx, ry)
            
    return slices, center_lesion, center_coords

slices, true_lesion, (tx, ty, rx, ry) = get_brain_ct_volume()
curr_img = slices[1]
H, W = 256, 256
y, x = np.mgrid[:H, :W]

# =========================================================================
# 1. CARD 1: Objective Boundary Delineation (Matching Client Ref Card 1)
# =========================================================================
fig, (ax_wo, ax_w) = plt.subplots(1, 2, figsize=(7.2, 3.2), facecolor='#ffffff')
fig.subplots_adjust(wspace=0.15, left=0.03, right=0.97, top=0.88, bottom=0.15)

# Without
img_wo = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)
m1 = (((x - (tx+3)) ** 2) / ((rx+4) ** 2) + ((y - (ty-2)) ** 2) / ((ry+5) ** 2) <= 1.0).astype(np.uint8)
cnt1, _ = cv2.findContours(m1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_wo, cnt1, -1, (239, 68, 68), 2)

m2 = (((x - (tx-4)) ** 2) / ((rx-3) ** 2) + ((y - (ty+4)) ** 2) / ((ry-2) ** 2) <= 1.0).astype(np.uint8)
cnt2, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_wo, cnt2, -1, (245, 158, 11), 2)

m3 = (((x - (tx+1)) ** 2) / ((rx-2) ** 2) + ((y - ty) ** 2) / ((ry+1) ** 2) <= 1.0).astype(np.uint8)
cnt3, _ = cv2.findContours(m3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_wo, cnt3, -1, (59, 130, 246), 2)

crop_wo = img_wo[ty-40:ty+40, tx-40:tx+40]
ax_wo.imshow(crop_wo)
ax_wo.axis('off')
ax_wo.set_title("✖ Without (3 Clinicians)\nDice 65% / 75% / 70% ?", fontsize=10, color='#b91c1c', fontweight='bold', pad=4)
# Sub-caption
fig.text(0.26, 0.04, "Same scan, 3 readers, 3 conflicting contours", ha='center', fontsize=9, fontstyle='italic', color='#b91c1c')

# With Mod-Seg-SE(2)
img_w = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)
m_true = true_lesion.astype(np.uint8)
blended = np.where(m_true[..., None] > 0, (img_w * 0.35 + np.array([34, 197, 94]) * 0.65).astype(np.uint8), img_w)
cnt_true, _ = cv2.findContours(m_true, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(blended, cnt_true, -1, (22, 163, 74), 2)

crop_w = blended[ty-40:ty+40, tx-40:tx+40]
ax_w.imshow(crop_w)
ax_w.axis('off')
ax_w.set_title("✔ With Mod-Seg-SE(2)\nConsensus Dice 84.2% (Conf 0.94)", fontsize=10, color='#15803d', fontweight='bold', pad=4)
fig.text(0.74, 0.04, "One reproducible contour + consensus ROI", ha='center', fontsize=9, fontstyle='italic', color='#15803d')

plt.savefig("presentation_assets/slide_cards/card1_full_visual.png", dpi=220, bbox_inches='tight', facecolor='#ffffff')
plt.close()

# =========================================================================
# 2. CARD 2: Higher Diagnostic Accuracy (Matching Client Ref Card 2)
# =========================================================================
fig, (ax_wo, ax_w) = plt.subplots(1, 2, figsize=(7.2, 3.2), facecolor='#ffffff')
fig.subplots_adjust(wspace=0.25, left=0.05, right=0.95, top=0.88, bottom=0.15)

# Left: Without
ax_wo.set_facecolor('#fef2f2')
ax_wo.set_xlim(0, 100)
ax_wo.set_ylim(-0.6, 1.8)
ax_wo.axis('off')

# Bars
ax_wo.text(0, 1.35, "✖ Without", fontsize=11, fontweight='bold', color='#dc2626')
ax_wo.text(0, 0.95, "Clinician agreement", fontsize=8.5, color='#475569')
ax_wo.barh(0.65, 70, height=0.35, left=0, color='#fca5a5', edgecolor='#ef4444')
ax_wo.barh(0.65, 40, height=0.35, left=0, color='#ef4444')
ax_wo.text(45, 0.65, "40–70%", color='white', fontsize=8, fontweight='bold', va='center')

ax_wo.text(0, 0.25, "Baseline CNN (U-Net)", fontsize=8.5, color='#475569')
ax_wo.barh(-0.05, 74.2, height=0.35, left=0, color='#fb923c', edgecolor='#ea580c')
ax_wo.text(50, -0.05, "74.2%", color='white', fontsize=8, fontweight='bold', va='center')

fig.text(0.26, 0.04, "Failed on domain-shift & low contrast", ha='center', fontsize=9, fontstyle='italic', color='#b91c1c')

# Right: With
ax_w.set_facecolor('#f0fdf4')
ax_w.set_xlim(0, 100)
ax_w.set_ylim(-0.6, 1.8)
ax_w.axis('off')

ax_w.text(0, 1.35, "✔ With Mod-Seg-SE(2)", fontsize=11, fontweight='bold', color='#16a34a')
ax_w.text(0, 0.95, "Mod-Seg-SE(2) [OURS]", fontsize=8.5, color='#166534', fontweight='bold')
ax_w.barh(0.65, 84.2, height=0.35, left=0, color='#16a34a', edgecolor='#15803d')
ax_w.text(55, 0.65, "84.2% Dice", color='white', fontsize=8.5, fontweight='bold', va='center')

# Big stat
ax_w.text(50, -0.05, "84.2%", fontsize=28, fontweight='heavy', color='#15803d', ha='center', va='center')
ax_w.text(50, -0.38, "Dice Coefficient (73.1% IoU)", fontsize=8, color='#166534', ha='center', fontweight='bold')

fig.text(0.74, 0.04, "SOTA Dice accuracy on external NCCT cohort", ha='center', fontsize=9, fontstyle='italic', color='#15803d')

plt.savefig("presentation_assets/slide_cards/card2_full_visual.png", dpi=220, bbox_inches='tight', facecolor='#ffffff')
plt.close()

# =========================================================================
# 3. CARD 3: Rotation Invariance (Matching Client Ref Card 3)
# =========================================================================
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.2, 3.2), facecolor='#ffffff')
fig.subplots_adjust(wspace=0.15, left=0.03, right=0.97, top=0.88, bottom=0.15)

img_rot_wo = cv2.imread("presentation_assets/slide_cards/card3_rot_without.png")
img_rot_w = cv2.imread("presentation_assets/slide_cards/card3_rot_with.png")

ax_l.imshow(cv2.cvtColor(img_rot_wo, cv2.COLOR_BGR2RGB))
ax_l.axis('off')
ax_l.set_title("✖ Without (Conventional CNN)", fontsize=10, color='#b91c1c', fontweight='bold')
fig.text(0.26, 0.04, "Rotate / tilt: boundary collapses & flips (drop 17%)", ha='center', fontsize=9, fontstyle='italic', color='#b91c1c')

ax_r.imshow(cv2.cvtColor(img_rot_w, cv2.COLOR_BGR2RGB))
ax_r.axis('off')
ax_r.set_title("✔ With Mod-Seg-SE(2)", fontsize=10, color='#15803d', fontweight='bold')
fig.text(0.74, 0.04, "Rotation-invariant: same lesion, identical boundary", ha='center', fontsize=9, fontstyle='italic', color='#15803d')

plt.savefig("presentation_assets/slide_cards/card3_full_visual.png", dpi=220, bbox_inches='tight', facecolor='#ffffff')
plt.close()

# =========================================================================
# 4. CARD 4: Structured Reporting (Matching Client Ref Card 4)
# =========================================================================
fig, (ax_wo, ax_w) = plt.subplots(1, 2, figsize=(7.2, 3.2), facecolor='#ffffff')
fig.subplots_adjust(wspace=0.15, left=0.03, right=0.97, top=0.88, bottom=0.15)

# Left: Free-text note
ax_wo.set_facecolor('#fef2f2')
ax_wo.set_xlim(0, 10)
ax_wo.set_ylim(0, 10)
ax_wo.axis('off')
ax_wo.text(0.5, 9.0, "✖ Without", fontsize=11, fontweight='bold', color='#dc2626')
ax_wo.text(0.5, 7.8, "Subjective Physician Dictation:", fontsize=8.5, color='#991b1b', fontweight='bold')

quote_box = (
    "\"Mild-to-moderate mass effect in\n"
    "right frontal-parietal region, subtle\n"
    "hyperdensity?? Maybe ~15-25 mL,\n"
    "midline shift questionable\"\n"
    "— wording differs per physician"
)
p_q = FancyBboxPatch((0.4, 1.2), 9.2, 6.0, boxstyle="round,pad=0.2,rounding_size=0.4",
                     facecolor='#ffffff', edgecolor='#fca5a5', linewidth=1.2)
ax_wo.add_patch(p_q)
ax_wo.text(0.8, 4.2, quote_box, fontsize=8, color='#475569', fontstyle='italic', linespacing=1.3)
fig.text(0.26, 0.04, "Free-text notes, hard to compare over time", ha='center', fontsize=9, fontstyle='italic', color='#b91c1c')

# Right: Structured Volumetric Table
ax_w.set_facecolor('#f0fdf4')
ax_w.set_xlim(0, 10)
ax_w.set_ylim(0, 10)
ax_w.axis('off')
ax_w.text(0.5, 9.0, "✔ With Mod-Seg-SE(2)", fontsize=11, fontweight='bold', color='#16a34a')
ax_w.text(0.5, 7.8, "Automated Structured Volumetry:", fontsize=8.5, color='#166534', fontweight='bold')

# Mini Table
t_data = [
    ["Biomarker", "Value", "Status"],
    ["Tumor Volume", "18.4 mL", "High Risk"],
    ["Edema Core", "8.2 mL", "Moderate"],
    ["Sphericity", "0.82", "Compact"],
    ["Hemis. Shift", "2.1 mm", "Mild"]
]
p_t = FancyBboxPatch((0.4, 1.0), 9.2, 6.2, boxstyle="round,pad=0.2,rounding_size=0.4",
                     facecolor='#ffffff', edgecolor='#86efac', linewidth=1.2)
ax_w.add_patch(p_t)

table = ax_w.table(cellText=t_data, loc='center', bbox=[0.05, 0.12, 0.9, 0.62])
table.auto_set_font_size(False)
table.set_fontsize(7.5)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor('#cbd5e1')
    if r == 0:
        cell.set_facecolor('#15803d')
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')
    else:
        cell.set_facecolor('#f8fafc' if r % 2 == 1 else '#ffffff')
fig.text(0.74, 0.04, "Same quantitative fields every exam, easy to compare", ha='center', fontsize=9, fontstyle='italic', color='#15803d')

plt.savefig("presentation_assets/slide_cards/card4_full_visual.png", dpi=220, bbox_inches='tight', facecolor='#ffffff')
plt.close()

# =========================================================================
# 5. SLIDE 4 ARCHITECTURE: Detailed Visual Blocks & Flow
# =========================================================================
# Let's create an enriched visual architecture canvas
fig = plt.figure(figsize=(15, 8.2), facecolor='#ffffff')
ax = fig.add_subplot(111)
ax.set_facecolor('#ffffff')
ax.axis('off')

# Section 1: Flow across 5 Main Stages
stages = [
    {"title": "1. 2.5D Input Triplet", "desc": "• [img_prev, img_curr, img_next]\n• Tensor: [B, 3, 256, 256]\n• Inter-slice Z-depth context\n• Without heavy 3D-CNN cost", "x": 0.02, "w": 0.17, "col": "#0284c7", "bg": "#e0f2fe"},
    {"title": "2. C8 Group R2Conv", "desc": "• Discrete rotation group C8\n• 8 steerable filter banks (45°)\n• Pointwise MaxPool & GroupNorm\n• Equivariant: f(g·x) = g·f(x)", "x": 0.22, "w": 0.18, "col": "#2563eb", "bg": "#eff6ff"},
    {"title": "3. Equivariant Bottleneck", "desc": "• Deep invariant latent space\n• 192 equivariant channels\n• Orientation-agnostic features\n• Eliminates head-tilt bias", "x": 0.43, "w": 0.17, "col": "#7c3aed", "bg": "#f5f3ff"},
    {"title": "4. Equivariant Decoder", "desc": "• R2Upsampling (2x scale)\n• Skip direct-sum concatenation\n• Multi-scale edge refinement\n• Channels: 96 -> 48 -> 24", "x": 0.63, "w": 0.18, "col": "#2563eb", "bg": "#eff6ff"},
    {"title": "5. Group Projection Head", "desc": "• Equivariant -> Trivial (1x1)\n• Softmax classification\n• Output: [B, 2, 256, 256]\n• Background vs Tumor mask", "x": 0.84, "w": 0.15, "col": "#15803d", "bg": "#f0fdf4"}
]

for s in stages:
    patch = FancyBboxPatch((s["x"], 0.46), s["w"], 0.46, transform=ax.transAxes,
                           boxstyle="round,pad=0.015,rounding_size=0.025",
                           facecolor=s["bg"], edgecolor=s["col"], linewidth=2.0)
    ax.add_patch(patch)
    ax.text(s["x"] + s["w"]/2, 0.87, s["title"], transform=ax.transAxes,
            ha='center', va='center', fontsize=11, fontweight='bold', color=s["col"])
    ax.text(s["x"] + 0.01, 0.65, s["desc"], transform=ax.transAxes,
            ha='left', va='center', fontsize=9.2, color='#1e293b', linespacing=1.35)

# Horizontal Flow Arrows
arr = dict(arrowstyle="-|>", color="#64748b", lw=2.5, mutation_scale=16)
ax.annotate("", xy=(0.22, 0.69), xytext=(0.19, 0.69), xycoords='axes fraction', arrowprops=arr)
ax.annotate("", xy=(0.43, 0.69), xytext=(0.40, 0.69), xycoords='axes fraction', arrowprops=arr)
ax.annotate("", xy=(0.63, 0.69), xytext=(0.60, 0.69), xycoords='axes fraction', arrowprops=arr)
ax.annotate("", xy=(0.84, 0.69), xytext=(0.81, 0.69), xycoords='axes fraction', arrowprops=arr)

# Skip Connection Arch
skip_arr = dict(arrowstyle="-|>", color="#9333ea", lw=2.2, linestyle="--", connectionstyle="arc3,rad=-0.32", mutation_scale=15)
ax.annotate("", xy=(0.72, 0.92), xytext=(0.31, 0.92), xycoords='axes fraction', arrowprops=skip_arr)
ax.text(0.515, 0.98, "Equivariant Direct-Sum Skip Connections (Preserves Sub-Millimeter Boundaries)",
        transform=ax.transAxes, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#9333ea', backgroundcolor='#ffffff')

# Bottom Row: 3 Feature Blocks (Loss, Rotational Group Theory, Consensus Engine)
# Bottom Block 1: Combined Loss
b1 = FancyBboxPatch((0.02, 0.04), 0.31, 0.36, transform=ax.transAxes,
                    boxstyle="round,pad=0.015,rounding_size=0.02",
                    facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.5)
ax.add_patch(b1)
ax.text(0.04, 0.35, "Advanced Combined Loss", transform=ax.transAxes, fontsize=11, fontweight='bold', color='#0f2942')
loss_desc = (
    r"$\mathcal{L}_{\mathrm{total}} = 2.0 \cdot \mathcal{L}_{\mathrm{Dice}} + 0.5 \cdot \mathcal{L}_{\mathrm{Focal}} + 0.5 \cdot \mathcal{L}_{\mathrm{Edge}}$"
    "\n\n• Class Weighting [1.0, 10.0] tackles"
    "\n  extreme foreground/background imbalance"
    "\n• Edge Loss sharpens indistinct tumor boundaries"
    "\n• Focal Loss suppresses easy negative background"
)
ax.text(0.04, 0.18, loss_desc, transform=ax.transAxes, fontsize=9.2, color='#334155', linespacing=1.35)

# Bottom Block 2: C8 Group Theory Guarantee
b2 = FancyBboxPatch((0.35, 0.04), 0.31, 0.36, transform=ax.transAxes,
                    boxstyle="round,pad=0.015,rounding_size=0.02",
                    facecolor='#eff6ff', edgecolor='#93c5fd', linewidth=1.5)
ax.add_patch(b2)
ax.text(0.37, 0.35, "C8 Group Equivariance", transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1d4ed8')
c8_desc = (
    r"$\rho_{\mathrm{out}}(g)\left[\mathcal{K} \star f\right](x) = \left[\mathcal{K} \star \rho_{\mathrm{in}}(g)f\right](x)$"
    "\n\n• 8 Rotational symmetries spaced by 45°"
    "\n• Steerable harmonic basis filters"
    "\n• Mathematical guarantee: tumor segmented"
    "\n  identically regardless of patient head angle"
)
ax.text(0.37, 0.18, c8_desc, transform=ax.transAxes, fontsize=9.2, color='#1e3a8a', linespacing=1.35)

# Bottom Block 3: Consensus Engine & Volumetry
b3 = FancyBboxPatch((0.68, 0.04), 0.31, 0.36, transform=ax.transAxes,
                    boxstyle="round,pad=0.015,rounding_size=0.02",
                    facecolor='#f0fdf4', edgecolor='#86efac', linewidth=1.5)
ax.add_patch(b3)
ax.text(0.70, 0.35, "Multi-Model Consensus Engine", transform=ax.transAxes, fontsize=11, fontweight='bold', color='#15803d')
eng_desc = (
    "• 6 Benchmark Models Ensemble Agreement"
    "\n• Mod-Seg-SE(2) SOTA: 84.2% Dice | 73.1% IoU"
    "\n• Pixel-wise uncertainty thresholding at 0.80"
    "\n• Automated 3D Volumetric ROI calculation"
    "\n  (Tumor mL, Edema mL, Midline shift mm)"
)
ax.text(0.70, 0.18, eng_desc, transform=ax.transAxes, fontsize=9.2, color='#166534', linespacing=1.35)

plt.tight_layout()
plt.savefig("presentation_assets/slide_cards/arch_flow_detailed.png", dpi=250, bbox_inches='tight', facecolor='#ffffff')
plt.close()

print("All slide 3 & slide 4 visual assets generated successfully!")
