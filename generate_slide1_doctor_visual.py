import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

# Generate synthetic brain CT scan
H, W = 160, 160
y, x = np.mgrid[:H, :W]
cx, cy = 80, 80

brain_mask = ((x - cx) ** 2) / (60 ** 2) + ((y - cy) ** 2) / (70 ** 2) <= 1.0
skull = (((x - cx) ** 2) / (64 ** 2) + ((y - cy) ** 2) / (74 ** 2) <= 1.0) & (~brain_mask)

img1 = np.zeros((H, W), dtype=np.float32)
img1[brain_mask] = 0.38 + 0.03 * np.sin(x[brain_mask]/10.0)
img1[skull] = 0.95
tx, ty = cx + 22, cy - 14
lesion1 = (((x - tx)**2)/(14**2) + ((y - ty)**2)/(12**2) <= 1.0) & brain_mask
img1[lesion1] = 0.75

img2 = np.zeros((H, W), dtype=np.float32)
img2[brain_mask] = 0.38 + 0.03 * np.sin(x[brain_mask]/10.0)
img2[skull] = 0.95
lesion2 = (((x - tx)**2)/(18**2) + ((y - ty)**2)/(15**2) <= 1.0) & brain_mask
img2[lesion2] = 0.72

fig, ax = plt.subplots(figsize=(4.0, 2.5), facecolor='#ffffff')
ax.set_facecolor('#ffffff')
ax.axis('off')

# Display 2 scans stacked
sub1 = fig.add_axes([0.05, 0.52, 0.40, 0.42])
sub1.imshow(img1, cmap='bone')
sub1.axis('off')
sub1.patch.set_edgecolor('#cbd5e1')
sub1.patch.set_linewidth(1.0)

sub2 = fig.add_axes([0.05, 0.06, 0.40, 0.42])
sub2.imshow(img2, cmap='bone')
sub2.axis('off')
sub2.patch.set_edgecolor('#cbd5e1')
sub2.patch.set_linewidth(1.0)

# Speech Bubble for Doctor Confusion
p_b = FancyBboxPatch((0.52, 0.20), 0.42, 0.65, transform=ax.transAxes,
                     boxstyle="round,pad=0.03,rounding_size=0.15",
                     facecolor='#0284c7', edgecolor='#0369a1', linewidth=1.5)
ax.add_patch(p_b)

# Bubble pointer tail
poly = Polygon([[0.54, 0.35], [0.47, 0.38], [0.54, 0.45]], transform=ax.transAxes,
               facecolor='#0284c7', edgecolor='#0369a1', linewidth=1.0)
ax.add_patch(poly)

ax.text(0.73, 0.64, "Dice 65%?", transform=ax.transAxes, fontsize=10, fontweight='bold', color='white', ha='center')
ax.text(0.73, 0.48, "Dice 75%?", transform=ax.transAxes, fontsize=10, fontweight='bold', color='white', ha='center')
ax.text(0.73, 0.32, "Tumor vs Edema?", transform=ax.transAxes, fontsize=8.5, fontweight='bold', color='white', ha='center')

plt.savefig("presentation_assets/slide1_doctor_visual.png", dpi=220, bbox_inches='tight', facecolor='#ffffff')
plt.close()
print("Saved presentation_assets/slide1_doctor_visual.png")
