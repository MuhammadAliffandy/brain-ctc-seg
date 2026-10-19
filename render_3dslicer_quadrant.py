import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

vol = np.load("sample_dicom/genuine_3d_volume.npy")
mask = np.load("sample_dicom/genuine_3d_mask.npy")

nz, ny, nx = vol.shape
# Tumor center
zc, yc, xc = 38, 155, 92

# Function to window CT HU to uint8 [0, 255]
def window_ct(img, window_center=40, window_width=80):
    min_val = window_center - window_width / 2.0
    max_val = window_center + window_width / 2.0
    img_win = np.clip(img, min_val, max_val)
    img_win = (img_win - min_val) / (max_val - min_val) * 255.0
    return img_win.astype(np.uint8)

# 1. Axial view at z=zc
img_ax = window_ct(vol[zc, :, :])
mask_ax = mask[zc, :, :]
# Convert to BGR
ax_bgr = cv2.cvtColor(img_ax, cv2.COLOR_GRAY2BGR)
# Overlay tumor green with white boundary like 3D slicer
green_overlay = ax_bgr.copy()
green_overlay[mask_ax > 0] = [60, 160, 80] # Sage/forest green like reference
ax_bgr = cv2.addWeighted(green_overlay, 0.75, ax_bgr, 0.25, 0)
# Find contours and draw boundary in light green/white
contours, _ = cv2.findContours(mask_ax, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(ax_bgr, contours, -1, (200, 255, 200), 1)

# 2. Coronal view at y=yc
# In coronal: vertical is Z (up), horizontal is X (right-left)
# Radiologic convention: Patient Right is Image Left
img_cor = window_ct(vol[:, yc, :]) # shape (nz, nx)
mask_cor = mask[:, yc, :]
# Need to resize along Z to preserve aspect ratio (assuming isotropic display)
img_cor = cv2.resize(img_cor, (nx, ny), interpolation=cv2.INTER_CUBIC)
mask_cor = cv2.resize(mask_cor, (nx, ny), interpolation=cv2.INTER_NEAREST)
# Flip Z so top of head is UP
img_cor = np.flipud(img_cor)
mask_cor = np.flipud(mask_cor)

cor_bgr = cv2.cvtColor(img_cor, cv2.COLOR_GRAY2BGR)
green_overlay = cor_bgr.copy()
green_overlay[mask_cor > 0] = [60, 160, 80]
cor_bgr = cv2.addWeighted(green_overlay, 0.75, cor_bgr, 0.25, 0)
contours_cor, _ = cv2.findContours(mask_cor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(cor_bgr, contours_cor, -1, (200, 255, 200), 1)

# 3. Sagittal view at x=xc
# In sagittal: vertical is Z (up), horizontal is Y (anterior-posterior)
# Client reference: Face/nose is looking to the LEFT!
# Anterior (y small) should be on the Left.
img_sag = window_ct(vol[:, :, xc]) # shape (nz, ny)
mask_sag = mask[:, :, xc]
img_sag = cv2.resize(img_sag, (ny, ny), interpolation=cv2.INTER_CUBIC)
mask_sag = cv2.resize(mask_sag, (ny, ny), interpolation=cv2.INTER_NEAREST)
# Flip Z so top is UP
img_sag = np.flipud(img_sag)
mask_sag = np.flipud(mask_sag)

sag_bgr = cv2.cvtColor(img_sag, cv2.COLOR_GRAY2BGR)
green_overlay = sag_bgr.copy()
green_overlay[mask_sag > 0] = [60, 160, 80]
sag_bgr = cv2.addWeighted(green_overlay, 0.75, sag_bgr, 0.25, 0)
contours_sag, _ = cv2.findContours(mask_sag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(sag_bgr, contours_sag, -1, (200, 255, 200), 1)

# 4. Generate 3D Slicer Viewport (Kanan Atas) with Purple Cube and 3D Organic Tumor Surface Mesh!
fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('#6a708e') # Lavender/grey slate background like 3D Slicer
ax.set_facecolor('#6a708e')

# Marching cubes on tumor mask
verts, faces, normals, values = measure.marching_cubes(mask, level=0.5, spacing=(1.0, 1.0, 1.0))
# Plot 3D surface
mesh = ax.plot_trisurf(verts[:, 2], verts[:, 1], faces, verts[:, 0],
                       color='#4ea365', edgecolor='#2e633d', lw=0.1, alpha=0.95, shade=True)

# Draw Magenta 3D Bounding Box around ROI
ax.set_xlim([0, nx])
ax.set_ylim([0, ny])
ax.set_zlim([0, nz])

# Draw bounding box lines in magenta/pink
for s, e in [([0, 0, 0], [nx, 0, 0]), ([0, ny, 0], [nx, ny, 0]), ([0, 0, nz], [nx, 0, nz]), ([0, ny, nz], [nx, ny, nz]),
            ([0, 0, 0], [0, ny, 0]), ([nx, 0, 0], [nx, ny, 0]), ([0, 0, nz], [0, ny, nz]), ([nx, 0, nz], [nx, ny, nz]),
            ([0, 0, 0], [0, 0, nz]), ([nx, 0, 0], [nx, 0, nz]), ([0, ny, 0], [0, ny, nz]), ([nx, ny, 0], [nx, ny, nz])]:
    ax.plot3D(*zip(s, e), color='#d13fa8', lw=1.2)

# Set labels R, L, I, S
ax.text(nx + 10, ny/2, nz/2, 'L', color='white', fontsize=14, weight='bold')
ax.text(-20, ny/2, nz/2, 'R', color='white', fontsize=14, weight='bold')
ax.text(nx/2, ny/2, -10, 'I', color='white', fontsize=14, weight='bold')

ax.axis('off')
ax.view_init(elev=20, azim=45)
plt.tight_layout(pad=0)
plt.savefig("sample_dicom/slicer_3d_box.png", facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
plt.close()

# Load rendered 3D box image and resize to (ny, ny)
three_d_bgr = cv2.imread("sample_dicom/slicer_3d_box.png")
three_d_bgr = cv2.resize(three_d_bgr, (ny, ny))

# Add 3D Slicer style sliders and corner HUDs
def add_slicer_hud(img, title, color_bar=(0, 200, 0)):
    h, w = img.shape[:2]
    # Header bar
    cv2.rectangle(img, (0, 0), (w, 16), (20, 20, 20), -1)
    cv2.putText(img, title, (8, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    # Slicer bottom status bar
    cv2.rectangle(img, (0, h - 18), (w, h), (30, 30, 30), -1)
    # Colored thin line above status
    cv2.rectangle(img, (0, h - 20), (w, h - 18), color_bar, -1)
    cv2.putText(img, "PATIENT^NTUH | L:40 W:80 | 1.0mm", (8, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    return img

ax_bgr = add_slicer_hud(ax_bgr, "Axial (Red Slice: z=38)", (0, 0, 255))
cor_bgr = add_slicer_hud(cor_bgr, "Coronal (Green Slice: y=155)", (0, 255, 0))
sag_bgr = add_slicer_hud(sag_bgr, "Sagittal (Yellow Slice: x=92)", (0, 255, 255))

# Top row: Axial + 3D
top_row = np.hstack([ax_bgr, three_d_bgr])
# Bottom row: Coronal + Sagittal
bot_row = np.hstack([cor_bgr, sag_bgr])
# Full 4-quadrant collage
collage = np.vstack([top_row, bot_row])

# Save collage
cv2.imwrite("sample_dicom/genuine_3dslicer_quadrant.png", collage)
# Also copy to artifacts directory
os.system("cp sample_dicom/genuine_3dslicer_quadrant.png /Users/aliffandy/.gemini/antigravity-ide/brain/b8d523fb-4850-4c6d-ad2b-f2542f00798d/genuine_3dslicer_quadrant.png")
print("Rendering finished. Output saved to sample_dicom/genuine_3dslicer_quadrant.png")
