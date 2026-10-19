import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. C8 Rotation Rosette (300x300)
fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor='#eff6ff')
ax.set_facecolor('#eff6ff')
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.25, 1.25)
ax.axis('off')

# Outer circle guide
circle = plt.Circle((0, 0), 1.0, color='#bfdbfe', fill=False, linestyle='--', linewidth=1.5)
ax.add_patch(circle)

for ang_i in range(8):
    theta = ang_i * (2 * np.pi / 8)
    dx = 0.88 * np.cos(theta)
    dy = 0.88 * np.sin(theta)
    ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#1d4ed8", lw=2.2, mutation_scale=12))
    deg = int(ang_i * 45)
    tx = 1.10 * np.cos(theta)
    ty = 1.10 * np.sin(theta)
    ax.text(tx, ty, f"{deg}°", fontsize=9.5, ha='center', va='center', color='#1e3a8a', fontweight='bold')

ax.plot(0, 0, 'o', color='#1d4ed8', markersize=8)
ax.text(0, -0.22, "C8 Group", fontsize=8.5, ha='center', color='#1e3a8a', fontweight='bold')

plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
plt.savefig("presentation_assets/c8_rotation_roset.png", dpi=250, facecolor='#eff6ff')
plt.close()

# 2. Deep Latent Feature Map (Bottleneck)
fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor='#f5f3ff')
ax.set_facecolor('#f5f3ff')
ax.axis('off')
feat = np.zeros((32, 32))
for i in range(4):
    for j in range(4):
        val = np.sin((i+1)*1.2) * np.cos((j+1)*0.9)
        feat[i*8:(i+1)*8, j*8:(j+1)*8] = val
ax.imshow(feat, cmap='magma')
ax.set_title("192 Equivariant\nChannels", fontsize=9.5, fontweight='bold', color='#6d28d9', pad=6)
plt.subplots_adjust(left=0.05, right=0.95, top=0.82, bottom=0.05)
plt.savefig("presentation_assets/deep_latent_feature.png", dpi=250, facecolor='#f5f3ff')
plt.close()

# 3. Decoder Upsampled Feature Map
fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor='#eff6ff')
ax.set_facecolor('#eff6ff')
ax.axis('off')
up_feat = cv2.resize(feat, (64, 64), interpolation=cv2.INTER_LINEAR)
ax.imshow(up_feat, cmap='viridis')
ax.set_title("Multi-Scale\nRefinement", fontsize=9.5, fontweight='bold', color='#1d4ed8', pad=6)
plt.subplots_adjust(left=0.05, right=0.95, top=0.82, bottom=0.05)
plt.savefig("presentation_assets/decoder_feature.png", dpi=250, facecolor='#eff6ff')
plt.close()

print("Architecture component graphics generated successfully!")
