import os
import kagglehub
path = kagglehub.dataset_download("vbookshelf/computed-tomography-ct-images")
masks = []
for root, dirs, files in os.walk(path):
    for f in files:
        if 'mask' in f.lower() or 'seg' in f.lower() or 'label' in f.lower() or 'ann' in f.lower() or 'hem' in f.lower():
            masks.append(os.path.join(root, f))

print(f"Total potential mask/labels: {len(masks)}")
print("Sample:")
for m in masks[:20]:
    print(m)
