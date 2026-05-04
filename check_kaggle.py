import kagglehub
import os

path = kagglehub.dataset_download("vbookshelf/computed-tomography-ct-images")
print(f"Downloaded to {path}")
for root, dirs, files in os.walk(path):
    print(f"DIR: {root}")
    for f in files[:3]: print(f"  f: {f}")
    if len(files) > 0: break
