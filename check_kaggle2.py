import os
import kagglehub
path = kagglehub.dataset_download("vbookshelf/computed-tomography-ct-images")
for root, dirs, files in os.walk(path):
    if any(f.endswith('.jpg') for f in files):
        print(f"DIR: {root}")
        print(f"  Files: {files[:10]}")
        break
