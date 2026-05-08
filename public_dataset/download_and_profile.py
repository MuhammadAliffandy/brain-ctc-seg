import os
import sys

try:
    import kagglehub
except ImportError:
    print("❌ Error: kagglehub library not found.")
    print("Please install it first using: pip install kagglehub[pandas-datasets]")
    sys.exit(1)

def main():
    print("="*60)
    print("📥 DOWNLOADING & PROFILING KAGGLE DATASET")
    print("Dataset: ozguraslank/brain-stroke-ct-dataset")
    print("="*60)
    
    try:
        # Download the dataset (kagglehub caches it automatically)
        print("\nDownloading dataset (this may take a while if not cached)...")
        path = kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset")
        print(f"\n✅ Dataset successfully downloaded/located at:\n{path}\n")
        
        # Profile the directory structure
        print("📂 Directory Structure & Sample Files:")
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            
            if len(files) > 0:
                print(f"{subindent}({len(files)} files found)")
                # Print first 3 files to see extensions
                for f in files[:3]:
                    print(f"{subindent}📄 {f}")
                if len(files) > 3:
                    print(f"{subindent}... and {len(files)-3} more files")
            
    except Exception as e:
        print(f"❌ Failed to process dataset: {e}")

if __name__ == "__main__":
    main()
