import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import timm

# =====================================================================
# 1. KONFIGURASI (UBAH SESUAI PATH DI SERVER DGX)
# =====================================================================
NUM_CLASSES = 4 # MES 0, 1, 2, 3
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATHS = {
    'ResNet-50': '/path/to/resnet50.pth',
    'DenseNet-121': '/path/to/densenet121.pth',
    'EfficientNet-B4': '/path/to/efficientnet_b4.pth',
    'ConvNeXt-Tiny': '/path/to/convnext_tiny.pth',
    'ViT-B/16': '/path/to/vit_b_16.pth'
}

DATASET_CSV_OR_FOLDER = '/path/to/test_dataset' 

# =====================================================================
# 2. FUNGSI INISIALISASI MODEL
# =====================================================================
def load_models():
    models_dict = {}
    
    # 1. ResNet50
    m1 = models.resnet50(pretrained=False)
    m1.fc = torch.nn.Linear(m1.fc.in_features, NUM_CLASSES)
    # m1.load_state_dict(torch.load(MODEL_PATHS['ResNet-50'], map_location=DEVICE))
    m1.to(DEVICE).eval()
    models_dict['ResNet-50'] = {'model': m1, 'target_layer': [m1.layer4[-1]]}
    
    # 2. DenseNet121
    m2 = models.densenet121(pretrained=False)
    m2.classifier = torch.nn.Linear(m2.classifier.in_features, NUM_CLASSES)
    # m2.load_state_dict(torch.load(MODEL_PATHS['DenseNet-121'], map_location=DEVICE))
    m2.to(DEVICE).eval()
    models_dict['DenseNet-121'] = {'model': m2, 'target_layer': [m2.features[-1]]}
    
    # 3. EfficientNet-B4
    m3 = models.efficientnet_b4(pretrained=False)
    m3.classifier[1] = torch.nn.Linear(m3.classifier[1].in_features, NUM_CLASSES)
    # m3.load_state_dict(torch.load(MODEL_PATHS['EfficientNet-B4'], map_location=DEVICE))
    m3.to(DEVICE).eval()
    models_dict['EfficientNet-B4'] = {'model': m3, 'target_layer': [m3.features[-1]]}
    
    # 4. ConvNeXt-Tiny
    m4 = models.convnext_tiny(pretrained=False)
    m4.classifier[2] = torch.nn.Linear(m4.classifier[2].in_features, NUM_CLASSES)
    # m4.load_state_dict(torch.load(MODEL_PATHS['ConvNeXt-Tiny'], map_location=DEVICE))
    m4.to(DEVICE).eval()
    models_dict['ConvNeXt-Tiny'] = {'model': m4, 'target_layer': [m4.features[-1]]}
    
    # 5. ViT-B/16 (GradCAM untuk ViT sedikit berbeda, target layer di blocks terakhir)
    m5 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
    # m5.load_state_dict(torch.load(MODEL_PATHS['ViT-B/16'], map_location=DEVICE))
    m5.to(DEVICE).eval()
    models_dict['ViT-B/16'] = {'model': m5, 'target_layer': [m5.blocks[-1].norm1]} 

    return models_dict

# =====================================================================
# 3. FUNGSI LOGIKA ENSEMBLE UNTUK MENCARI KASUS
# =====================================================================
def get_ensemble_prediction(preds, probs):
    """ preds: list of 5 ints, probs: list of 5 arrays """
    counts = np.bincount(preds, minlength=NUM_CLASSES)
    max_votes = np.max(counts)
    winners = np.where(counts == max_votes)[0]
    
    if len(winners) == 1:
        # Majority / Unanimous
        return winners[0], counts
    else:
        # Tie-break (Mean-probability tie-break)
        mean_probs = np.mean(probs, axis=0)
        return winners[np.argmax(mean_probs[winners])], counts

def find_representative_cases(models_dict, dataloader):
    """
    Fungsi ini harus diisi dengan iterasi ke dataset untuk mencari 5 contoh gambar:
    1. Unanimous: Semua 5 model benar (5/5).
    2. Majority rescue: 3 model benar, 2 salah.
    3. Tie-break: 2 model tebak A, 2 tebak B, 1 tebak C. (Ensemble benar).
    4. Safety fallback: 2:1:1:1 (Ensemble benar).
    5. Ensemble error: Majority salah (Misal 3/5 model salah).
    
    Sebagai contoh, kita akan kembalikan dictionary dummy path gambar.
    Di server, ganti bagian ini dengan iterasi riil!
    """
    # DUMMY RETURN (Ganti dengan logic pencarian dataset betulan)
    return {
        'Unanimous': {'path': 'dummy.jpg', 'label': 3},
        'Majority rescue': {'path': 'dummy.jpg', 'label': 1},
        'Tie-break': {'path': 'dummy.jpg', 'label': 2},
        'Safety fallback': {'path': 'dummy.jpg', 'label': 3},
        'Ensemble error': {'path': 'dummy.jpg', 'label': 1}
    }

# =====================================================================
# 4. FUNGSI PLOTTING GAMBAR (MIMIC THE REFERENCE FIGURE)
# =====================================================================
def generate_figure(models_dict, cases):
    fig = plt.figure(figsize=(24, 14))
    
    # Grid layout: 5 Baris (Cases), 8 Kolom (CaseName, RefImage, 5 Models, EnsembleLogic)
    gs = GridSpec(5, 8, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1.5], wspace=0.1, hspace=0.3)
    
    model_names = list(models_dict.keys())
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for row_idx, (case_name, case_data) in enumerate(cases.items()):
        img_path = case_data['path']
        gt_label = case_data['label']
        
        # Load Image (Simulasi gambar kosong jika path dummy)
        if not os.path.exists(img_path):
            rgb_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        else:
            rgb_img = np.array(Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
            
        input_tensor = transform(Image.fromarray((rgb_img * 255).astype(np.uint8))).unsqueeze(0).to(DEVICE)
        
        # 1. Plot Text Case Name
        ax_name = fig.add_subplot(gs[row_idx, 0])
        ax_name.axis('off')
        ax_name.text(0.5, 0.6, case_name, fontsize=14, fontweight='bold', ha='center', va='center')
        ax_name.text(0.5, 0.4, f"MES {gt_label}", fontsize=12, ha='center', va='center', 
                     bbox=dict(facecolor='red' if 'error' in case_name else 'green', alpha=0.2))
        
        # 2. Plot Reference Image
        ax_ref = fig.add_subplot(gs[row_idx, 1])
        ax_ref.imshow(rgb_img)
        ax_ref.axis('off')
        if row_idx == 0: ax_ref.set_title("Reference", fontweight='bold')
        
        # 3. Plot 5 Models (Grad-CAM)
        preds = []
        probs_list = []
        
        for col_idx, m_name in enumerate(model_names):
            model_info = models_dict[m_name]
            model = model_info['model']
            target_layers = model_info['target_layer']
            
            # Forward pass
            with torch.no_grad():
                logits = model(input_tensor)
                prob = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred = np.argmax(prob)
                preds.append(pred)
                probs_list.append(prob)
            
            # GradCAM (gunakan block try-except jika ViT butuh reshape khusus)
            try:
                cam = GradCAM(model=model, target_layers=target_layers)
                targets = [ClassifierOutputTarget(gt_label)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            except:
                cam_image = rgb_img # Fallback jika error (misal ViT perlu reshape2D)

            ax_cam = fig.add_subplot(gs[row_idx, col_idx + 2])
            ax_cam.imshow(cam_image)
            ax_cam.axis('off')
            
            # Label Prob & Match
            match_symbol = "✔️" if pred == gt_label else "❌"
            ax_cam.text(0.05, 0.05, f"MES {pred} - {prob[pred]:.2f} {match_symbol}", 
                        transform=ax_cam.transAxes, color='white', 
                        fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
            
            if row_idx == 0: ax_cam.set_title(m_name, fontweight='bold')
            
        # 4. Plot Ensemble Logic
        final_pred, votes = get_ensemble_prediction(preds, probs_list)
        ax_ens = fig.add_subplot(gs[row_idx, 7])
        ax_ens.axis('off')
        
        vote_str = "/".join(map(str, sorted([v for v in votes if v > 0], reverse=True)))
        ens_match = "✔️" if final_pred == gt_label else "❌"
        ax_ens.text(0.1, 0.6, f"{vote_str} ➡️ MES {final_pred}  {ens_match}", fontsize=14, fontweight='bold')
        ax_ens.text(0.1, 0.4, f"Action: MES {final_pred}", fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.3))
        
        if row_idx == 0: ax_ens.set_title("Adjudication Agent", fontweight='bold')

    plt.tight_layout()
    plt.savefig('Ensemble_GradCAM_Figure.png', dpi=300, bbox_inches='tight')
    print("✅ Gambar berhasil disimpan sebagai 'Ensemble_GradCAM_Figure.png'")

if __name__ == "__main__":
    print("Memuat model...")
    models_dict = load_models()
    print("Mencari sampel dari dataset (DUMMY MODE)...")
    cases = find_representative_cases(models_dict, None)
    print("Membuat gambar Grad-CAM...")
    generate_figure(models_dict, cases)
