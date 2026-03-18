import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

# ─── E2CNN / ESCNN imports ────────────────────────────────────────────────────
from escnn import gspaces
import escnn.nn as enn

# ─── Model path (training/ folder, relative to project root) ─────────────────
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "training")

# ─── RE-DEFINE the exact same architecture as in train.py ────────────────────

class DoubleEquivariantConv(nn.Module):
    """Blok konvolusi ganda yang equivariant."""
    def __init__(self, in_type, out_type, mid_type=None):
        super().__init__()
        if not mid_type:
            mid_type = out_type
        self.double_conv = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(mid_type),
            enn.ReLU(mid_type, inplace=True),
            enn.R2Conv(mid_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Blok downsampling menggunakan MaxPool diikuti DoubleEquivariantConv."""
    def __init__(self, in_type, out_type):
        super().__init__()
        self.pool = enn.PointwiseMaxPool(in_type, kernel_size=2)
        self.conv = DoubleEquivariantConv(in_type, out_type)
    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class Up(nn.Module):
    """Blok upsampling diikuti penggabungan skip connection dan DoubleEquivariantConv."""
    def __init__(self, in_type, out_type):
        super().__init__()
        self.up = enn.R2Upsampling(in_type, scale_factor=2, mode='bilinear', align_corners=True)
        # Tipe input untuk konvolusi adalah gabungan dari tensor setelah upsampling dan tensor dari skip connection
        self.conv = DoubleEquivariantConv(in_type + out_type, out_type)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Menggabungkan tensor dari skip connection (x2) dan tensor yang di-upsample (x1)
        x = enn.tensor_directsum([x2, x1])
        return self.conv(x)

class OutConv(nn.Module):
    """Konvolusi 1x1 di akhir untuk memetakan fitur ke jumlah kelas output."""
    def __init__(self, in_type, n_classes):
        super().__init__()
        gspace = in_type.gspace
        # Tipe output adalah trivial representation, karena output segmentasi harus invarian terhadap rotasi
        out_type = enn.FieldType(gspace, n_classes * [gspace.trivial_repr])
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class SE2_CNNET(nn.Module):
    """
    Arsitektur U-Net Equivariant SE(2) untuk segmentasi.
    N: Jumlah rotasi diskrit yang akan dipertimbangkan (misal, N=8 untuk rotasi kelipatan 45 derajat).
    base_channels: Jumlah channel dasar pada lapisan pertama.
    """
    def __init__(self, n_channels=1, n_classes=2, N=8, base_channels=24):
        super().__init__()
        self.r2_act = gspaces.rot2dOnR2(N=N)
        c = base_channels

        # Parameters for Grad-CAM
        self.gradients = None
        self.activations = None

        # Mendefinisikan tipe field untuk setiap level kedalaman U-Net
        self.feat_type_in = enn.FieldType(self.r2_act, n_channels * [self.r2_act.trivial_repr])
        self.feat_type_1 = enn.FieldType(self.r2_act, c * [self.r2_act.regular_repr])
        self.feat_type_2 = enn.FieldType(self.r2_act, (c*2) * [self.r2_act.regular_repr])
        self.feat_type_3 = enn.FieldType(self.r2_act, (c*4) * [self.r2_act.regular_repr])
        self.feat_type_4 = enn.FieldType(self.r2_act, (c*8) * [self.r2_act.regular_repr])
        self.feat_type_5 = enn.FieldType(self.r2_act, (c*16) * [self.r2_act.regular_repr])

        # Encoder Path
        self.inc = DoubleEquivariantConv(self.feat_type_in, self.feat_type_1)
        self.down1 = Down(self.feat_type_1, self.feat_type_2)
        self.down2 = Down(self.feat_type_2, self.feat_type_3)
        self.down3 = Down(self.feat_type_3, self.feat_type_4)
        self.down4 = Down(self.feat_type_4, self.feat_type_5)

        # Decoder Path
        self.up1 = Up(self.feat_type_5, self.feat_type_4)
        self.up2 = Up(self.feat_type_4, self.feat_type_3)
        self.up3 = Up(self.feat_type_3, self.feat_type_2)
        self.up4 = Up(self.feat_type_2, self.feat_type_1)

        # Output Layer
        self.outc = OutConv(self.feat_type_1, n_classes)

    def hook_activations(self, x):
        self.activations = x

    def hook_gradients(self, x):
        self.gradients = x

    def forward(self, x):
        # Konversi input tensor menjadi GeometricTensor
        x_geom = enn.GeometricTensor(x, self.feat_type_in)

        # Encoder
        x1 = self.inc(x_geom)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Capture Hook for Grad-CAM
        x_tensor = x.tensor
        if x_tensor.requires_grad:
            x_tensor.register_hook(self.hook_gradients)
        self.hook_activations(x_tensor)

        # Mengembalikan tensor biasa untuk dihitung loss-nya
        # Create a new GeometricTensor object from the captured tensor, since OutConv expects a GeometricTensor
        x_out = enn.GeometricTensor(x_tensor, self.feat_type_1)
        logits = self.outc(x_out).tensor
        return logits


# ─── Model cache so we don't reload on every button press ────────────────────
_model_cache: dict = {}

def _load_model(epoch: int) -> SE2_CNNET:
    """Load (and cache) the SE2_CNNET weights for the requested epoch."""
    if epoch in _model_cache:
        return _model_cache[epoch]

    weight_path = os.path.join(MODELS_DIR, f"model_epoch_{epoch}.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Model weights for epoch {epoch} not found at: {weight_path}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SE2_CNNET(n_channels=1, n_classes=2, N=8, base_channels=24)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    # We set to train initially if we want to enable grads. But best practice is to eval() 
    # and just use torch.enable_grad() later for the image.
    model.eval()

    _model_cache[epoch] = model
    return model


# ─── Public inference function ────────────────────────────────────────────────

def predict_segmentation(image_file, modality: str, epoch: int = 5, enable_gradcam: bool = False):
    """
    Run real SE2-CNNET inference for brain CTC segmentation.

    Args:
        image_file     : UploadedFile (Streamlit) or file-like object.
        modality       : Scan type string — informational only, passed to caller.
        epoch          : Which saved epoch weights to use (1-5).
        enable_gradcam : Whether to generate an explainability heatmap.

    Returns:
        original_image : PIL.Image  — grayscale original resized to 256×256.
        mask_image     : PIL.Image  — predicted binary segmentation mask (0 / 255).
        cam_image      : PIL.Image or None — heatmap overlaid on the original image if requested.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load & preprocess image ──────────────────────────────────────────
    image = Image.open(image_file).convert("L")                 # grayscale
    image_256 = image.resize((256, 256), Image.BILINEAR)

    img_array = np.array(image_256, dtype=np.float32)

    # Normalise to [0, 1]
    if img_array.max() > img_array.min():
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    else:
        img_array = np.zeros_like(img_array)

    # Shape: [1, 1, 256, 256]
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

    # ── 2. Load model ────────────────────────────────────────────────────────
    model = _load_model(epoch)

    # ── 3. Inference ─────────────────────────────────────────────────────────
    cam_image = None

    if enable_gradcam:
        img_tensor.requires_grad_(True)
        # Enable grad calculation even though model is in eval mode
        with torch.enable_grad():
            logits = model(img_tensor)          # [1, 2, 256, 256]
            probs  = F.softmax(logits, dim=1)   # class probabilities
            pred   = torch.argmax(probs, dim=1) # [1, 256, 256]  → 0 or 1

            # Get the score for the Target class (class 1)
            # We want to see why the model thinks a pixel is Lesion (1).
            # We sum all the logits for class 1 to get a single scalar to backpropagate.
            target_class_score = logits[:, 1, :, :].sum()
            model.zero_grad()
            target_class_score.backward()

            # Grad-CAM Calculation
            gradients = model.gradients  # [1, C, H, W]
            activations = model.activations # [1, C, H, W]

            if gradients is not None and activations is not None:
                # Global average pooling on the gradients (per channel)
                pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) # [C]

                # Weight the channels by the corresponding gradients
                for i in range(activations.size(1)):
                    activations[:, i, :, :] *= pooled_gradients[i]

                # Average the channels of the activations
                heatmap = torch.mean(activations, dim=1).squeeze(0).cpu().detach().numpy() # [H, W]

                # ReLU on heatmap to only keep positive influence
                heatmap = np.maximum(heatmap, 0)

                # Normalize the heatmap
                if np.max(heatmap) > 0:
                    heatmap /= np.max(heatmap)

                # Resize heatmap to match image size
                heatmap = cv2.resize(heatmap, (256, 256))
                
                # Convert to uint8 (0-255)
                heatmap = np.uint8(255 * heatmap)

                # Apply colormap (JET)
                colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                # Convert original grayscale image back to BGR for blending
                orig_img_cv = np.array(image_256)
                orig_img_cv = cv2.cvtColor(orig_img_cv, cv2.COLOR_GRAY2BGR)

                # Overlay heatmap on original image (0.4 alpha for heatmap, 0.6 for image)
                superimposed_img = cv2.addWeighted(colormap, 0.4, orig_img_cv, 0.6, 0)
                
                # Convert BGR back to RGB for PIL
                superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
                cam_image = Image.fromarray(superimposed_img_rgb)
    else:
        with torch.no_grad():
            logits = model(img_tensor)          # [1, 2, 256, 256]
            probs  = F.softmax(logits, dim=1)   # class probabilities
            pred   = torch.argmax(probs, dim=1) # [1, 256, 256]  → 0 or 1

    pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)  # 0 or 1
    mask_array = (pred_np * 255).astype(np.uint8)              # 0 or 255

    mask_image = Image.fromarray(mask_array)

    return image_256, mask_image, cam_image

