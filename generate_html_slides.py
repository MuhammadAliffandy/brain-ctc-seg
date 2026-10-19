import os
import base64

def get_base64_img(img_path):
    if not os.path.exists(img_path): return ""
    with open(img_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

logo_osl_b64 = get_base64_img("presentation_assets/logo_osl.png")
logo_ntu_b64 = get_base64_img("presentation_assets/logo_ntu.png")
mpr_b64 = get_base64_img("presentation_assets/dashboard_mpr_consensus_1792346200853.png")
card1_b64 = get_base64_img("presentation_assets/slide_cards/card1_full_visual.png")
card2_b64 = get_base64_img("presentation_assets/slide_cards/card2_full_visual.png")
card3_b64 = get_base64_img("presentation_assets/slide_cards/card3_full_visual.png")
card4_b64 = get_base64_img("presentation_assets/slide_cards/card4_full_visual.png")
ct_mach_b64 = get_base64_img("presentation_assets/ct_scanner_machine.jpg")
sw_b64 = get_base64_img("presentation_assets/ref_software.png")
doc_b64 = get_base64_img("presentation_assets/slide1_doctor_visual.png")

# Slide 4 images
st1_b64 = get_base64_img("presentation_assets/real_raw_ct_input.png")
st2_b64 = get_base64_img("presentation_assets/c8_rotation_roset.png")
st3_b64 = get_base64_img("presentation_assets/deep_latent_feature.png")
st4_b64 = get_base64_img("presentation_assets/decoder_feature.png")
st5_b64 = get_base64_img("presentation_assets/real_se2_pred_output.png")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brain-CT-Seg Client Presentation (4 Slides Deck)</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }}
    
    body {{
        background: #090d16;
        color: #1e293b;
        font-family: 'Plus Jakarta Sans', sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 40px 20px;
        gap: 40px;
    }}

    .nav-header {{
        position: sticky;
        top: 20px;
        z-index: 100;
        background: rgba(15, 23, 42, 0.90);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 999px;
        padding: 12px 28px;
        display: flex;
        align-items: center;
        gap: 20px;
        color: white;
        box-shadow: 0 10px 35px rgba(0,0,0,0.6);
    }}

    .nav-header h1 {{
        font-size: 15px;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    .download-btn {{
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        text-decoration: none;
        padding: 9px 20px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        transition: all 0.2s;
        border: 1px solid rgba(255,255,255,0.2);
    }}

    .download-btn:hover {{
        background: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(37,99,235,0.5);
    }}

    /* 16:9 Slide Canvas */
    .slide-canvas {{
        width: 1200px;
        height: 675px; /* 16:9 exact */
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 25px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        padding: 28px 44px;
    }}

    .slide-header {{
        display: flex;
        align-items: center;
        margin-bottom: 14px;
        gap: 24px;
    }}

    .lab-logo img {{
        height: 44px;
        object-fit: contain;
    }}

    .title-group {{
        flex: 1;
    }}

    .slide-title {{
        font-size: 22px;
        font-weight: 800;
        color: #1e3a8a;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}

    .slide-subtitle {{
        font-size: 12.5px;
        color: #64748b;
        margin-top: 2px;
        font-weight: 500;
    }}

    .slide-footer {{
        position: absolute;
        bottom: 14px;
        left: 44px;
        right: 44px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #e2e8f0;
        padding-top: 8px;
        font-size: 11px;
        color: #94a3b8;
        font-weight: 600;
    }}

    .footer-logo img {{
        height: 30px;
        object-fit: contain;
    }}

    .slide-number {{
        background: #f1f5f9;
        color: #0f172a;
        padding: 2px 12px;
        border-radius: 999px;
        font-weight: 700;
    }}

    /* SLIDE 1 */
    .s1-columns {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-bottom: 14px;
    }}

    .s1-col {{
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 10px;
    }}

    .s1-col h3 {{
        font-size: 13.5px;
        font-weight: 800;
        color: #0f2942;
        margin-bottom: 6px;
    }}

    .s1-col-img {{
        height: 85px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 6px;
    }}

    .s1-col-img img {{
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;
    }}

    .s1-col p {{
        font-size: 10.5px;
        color: #475569;
        line-height: 1.35;
    }}

    .s1-platform-box {{
        background: #ffffff;
        border: 2px solid #16a34a;
        border-radius: 12px;
        padding: 12px 18px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(22, 163, 74, 0.08);
        margin-bottom: 12px;
    }}

    .s1-p-title {{
        font-size: 16px;
        font-weight: 800;
        color: #15803d;
    }}

    .s1-p-sub {{
        font-size: 12px;
        font-weight: 700;
        color: #0f2942;
        margin-top: 2px;
    }}

    .s1-pills {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-top: 8px;
    }}

    .s1-pill {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 6px 10px;
        text-align: left;
    }}

    .s1-pill strong {{
        font-size: 10.5px;
        color: #1d4ed8;
        display: block;
        margin-bottom: 2px;
    }}

    .s1-pill span {{
        font-size: 9px;
        color: #475569;
        line-height: 1.25;
        display: block;
    }}

    .s1-score-banner {{
        background: linear-gradient(135deg, #15803d, #16a34a);
        color: white;
        text-align: center;
        padding: 9px;
        border-radius: 8px;
        font-size: 14.5px;
        font-weight: 800;
        letter-spacing: 0.02em;
    }}

    /* SLIDE 2 */
    .prototype-layout {{
        display: grid;
        grid-template-columns: 3.2fr 1fr;
        gap: 20px;
        flex: 1;
        margin-bottom: 14px;
    }}

    .screenshot-frame {{
        border-radius: 12px;
        overflow: hidden;
        border: 1.5px solid #cbd5e1;
        background: #000;
        display: flex;
    }}

    .screenshot-frame img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
    }}

    .badge-column {{
        display: flex;
        flex-direction: column;
        gap: 12px;
    }}

    .feature-badge {{
        border-radius: 10px;
        padding: 12px;
        color: white;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    .fb-green {{ background: #15803d; }}
    .fb-amber {{ background: #d97706; }}
    .fb-red {{ background: #dc2626; }}

    .fb-title {{ font-size: 14px; font-weight: 800; margin-bottom: 3px; }}
    .fb-desc {{ font-size: 10.5px; opacity: 0.95; line-height: 1.35; }}

    .pipeline-ribbon {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
    }}

    .ribbon-step {{
        color: white;
        text-align: center;
        padding: 8px;
        border-radius: 8px;
        font-size: 10.5px;
        font-weight: 700;
    }}

    /* SLIDE 3 */
    .advantage-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        flex: 1;
    }}

    .adv-card {{
        background: #f8fafc;
        border: 1.5px solid #e2e8f0;
        border-radius: 12px;
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
    }}

    .adv-title {{
        font-size: 13px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 5px;
    }}

    .adv-img-box {{
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #ffffff;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }}

    .adv-img-box img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
    }}

    /* SLIDE 4 */
    .arch-pipeline {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-bottom: 12px;
    }}

    .arch-card {{
        border: 1.5px solid;
        border-radius: 10px;
        padding: 8px 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }}

    .ac-title {{
        font-size: 11px;
        font-weight: 800;
        margin-bottom: 6px;
    }}

    .ac-img {{
        height: 105px;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #ffffff;
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 6px;
        border: 1px solid #e2e8f0;
    }}

    .ac-img img {{
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;
    }}

    .ac-desc {{
        font-size: 9.5px;
        color: #334155;
        line-height: 1.3;
        text-align: left;
        width: 100%;
    }}

    .ac-cyan {{ background: #f0f9ff; border-color: #0284c7; }}
    .ac-cyan .ac-title {{ color: #0284c7; }}

    .ac-blue {{ background: #eff6ff; border-color: #2563eb; }}
    .ac-blue .ac-title {{ color: #1d4ed8; }}

    .ac-purple {{ background: #f5f3ff; border-color: #7c3aed; }}
    .ac-purple .ac-title {{ color: #6d28d9; }}

    .ac-green {{ background: #f0fdf4; border-color: #15803d; }}
    .ac-green .ac-title {{ color: #15803d; }}

    .skip-banner {{
        background: #7c3aed;
        color: white;
        text-align: center;
        padding: 4px;
        border-radius: 6px;
        font-size: 10px;
        font-weight: 700;
        margin-bottom: 10px;
    }}

    .bottom-cards-row {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
    }}

    .b-card {{
        background: #f8fafc;
        border: 1.5px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 12px;
    }}

    .bc-title {{
        font-size: 11px;
        font-weight: 800;
        margin-bottom: 4px;
    }}

    .b-card ul {{
        list-style: none;
        display: flex;
        flex-direction: column;
        gap: 3px;
    }}

    .b-card li {{
        font-size: 9.2px;
        color: #475569;
        line-height: 1.3;
    }}

    .b-card .formula {{
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 9.5px;
        margin-bottom: 4px;
    }}
</style>
</head>
<body>

<div class="nav-header">
    <h1>🧠 Brain-CT-Seg Client Presentation (4 Slides Exact Deck)</h1>
    <a href="Brain_CT_Seg_Client_Presentation.pptx" class="download-btn" download>📥 Download .PPTX Presentation</a>
</div>

<!-- ========================================== -->
<!-- SLIDE 1 -->
<!-- ========================================== -->
<div class="slide-canvas">
    <div class="slide-header">
        <div class="lab-logo"><img src="{logo_osl_b64}" alt="OSL Logo"></div>
        <div class="title-group">
            <h2 class="slide-title">Brain-CT-Seg: AI for Brain Tumor Segmentation & Risk Volumetry</h2>
            <div class="slide-subtitle">Overcoming Machine, Deep Learning, and Clinical Radiologist Barriers in Non-Contrast CT</div>
        </div>
    </div>

    <div class="s1-columns">
        <div class="s1-col">
            <h3>Machine / Acquisition</h3>
            <div class="s1-col-img"><img src="{ct_mach_b64}" alt="CT Scanner"></div>
            <p>• The rotation problem (head tilt & couch skew)<br>• Vendor variability (Siemens, GE, Philips)</p>
        </div>
        <div class="s1-col">
            <h3>Software / Deep Learning</h3>
            <div class="s1-col-img"><img src="{sw_b64}" alt="Software AI"></div>
            <p>• Failure of standard CNNs under rotation<br>• Severe 12–18% Dice drop under domain shift</p>
        </div>
        <div class="s1-col">
            <h3>Medical Doctor / Radiologist</h3>
            <div class="s1-col-img"><img src="{doc_b64}" alt="Doctor Scans"></div>
            <p>• Subjectivity of manual tumor delineation<br>• High inter-observer variance (Dice 65–75%)</p>
        </div>
    </div>

    <div class="s1-platform-box">
        <div class="s1-p-title">CT-SE(2) Consensus Platform</div>
        <div class="s1-p-sub">Equivariant Group Theory Backbone (C8 / N=8 Rotations) + 6-Model Consensus Engine</div>
        <div class="s1-pills">
            <div class="s1-pill">
                <strong>Equivariant R2Conv (C8)</strong>
                <span>Guarantees f(g·x) = g·f(x). Boundary is identical at any head tilt or couch angle.</span>
            </div>
            <div class="s1-pill">
                <strong>6-Model Consensus Engine</strong>
                <span>Combines CT-SE(2), HarmonicNet, nnU-Net with calibrated 0.80 agreement threshold.</span>
            </div>
            <div class="s1-pill">
                <strong>Automated 3D Volumetry</strong>
                <span>Quantifies Tumor Volume (mL), Edema Core, Sphericity, and Hemispheric Midline Shift.</span>
            </div>
        </div>
    </div>

    <div class="s1-score-banner">
        84.2% Dice | 73.1% IoU SOTA Segmentation & Quantitative 3D Volumetry
    </div>

    <div class="slide-footer">
        <div class="footer-logo"><img src="{logo_ntu_b64}" alt="NTU Logo"></div>
        <div class="slide-number">Slide 1 of 4</div>
    </div>
</div>

<!-- ========================================== -->
<!-- SLIDE 2 -->
<!-- ========================================== -->
<div class="slide-canvas">
    <div class="slide-header">
        <div class="lab-logo"><img src="{logo_osl_b64}" alt="OSL Logo"></div>
        <div class="title-group">
            <h2 class="slide-title">Framework Prototype</h2>
            <div class="slide-subtitle">A clinical brain tumor evaluation platform in a browser tab (Live Ngrok Deployment)</div>
        </div>
    </div>

    <div class="prototype-layout">
        <div class="screenshot-frame">
            <img src="{mpr_b64}" alt="MPR Dashboard Screenshot">
        </div>
        <div class="badge-column">
            <div class="feature-badge fb-green">
                <div class="fb-title">Zero-install</div>
                <div class="fb-desc">No GPU required on client. Instant browser access via cloud/edge tunnel.</div>
            </div>
            <div class="feature-badge fb-amber">
                <div class="fb-title">Vendor-agnostic</div>
                <div class="fb-desc">Calibrated for NCCT & CTC across Siemens, GE, and Philips scanners.</div>
            </div>
            <div class="feature-badge fb-red">
                <div class="fb-title">Privacy by design</div>
                <div class="fb-desc">Local edge inference, zero external data leaks, automated PACS & DOCX reports.</div>
            </div>
        </div>
    </div>

    <div class="pipeline-ribbon">
        <div class="ribbon-step" style="background:#0284c7;">1. 2.5D Slice Triplet</div>
        <div class="ribbon-step" style="background:#0284c7;">2. Mod-SE(2) Backbone (N=8)</div>
        <div class="ribbon-step" style="background:#1e3a8a;">3. 6-Model Consensus Engine</div>
        <div class="ribbon-step" style="background:#d97706;">4. High-Clarity MPR & 3D BBox</div>
        <div class="ribbon-step" style="background:#15803d;">5. Automated DOCX / PACS Report</div>
    </div>

    <div class="slide-footer">
        <div class="footer-logo"><img src="{logo_ntu_b64}" alt="NTU Logo"></div>
        <div class="slide-number">Slide 2 of 4</div>
    </div>
</div>

<!-- ========================================== -->
<!-- SLIDE 3 -->
<!-- ========================================== -->
<div class="slide-canvas">
    <div class="slide-header">
        <div class="lab-logo"><img src="{logo_osl_b64}" alt="OSL Logo"></div>
        <div class="title-group">
            <h2 class="slide-title">Advantage of CT-SE(2)</h2>
            <div class="slide-subtitle">Clinical Superiority Against Human Radiologists and Baseline CNNs</div>
        </div>
    </div>

    <div class="advantage-grid">
        <div class="adv-card">
            <div class="adv-title">Objective tumor boundary delineation</div>
            <div class="adv-img-box">
                <img src="{card1_b64}" alt="Objective Delineation">
            </div>
        </div>

        <div class="adv-card">
            <div class="adv-title">Higher diagnostic accuracy</div>
            <div class="adv-img-box">
                <img src="{card2_b64}" alt="Diagnostic Accuracy">
            </div>
        </div>

        <div class="adv-card">
            <div class="adv-title">Stable delineation of borderline & tilted cases</div>
            <div class="adv-img-box">
                <img src="{card3_b64}" alt="Rotation Invariance">
            </div>
        </div>

        <div class="adv-card">
            <div class="adv-title">Consistent, structured reporting</div>
            <div class="adv-img-box">
                <img src="{card4_b64}" alt="Structured Reporting">
            </div>
        </div>
    </div>

    <div class="slide-footer">
        <div class="footer-logo"><img src="{logo_ntu_b64}" alt="NTU Logo"></div>
        <div class="slide-number">Slide 3 of 4</div>
    </div>
</div>

<!-- ========================================== -->
<!-- SLIDE 4 -->
<!-- ========================================== -->
<div class="slide-canvas">
    <div class="slide-header">
        <div class="lab-logo"><img src="{logo_osl_b64}" alt="OSL Logo"></div>
        <div class="title-group">
            <h2 class="slide-title">CT-SE(2) Architecture & Pipeline Flow</h2>
            <div class="slide-subtitle">Group Equivariant Convolutional Network (C8 / N=8 Rotations) with Multi-Model Consensus Engine</div>
        </div>
    </div>

    <div class="skip-banner">
        Equivariant Direct-Sum Skip Connections (⊕ Preserves Fine Sub-Millimeter Boundaries)
    </div>

    <div class="arch-pipeline">
        <div class="arch-card ac-cyan">
            <div class="ac-title">1. 2.5D CT Triplet</div>
            <div class="ac-img"><img src="{st1_b64}" alt="Real Raw CT"></div>
            <div class="ac-desc">• Raw CT [B, 3, 256, 256]<br>• [z-1, z, z+1] Inter-slice<br>• Real Patient NCCT scan</div>
        </div>
        <div class="arch-card ac-blue">
            <div class="ac-title">2. C8 Group R2Conv</div>
            <div class="ac-img"><img src="{st2_b64}" alt="C8 Rosette"></div>
            <div class="ac-desc">• 8 Discrete Rotations (45°)<br>• Pointwise MaxPool (2x2)<br>• Channels: 24 -> 48 -> 96</div>
        </div>
        <div class="arch-card ac-purple">
            <div class="ac-title">3. Group Bottleneck</div>
            <div class="ac-img"><img src="{st3_b64}" alt="Latent Features"></div>
            <div class="ac-desc">• 192 Equivariant channels<br>• Invariant representation<br>• f(g · x) = g · f(x)</div>
        </div>
        <div class="arch-card ac-blue">
            <div class="ac-title">4. Equivariant Decoder</div>
            <div class="ac-img"><img src="{st4_b64}" alt="Decoder Features"></div>
            <div class="ac-desc">• R2Upsampling (2x scale)<br>• Direct-Sum Skip (⊕)<br>• Sub-mm edge refinement</div>
        </div>
        <div class="arch-card ac-green">
            <div class="ac-title">5. Output SOTA Mask</div>
            <div class="ac-img"><img src="{st5_b64}" alt="Real Output Scan"></div>
            <div class="ac-desc">• 84.2% Dice | 73.1% IoU<br>• Real Output from Platform<br>• 3D Volumetric ROI mask</div>
        </div>
    </div>

    <div class="bottom-cards-row">
        <div class="b-card">
            <div class="bc-title" style="color:#0f2942;">Advanced Combined Loss Function</div>
            <div class="formula" style="color:#0f2942;">L_total = 2.0*Dice + 0.5*Focal + 0.5*Edge</div>
            <ul>
                <li>• Class Weighting [1.0, 10.0] tackles extreme foreground imbalance</li>
                <li>• Focal Loss (gamma=2.0) suppresses trivial brain background</li>
                <li>• Edge Loss penalizes fuzzy, indistinct tumor boundaries</li>
                <li>• Cosine Annealing Optimizer with Warm Restarts (LR 3e-5)</li>
            </ul>
        </div>
        <div class="b-card" style="background:#eff6ff; border-color:#93c5fd;">
            <div class="bc-title" style="color:#1d4ed8;">C8 Group Invariance Guarantee</div>
            <div class="formula" style="color:#1d4ed8;">ρ_out(g)[K ⋆ f](x) = [K ⋆ ρ_in(g)f](x)</div>
            <ul>
                <li>• 8 Rotational symmetries spaced by 45° around scanner axis</li>
                <li>• Steerable harmonic basis convolutions in R2Conv filter banks</li>
                <li>• Mathematical guarantee: tumor boundary is segmented identically</li>
                <li>  regardless of patient head tilt or scanner couch skew angle</li>
            </ul>
        </div>
        <div class="b-card" style="background:#f0fdf4; border-color:#86efac;">
            <div class="bc-title" style="color:#15803d;">Multi-Model Consensus & 3D Volumetry</div>
            <ul>
                <li>• 6 Benchmark Architectures Ensemble Voting</li>
                <li>  (CT-SE(2), HarmonicNet C4, nnU-Net, Attention U-Net)</li>
                <li>• CT-SE(2) SOTA: 84.2% Dice | 73.1% IoU on NCCT</li>
                <li>• Pixel-wise uncertainty agreement heatmap at 0.80 threshold</li>
                <li>• Automated 3D Volumetry: Tumor (mL), Edema (mL), Sphericity</li>
            </ul>
        </div>
    </div>

    <div class="slide-footer">
        <div class="footer-logo"><img src="{logo_ntu_b64}" alt="NTU Logo"></div>
        <div class="slide-number">Slide 4 of 4</div>
    </div>
</div>

</body>
</html>
"""

with open("presentation_preview.html", "w") as f:
    f.write(html_content)

print("HTML presentation preview rebuilt perfectly!")
