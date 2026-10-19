import os
import base64

def get_base64_img(img_path):
    if not os.path.exists(img_path): return ""
    with open(img_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

mpr_b64 = get_base64_img("presentation_assets/dashboard_mpr_consensus_1792346200853.png")
card1_b64 = get_base64_img("presentation_assets/slide_cards/card1_full_visual.png")
card2_b64 = get_base64_img("presentation_assets/slide_cards/card2_full_visual.png")
card3_b64 = get_base64_img("presentation_assets/slide_cards/card3_full_visual.png")
card4_b64 = get_base64_img("presentation_assets/slide_cards/card4_full_visual.png")
arch_flow_b64 = get_base64_img("presentation_assets/slide_cards/arch_flow_visual_complete.png")

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
        padding: 34px 44px;
    }}

    .slide-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 20px;
    }}

    .lab-brand {{
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #0f2942;
        text-transform: uppercase;
        border-bottom: 2.5px solid #2563eb;
        padding-bottom: 3px;
    }}

    .title-group {{
        flex: 1;
        margin-left: 36px;
    }}

    .slide-title {{
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}

    .slide-subtitle {{
        font-size: 13.5px;
        color: #64748b;
        margin-top: 3px;
        font-weight: 500;
    }}

    .slide-footer {{
        position: absolute;
        bottom: 18px;
        left: 44px;
        right: 44px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #e2e8f0;
        padding-top: 10px;
        font-size: 11px;
        color: #94a3b8;
        font-weight: 600;
    }}

    .slide-number {{
        background: #f1f5f9;
        color: #0f172a;
        padding: 2px 10px;
        border-radius: 999px;
        font-weight: 700;
    }}

    /* SLIDE 1 SPECIFIC */
    .three-pillars {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-bottom: 20px;
    }}

    .pillar-card {{
        border-radius: 12px;
        padding: 18px 20px;
        border: 1.5px solid;
        display: flex;
        flex-direction: column;
    }}

    .card-machine {{ background: #fef2f2; border-color: #ef4444; }}
    .card-software {{ background: #fefce8; border-color: #f59e0b; }}
    .card-doctor {{ background: #eff6ff; border-color: #3b82f6; }}

    .pillar-card h3 {{
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 12px;
        text-align: center;
    }}

    .card-machine h3 {{ color: #b91c1c; }}
    .card-software h3 {{ color: #d97706; }}
    .card-doctor h3 {{ color: #1d4ed8; }}

    .pillar-card ul {{
        list-style: none;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }}

    .pillar-card li {{
        font-size: 12px;
        color: #334155;
        line-height: 1.4;
        display: flex;
        align-items: flex-start;
        gap: 6px;
    }}

    .pillar-card li::before {{
        content: "•";
        font-weight: bold;
        color: inherit;
    }}

    .platform-solution-box {{
        background: #ffffff;
        border: 2px solid #16a34a;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(22, 163, 74, 0.08);
        margin-bottom: 20px;
    }}

    .solution-title {{
        font-size: 20px;
        font-weight: 800;
        color: #15803d;
        letter-spacing: -0.01em;
    }}

    .solution-subtitle {{
        font-size: 14px;
        font-weight: 700;
        color: #0f2942;
        margin-top: 4px;
    }}

    .solution-desc {{
        font-size: 12px;
        color: #64748b;
        margin-top: 6px;
        max-width: 850px;
        margin-left: auto;
        margin-right: auto;
    }}

    .sota-banner {{
        background: linear-gradient(135deg, #15803d, #16a34a);
        color: white;
        text-align: center;
        padding: 12px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 15px rgba(22, 163, 74, 0.3);
    }}

    /* SLIDE 2 SPECIFIC */
    .prototype-layout {{
        display: grid;
        grid-template-columns: 3.2fr 1fr;
        gap: 20px;
        flex: 1;
        margin-bottom: 16px;
    }}

    .screenshot-frame {{
        border-radius: 12px;
        overflow: hidden;
        border: 1.5px solid #cbd5e1;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
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
        padding: 14px;
        color: white;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}

    .fb-green {{ background: #15803d; }}
    .fb-amber {{ background: #d97706; }}
    .fb-red {{ background: #dc2626; }}

    .fb-title {{ font-size: 15px; font-weight: 800; margin-bottom: 4px; }}
    .fb-desc {{ font-size: 11px; opacity: 0.95; line-height: 1.35; }}

    .pipeline-ribbon {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
    }}

    .ribbon-step {{
        color: white;
        text-align: center;
        padding: 10px 8px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 700;
    }}

    /* SLIDE 3 SPECIFIC */
    .advantage-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 18px;
        flex: 1;
    }}

    .adv-card {{
        background: #f8fafc;
        border: 1.5px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
    }}

    .adv-title {{
        font-size: 14px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 8px;
    }}

    .adv-img-box {{
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #ffffff;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }}

    .adv-img-box img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
    }}

    /* SLIDE 4 SPECIFIC */
    .arch-container {{
        flex: 1;
        display: flex;
        flex-direction: column;
        background: #ffffff;
        border-radius: 14px;
        overflow: hidden;
        border: 1.5px solid #e2e8f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }}

    .arch-container img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
    }}
</style>
</head>
<body>

<div class="nav-header">
    <h1>🧠 Brain-CT-Seg Client Presentation (4 Slides Exact Deck)</h1>
    <a href="Brain_CT_Seg_Client_Presentation.pptx" class="download-btn" download>📥 Download .PPTX Presentation</a>
</div>

<!-- ========================================== -->
<!-- SLIDE 1 (MATCHING REF 4 / SLIDE 5) -->
<!-- ========================================== -->
<div class="slide-canvas">
    <div class="slide-header">
        <div class="lab-brand">Optical Systems Lab</div>
        <div class="title-group">
            <h2 class="slide-title">Brain-CT-Seg: AI for Mayo-Grade Brain Tumor Segmentation</h2>
            <div class="slide-subtitle">Overcoming Machine, Deep Learning, and Clinical Radiologist Barriers in Non-Contrast CT</div>
        </div>
    </div>

    <div class="three-pillars">
        <div class="pillar-card card-machine">
            <h3>Machine / Acquisition</h3>
            <ul>
                <li>The rotation & head tilt problem</li>
                <li>Sparse 5mm slice thickness & couch gaps</li>
                <li>Cross-vendor variability (GE, Siemens, Philips)</li>
            </ul>
        </div>
        <div class="pillar-card card-software">
            <h3>Software / Deep Learning</h3>
            <ul>
                <li>Failure of standard CNNs on rotated scans</li>
                <li>Severe 12–18% Dice drop under domain shift</li>
                <li>Overconfident false positive tumor predictions</li>
            </ul>
        </div>
        <div class="pillar-card card-doctor">
            <h3>Medical Doctor / Radiologist</h3>
            <ul>
                <li>Subjectivity of human boundary delineation</li>
                <li>High inter-observer variance (Dice 65–75%)</li>
                <li>Subtle tumor contrast on Non-Contrast CT</li>
            </ul>
        </div>
    </div>

    <div class="platform-solution-box">
        <div class="solution-title">Mod-Seg-SE(2) Consensus Platform</div>
        <div class="solution-subtitle">Equivariant Group Theory Backbone (N=8 discrete rotations) + 6-Model Consensus Heatmap Engine</div>
        <div class="solution-desc">Solves rotational invariance at the mathematical group level • Eliminates radiologist inter-observer variability • Quantifies 3D multi-slice tumor volume</div>
    </div>

    <div class="sota-banner">
        84.2% Dice | 73.1% IoU SOTA Segmentation & Quantitative 3D Volumetry
    </div>

    <div class="slide-footer">
        <div>National Taiwan University • NTU GSAT | Brain-CTC-Seg Consortium</div>
        <div class="slide-number">Slide 1 of 4</div>
    </div>
</div>

<!-- ========================================== -->
<!-- SLIDE 2 (MATCHING REF 1 / SLIDE 6) -->
<!-- ========================================== -->
<div class="slide-canvas">
    <div class="slide-header">
        <div class="lab-brand">Optical Systems Lab</div>
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
        <div>National Taiwan University • NTU GSAT | Brain-CTC-Seg Consortium</div>
        <div class="slide-number">Slide 2 of 4</div>
    </div>
</div>

<!-- ========================================== -->
<!-- SLIDE 3 (MATCHING REF 2 / SLIDE 8) -->
<!-- ========================================== -->
<div class="slide-canvas">
    <div class="slide-header">
        <div class="lab-brand">Optical Systems Lab</div>
        <div class="title-group">
            <h2 class="slide-title">Advantage of Mod-Seg-SE(2)</h2>
            <div class="slide-subtitle">Clinical Superiority Against Human Radiologists and Baseline CNNs</div>
        </div>
    </div>

    <div class="advantage-grid">
        <!-- Card 1 -->
        <div class="adv-card">
            <div class="adv-title">Objective Tumor Boundary Delineation</div>
            <div class="adv-img-box">
                <img src="{card1_b64}" alt="Card 1 Objective Segmentation">
            </div>
        </div>

        <!-- Card 2 -->
        <div class="adv-card">
            <div class="adv-title">Higher Segmentation & Diagnostic Accuracy</div>
            <div class="adv-img-box">
                <img src="{card2_b64}" alt="Card 2 Higher Accuracy Bars">
            </div>
        </div>

        <!-- Card 3 -->
        <div class="adv-card">
            <div class="adv-title">Stable Delineation Under Head Tilt & Rotation</div>
            <div class="adv-img-box">
                <img src="{card3_b64}" alt="Card 3 Rotation Invariance">
            </div>
        </div>

        <!-- Card 4 -->
        <div class="adv-card">
            <div class="adv-title">Consistent, Structured Volumetric Reporting</div>
            <div class="adv-img-box">
                <img src="{card4_b64}" alt="Card 4 Structured Reporting">
            </div>
        </div>
    </div>

    <div class="slide-footer">
        <div>National Taiwan University • NTU GSAT | Brain-CTC-Seg Consortium</div>
        <div class="slide-number">Slide 3 of 4</div>
    </div>
</div>

<!-- ========================================== -->
<!-- SLIDE 4 (REPLACES MARKET SLIDE -> ARCHITECTURE) -->
<!-- ========================================== -->
<div class="slide-canvas">
    <div class="slide-header">
        <div class="lab-brand">Optical Systems Lab</div>
        <div class="title-group">
            <h2 class="slide-title">Mod-Seg-SE(2) Architecture & Pipeline Flow</h2>
            <div class="slide-subtitle">Group Equivariant Convolutional Network (C8 / N=8 Rotations) with Multi-Model Consensus Engine</div>
        </div>
    </div>

    <div class="arch-container">
        <img src="{arch_flow_b64}" alt="Architecture Flow and Visual Blocks">
    </div>

    <div class="slide-footer">
        <div>National Taiwan University • NTU GSAT | Brain-CTC-Seg Consortium</div>
        <div class="slide-number">Slide 4 of 4</div>
    </div>
</div>

</body>
</html>
"""

with open("presentation_preview.html", "w") as f:
    f.write(html_content)

print("Generated presentation_preview.html with exactly 4 slides and high-definition visual cards!")
