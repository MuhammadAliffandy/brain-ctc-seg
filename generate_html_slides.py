import os
import base64

def get_base64_img(img_path):
    if not os.path.exists(img_path): return ""
    with open(img_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

mpr_b64 = get_base64_img("presentation_assets/dashboard_mpr_consensus_1792346200853.png")
arch_b64 = get_base64_img("presentation_assets/Fig5_Mod_Seg_SE2_Architecture.png")
grid_b64 = get_base64_img("presentation_assets/Fig2_Segmentation_Grid.png")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brain-CT-Seg Client Presentation (5 Slides)</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }}
    
    body {{
        background: #0f172a;
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
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 999px;
        padding: 10px 24px;
        display: flex;
        align-items: center;
        gap: 16px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
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
        padding: 8px 18px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        transition: all 0.2s;
        border: 1px solid rgba(255,255,255,0.2);
    }}

    .download-btn:hover {{
        background: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37,99,235,0.4);
    }}

    /* 16:9 Slide Canvas */
    .slide-canvas {{
        width: 1200px;
        height: 675px; /* 16:9 exact */
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 25px 60px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        padding: 36px 48px;
    }}

    .slide-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 24px;
    }}

    .lab-brand {{
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #0f2942;
        text-transform: uppercase;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 3px;
    }}

    .title-group {{
        flex: 1;
        margin-left: 36px;
    }}

    .slide-title {{
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}

    .slide-subtitle {{
        font-size: 14px;
        color: #64748b;
        margin-top: 4px;
        font-weight: 500;
    }}

    .slide-footer {{
        position: absolute;
        bottom: 20px;
        left: 48px;
        right: 48px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #e2e8f0;
        padding-top: 12px;
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
        padding: 16px 20px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }}

    .pillar-red {{ background: #fef2f2; border: 1.5px solid #ef4444; }}
    .pillar-amber {{ background: #fffbeb; border: 1.5px solid #f59e0b; }}
    .pillar-blue {{ background: #eff6ff; border: 1.5px solid #3b82f6; }}

    .pillar-title {{
        font-size: 15px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .pillar-red .pillar-title {{ color: #b91c1c; }}
    .pillar-amber .pillar-title {{ color: #b45309; }}
    .pillar-blue .pillar-title {{ color: #1d4ed8; }}

    .pillar-list {{
        list-style: none;
        display: flex;
        flex-direction: column;
        gap: 6px;
        font-size: 12.5px;
        color: #334155;
    }}

    .pillar-list li:before {{
        content: "•";
        color: currentColor;
        font-weight: bold;
        display: inline-block;
        width: 1em;
        margin-left: -1em;
    }}

    .solution-box {{
        background: #ffffff;
        border: 2px solid #16a34a;
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(22, 163, 74, 0.08);
        margin-bottom: 16px;
    }}

    .solution-title {{
        font-size: 20px;
        font-weight: 800;
        color: #15803d;
        letter-spacing: -0.01em;
    }}

    .solution-sub {{
        font-size: 13.5px;
        font-weight: 700;
        color: #0f2942;
        margin: 4px 0 6px;
    }}

    .solution-desc {{
        font-size: 12px;
        color: #64748b;
    }}

    .target-banner {{
        background: #15803d;
        color: white;
        text-align: center;
        padding: 12px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 0.02em;
    }}

    /* SLIDE 2 SPECIFIC */
    .prototype-grid {{
        display: grid;
        grid-template-columns: 2.8fr 1fr;
        gap: 24px;
        flex: 1;
    }}

    .screen-container {{
        background: #0f172a;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #cbd5e1;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .screen-container img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}

    .badge-column {{
        display: flex;
        flex-direction: column;
        gap: 16px;
    }}

    .benefit-badge {{
        border-radius: 12px;
        padding: 18px;
        color: white;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }}

    .b-green {{ background: linear-gradient(135deg, #15803d, #166534); }}
    .b-amber {{ background: linear-gradient(135deg, #d97706, #b45309); }}
    .b-red {{ background: linear-gradient(135deg, #b91c1c, #991b1b); }}

    .b-title {{ font-size: 16px; font-weight: 800; }}
    .b-desc {{ font-size: 12px; opacity: 0.95; line-height: 1.4; }}

    .pipeline-bar {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-top: 16px;
    }}

    .pipeline-step {{
        background: #0284c7;
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
        padding: 16px 20px;
        display: flex;
        flex-direction: column;
    }}

    .adv-title {{
        font-size: 15px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 12px;
    }}

    .vs-box {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        flex: 1;
    }}

    .box-without {{
        background: #fef2f2;
        border: 1px solid #f87171;
        border-radius: 10px;
        padding: 12px;
        font-size: 12px;
    }}

    .box-with {{
        background: #f0fdf4;
        border: 1.5px solid #22c55e;
        border-radius: 10px;
        padding: 12px;
        font-size: 12px;
    }}

    .tag-without {{ font-weight: 800; color: #dc2626; margin-bottom: 6px; font-size: 11px; }}
    .tag-with {{ font-weight: 800; color: #16a34a; margin-bottom: 6px; font-size: 11px; }}

    /* SLIDE 4 SPECIFIC */
    .market-layout {{
        display: grid;
        grid-template-columns: 2.2fr 1fr;
        gap: 24px;
        flex: 1;
    }}

    .matrix-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }}

    .matrix-table th {{
        background: #0f2942;
        color: white;
        padding: 10px 12px;
        font-weight: 700;
        text-align: center;
    }}

    .matrix-table th:first-child {{ text-align: left; }}
    .matrix-table th:last-child {{ background: #15803d; }}

    .matrix-table td {{
        padding: 10px 12px;
        border-bottom: 1px solid #e2e8f0;
        color: #334155;
        text-align: center;
    }}

    .matrix-table td:first-child {{
        text-align: left;
        font-weight: 700;
        color: #1e293b;
    }}

    .matrix-table tr:nth-child(even) td {{ background: #f8fafc; }}
    .matrix-table td:last-child {{
        background: #f0fdf4 !important;
        font-weight: 800;
        color: #15803d;
    }}

    .market-callouts {{
        display: flex;
        flex-direction: column;
        gap: 16px;
    }}

    .m-card {{
        border-radius: 12px;
        padding: 18px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }}

    .mc-red {{ background: #fef2f2; border: 1.5px solid #ef4444; }}
    .mc-blue {{ background: #eff6ff; border: 1.5px solid #3b82f6; }}

    .mc-title {{ font-size: 16px; font-weight: 800; }}
    .mc-red .mc-title {{ color: #b91c1c; }}
    .mc-blue .mc-title {{ color: #1d4ed8; }}
    .mc-desc {{ font-size: 12px; color: #334155; line-height: 1.45; }}

    /* SLIDE 5 SPECIFIC */
    .arch-layout {{
        display: grid;
        grid-template-columns: 1.1fr 1fr;
        gap: 24px;
        flex: 1;
        align-items: center;
    }}

    .arch-visual {{
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 12px;
        padding: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    }}

    .arch-visual img {{
        max-width: 100%;
        max-height: 440px;
        object-fit: contain;
    }}

    .arch-specs {{
        display: flex;
        flex-direction: column;
        gap: 12px;
    }}

    .spec-item {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 10px 14px;
    }}

    .spec-item-title {{
        font-size: 13px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 3px;
    }}

    .spec-item-desc {{
        font-size: 11.5px;
        color: #475569;
        line-height: 1.4;
    }}
</style>
</head>
<body>

<div class="nav-header">
    <h1>🧠 Brain-CT-Seg Client Presentation (5 Slides)</h1>
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
        <div class="pillar-card pillar-red">
            <div class="pillar-title">🏥 Machine / Acquisition</div>
            <ul class="pillar-list">
                <li>The rotation & head tilt problem</li>
                <li>Sparse 5mm slice thickness & couch gaps</li>
                <li>Cross-vendor variability (GE, Siemens, Philips)</li>
            </ul>
        </div>
        <div class="pillar-card pillar-amber">
            <div class="pillar-title">🧠 Software / Deep Learning</div>
            <ul class="pillar-list">
                <li>Conventional CNNs fail on rotated scans</li>
                <li>Severe 12–18% Dice drop under domain shift</li>
                <li>Overconfident false positive tumor predictions</li>
            </ul>
        </div>
        <div class="pillar-card pillar-blue">
            <div class="pillar-title">👨‍⚕️ Medical Doctor</div>
            <ul class="pillar-list">
                <li>Subjectivity of human boundary delineation</li>
                <li>High inter-observer variance (Dice 65–75%)</li>
                <li>Subtle tumor contrast on Non-Contrast CT</li>
            </ul>
        </div>
    </div>

    <div class="solution-box">
        <div class="solution-title">Mod-Seg-SE(2) Consensus Platform</div>
        <div class="solution-sub">Equivariant Group Theory Backbone (N=8 discrete rotations) + 6-Model Consensus Heatmap Engine</div>
        <div class="solution-desc">Solves rotational invariance at the mathematical group level • Eliminates radiologist inter-observer variability • Quantifies 3D multi-slice tumor volume</div>
    </div>

    <div class="target-banner">
        🎯 84.2% Dice | 73.1% IoU SOTA Segmentation & Quantitative 3D Volumetry
    </div>

    <div class="slide-footer">
        <div>National Taiwan University • NTU GSAT | Brain-CTC-Seg Consortium</div>
        <div class="slide-number">5</div>
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
            <div class="slide-subtitle">A four-plane clinical brain tumor evaluation platform in a browser tab</div>
        </div>
        <div style="background: #4f46e5; color: white; padding: 6px 16px; border-radius: 999px; font-size: 13px; font-weight: 800;">Live Demo ⚡</div>
    </div>

    <div class="prototype-grid">
        <div class="screen-container">
            <img src="{mpr_b64}" alt="Web Dashboard MPR and Consensus">
        </div>
        <div class="badge-column">
            <div class="benefit-badge b-green">
                <div class="b-title">Zero-install</div>
                <div class="b-desc">No GPU required on client workstation; instant web browser access via Ngrok tunnel.</div>
            </div>
            <div class="benefit-badge b-amber">
                <div class="b-title">Vendor-agnostic</div>
                <div class="b-desc">Calibrated for NCCT & CTC scans across Siemens, GE, Philips hardware.</div>
            </div>
            <div class="benefit-badge b-red">
                <div class="b-title">Privacy by design</div>
                <div class="b-desc">Local edge server inference; exportable PACS DICOM and downloadable Word reports.</div>
            </div>
        </div>
    </div>

    <div class="pipeline-bar">
        <div class="pipeline-step" style="background:#0284c7;">2.5D Slice Triplet</div>
        <div class="pipeline-step" style="background:#0369a1;">+ Mod-SE(2) Backbone (N=8)</div>
        <div class="pipeline-step" style="background:#1e3a8a;">6-Model Consensus Engine</div>
        <div class="pipeline-step" style="background:#d97706;">High-Clarity MPR & 3D BBox</div>
        <div class="pipeline-step" style="background:#15803d;">Automated DOCX / PACS Report</div>
    </div>

    <div class="slide-footer">
        <div>National Taiwan University • NTU GSAT | Brain-CTC-Seg Consortium</div>
        <div class="slide-number">6</div>
    </div>
</div>

<!-- ========================================== -->
<!-- SLIDE 3 (MATCHING REF 2 / SLIDE 7) -->
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
            <div class="vs-box">
                <div class="box-without">
                    <div class="tag-without">✖ Without</div>
                    <div>Same CT scan, 3 readers delineate different margins (inter-observer Dice 65–75%).</div>
                </div>
                <div class="box-with">
                    <div class="tag-with">✔ With Mod-Seg-SE(2)</div>
                    <div>One reproducible consensus outline + 6-model agreement heatmap (Dice 84.2%).</div>
                </div>
            </div>
        </div>

        <!-- Card 2 -->
        <div class="adv-card">
            <div class="adv-title">Higher Segmentation & Diagnostic Accuracy</div>
            <div class="vs-box">
                <div class="box-without">
                    <div class="tag-without">✖ Without</div>
                    <div>Standard U-Net: 74.2% Dice<br>Attention U-Net: 76.5% Dice<br>High false positive rate.</div>
                </div>
                <div class="box-with">
                    <div class="tag-with">✔ With Mod-Seg-SE(2)</div>
                    <div><b>84.2% Dice | 73.1% IoU</b><br>SOTA performance on challenging non-contrast CT.</div>
                </div>
            </div>
        </div>

        <!-- Card 3 -->
        <div class="adv-card">
            <div class="adv-title">Stable Delineation Under Head Tilt & Rotation</div>
            <div class="vs-box">
                <div class="box-without">
                    <div class="tag-without">✖ Without</div>
                    <div>Tilting head or gantry by 45° drops conventional CNN Dice by 12–18%.</div>
                </div>
                <div class="box-with">
                    <div class="tag-with">✔ With Mod-Seg-SE(2)</div>
                    <div>Rotation-Equivariant (N=8 group convs): identical boundary at any angle.</div>
                </div>
            </div>
        </div>

        <!-- Card 4 -->
        <div class="adv-card">
            <div class="adv-title">Consistent, Structured Volumetric Reporting</div>
            <div class="vs-box">
                <div class="box-without">
                    <div class="tag-without">✖ Without</div>
                    <div>Subjective free-text notes ("moderate mass effect in parietal lobe"), hard to compare.</div>
                </div>
                <div class="box-with">
                    <div class="tag-with">✔ With Mod-Seg-SE(2)</div>
                    <div>Automated 3D volume (mL), min/max lesion diameter, risk tier & 1-click Word report.</div>
                </div>
            </div>
        </div>
    </div>

    <div class="slide-footer">
        <div>National Taiwan University • NTU GSAT | Brain-CTC-Seg Consortium</div>
        <div class="slide-number">7</div>
    </div>
</div>

<!-- ========================================== -->
<!-- SLIDE 4 (MATCHING REF 3 / SLIDE 8) -->
<!-- ========================================== -->
<div class="slide-canvas">
    <div class="slide-header">
        <div class="lab-brand">Optical Systems Lab</div>
        <div class="title-group">
            <h2 class="slide-title">Current Market Opportunity & Clinical Positioning</h2>
            <div class="slide-subtitle">AI in Neuroimaging market projection: US$14.2 B by 2032 (Growth 28.5%/year)</div>
        </div>
    </div>

    <div class="market-layout">
        <table class="matrix-table">
            <thead>
                <tr>
                    <th>Capability / Feature</th>
                    <th>Stroke CAD<br>(RapidAI / Viz)</th>
                    <th>General AI<br>(Aidoc)</th>
                    <th>Academic SOTA<br>(nnU-Net)</th>
                    <th>Mod-Seg-SE(2)<br>[OURS]</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Full 3D Multi-Planar Volumetry (Ax, Cor, Sag)</td>
                    <td>⚪ No</td><td>⚪ No</td><td>🟡 Partial</td><td>🟢 YES</td>
                </tr>
                <tr>
                    <td>Non-Contrast CT (NCCT) Tumor Detection</td>
                    <td>⚪ No</td><td>🟡 Hemorrhage</td><td>🟡 CT Only</td><td>🟢 YES</td>
                </tr>
                <tr>
                    <td>Group Equivariant Invariance (SE(2) N=8)</td>
                    <td>⚪ No</td><td>⚪ No</td><td>⚪ No</td><td>🟢 YES</td>
                </tr>
                <tr>
                    <td>Multi-Model Consensus & Uncertainty Map</td>
                    <td>⚪ No</td><td>⚪ No</td><td>⚪ No</td><td>🟢 YES</td>
                </tr>
                <tr>
                    <td>Zero-Install Browser Web & PACS Integration</td>
                    <td>🟡 Scope-locked</td><td>🟡 Cloud-locked</td><td>⚪ Script only</td><td>🟢 YES</td>
                </tr>
                <tr>
                    <td>Automated Downloadable Clinical Word/DICOM</td>
                    <td>⚪ No</td><td>🟡 PDF only</td><td>⚪ No</td><td>🟢 YES</td>
                </tr>
            </tbody>
        </table>

        <div class="market-callouts">
            <div class="m-card mc-red">
                <div class="mc-title">The Open Space</div>
                <div class="mc-desc">No approved clinical AI software performs automated brain tumor segmentation on raw Non-Contrast CT with rotation-equivariant invariance and zero-install browser deployment.<br><br>Traditional tools fail when heads are tilted; our platform fills this clinical void.</div>
            </div>
            <div class="m-card mc-blue">
                <div class="mc-title">Two Buyers, One Engine</div>
                <div class="mc-desc"><b>1. Hospital Routine Care:</b> Rapid neuro-oncology triage, volumetric mass effect quantification, and surgical planning.<br><br><b>2. Clinical Drug Trials:</b> Automated, reproducible central reading without radiologist fatigue or rater bias.</div>
            </div>
        </div>
    </div>

    <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px; padding: 6px 12px; font-size: 11px; color: #b45309; text-align: center; margin-top: 12px; font-weight: 600;">
        Present status: Working Web Platform Prototype (Verified on DGX-H100 & Taiwan Patient Cohort). Validated on 6 Baseline Architectures.
    </div>

    <div class="slide-footer">
        <div>National Taiwan University • NTU GSAT | Brain-CTC-Seg Consortium</div>
        <div class="slide-number">8</div>
    </div>
</div>

<!-- ========================================== -->
<!-- SLIDE 5: ARCHITECTURE SLIDE (EXTRA REQUEST) -->
<!-- ========================================== -->
<div class="slide-canvas">
    <div class="slide-header">
        <div class="lab-brand">Optical Systems Lab</div>
        <div class="title-group">
            <h2 class="slide-title">Mod-Seg-SE(2) Architecture & Training Pipeline</h2>
            <div class="slide-subtitle">Group Equivariant Convolutional Network (C8 / N=8 Rotations) for Robust Brain CT Segmentation</div>
        </div>
    </div>

    <div class="arch-layout">
        <div class="arch-visual">
            <img src="{arch_b64}" alt="Mod-Seg-SE(2) Architecture">
        </div>

        <div class="arch-specs">
            <div class="spec-item">
                <div class="spec-item-title">1. 2.5D Spatial Stacking Input</div>
                <div class="spec-item-desc">• Input tensor: [B, 3, 256, 256] stacking [img_prev, img_curr, img_next]<br>• Provides 3D volumetric context without massive 3D-CNN compute overhead.</div>
            </div>
            <div class="spec-item">
                <div class="spec-item-title">2. Group Equivariant R2Conv Encoder</div>
                <div class="spec-item-desc">• Discrete rotation group C8 (N=8 rotations spaced by 45°)<br>• Steerable filter banks guarantee f(g • x) = g • f(x)<br>• Invariant to scanner couch indexing & patient head tilt.</div>
            </div>
            <div class="spec-item">
                <div class="spec-item-title">3. Equivariant Bottleneck & Decoder</div>
                <div class="spec-item-desc">• Pointwise Max Pooling & Equivariant Group Normalization<br>• R2Upsampling + skip connection direct sum concatenation<br>• Preserves fine sub-millimeter tumor boundary details.</div>
            </div>
            <div class="spec-item">
                <div class="spec-item-title">4. Projection Head & Advanced Loss</div>
                <div class="spec-item-desc">• Equivariant-to-Trivial projection (1x1 conv) -> 2-class softmax<br>• AdvancedCombinedLoss: L = 2.0*Dice + 0.5*Focal + 0.5*Edge<br>• Class weighting [1.0, 10.0] tackles extreme foreground/background tumor imbalance.</div>
            </div>
            <div class="spec-item" style="border-left-color: #16a34a; background: #f0fdf4;">
                <div class="spec-item-title" style="color: #15803d;">5. Validated Performance (SOTA)</div>
                <div class="spec-item-desc">• Mod-Seg-SE(2) achieves <b>84.2% Dice & 73.1% IoU</b> on challenging NCCT<br>• Retains >82% Dice under severe 45°/90°/180° scan rotations.</div>
            </div>
        </div>
    </div>

    <div class="slide-footer">
        <div>National Taiwan University • NTU GSAT | Brain-CTC-Seg Consortium</div>
        <div class="slide-number">9</div>
    </div>
</div>

</body>
</html>
"""

with open("presentation_preview.html", "w") as f:
    f.write(html_content)

print("HTML slide preview generated at: presentation_preview.html")
