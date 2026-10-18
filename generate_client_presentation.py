import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333) # 16:9 Widescreen
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    # Palette constants
    NAVY = RGBColor(15, 41, 66)
    BLUE_DARK = RGBColor(30, 58, 138)
    TEXT_MAIN = RGBColor(30, 41, 59)
    TEXT_MUTED = RGBColor(100, 116, 139)
    WHITE = RGBColor(255, 255, 255)
    GREEN_DARK = RGBColor(21, 128, 61)
    GREEN_LIGHT = RGBColor(220, 252, 231)
    RED_DARK = RGBColor(220, 38, 38)
    RED_LIGHT = RGBColor(254, 226, 226)
    AMBER_DARK = RGBColor(217, 119, 6)
    AMBER_LIGHT = RGBColor(254, 243, 199)
    BG_CARD = RGBColor(248, 250, 252)
    BORDER_CARD = RGBColor(226, 232, 240)
    CYAN_ACCENT = RGBColor(2, 132, 199)

    def add_header(slide, title_text, subtitle_text=None, slide_num=5):
        # Lab Badge
        tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(2.5), Inches(0.6))
        tf = tx_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = "OPTICAL SYSTEMS LAB"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = NAVY
        
        # Slide Title
        t_box = slide.shapes.add_textbox(Inches(3.3), Inches(0.35), Inches(9.2), Inches(0.8))
        tf2 = t_box.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = title_text
        p2.font.size = Pt(22)
        p2.font.bold = True
        p2.font.color.rgb = BLUE_DARK
        
        if subtitle_text:
            p_sub = tf2.add_paragraph()
            p_sub.text = subtitle_text
            p_sub.font.size = Pt(13)
            p_sub.font.color.rgb = TEXT_MUTED

        # Footer
        f_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(5.0), Inches(0.4))
        p_foot = f_box.text_frame.paragraphs[0]
        p_foot.text = "National Taiwan University • NTU GSAT | Brain-CTC-Seg Consortium"
        p_foot.font.size = Pt(10)
        p_foot.font.color.rgb = TEXT_MUTED
        
        # Slide Number
        num_box = slide.shapes.add_textbox(Inches(12.2), Inches(7.0), Inches(0.8), Inches(0.4))
        p_num = num_box.text_frame.paragraphs[0]
        p_num.text = str(slide_num)
        p_num.font.size = Pt(11)
        p_num.font.bold = True
        p_num.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # SLIDE 1: Problem Formulation (matching Reference Screenshot 4 - slide 5)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    add_header(s1, "Brain-CT-Seg: AI for Brain Tumor Segmentation & Risk Volumetry", 
               "Overcoming Machine, Deep Learning, and Clinical Radiologist Barriers in Non-Contrast CT", slide_num=5)

    # 3 Top Columns: Machine, Software, Medical Doctor
    cols_data = [
        {
            "header": "Machine / Acquisition",
            "icon": "🏥",
            "items": ["The rotation & head tilt problem", "Sparse 5mm slice thickness & couch gaps", "Cross-vendor variability (GE, Siemens, Philips)"],
            "color": RED_DARK, "bg": RED_LIGHT
        },
        {
            "header": "Software / AI",
            "icon": "🧠",
            "items": ["Failure of standard CNNs on rotated scans", "Severe 12–18% Dice drop under domain shift", "Overconfident false positive tumor predictions"],
            "color": AMBER_DARK, "bg": AMBER_LIGHT
        },
        {
            "header": "Medical Doctor",
            "icon": "👨‍⚕️",
            "items": ["Subjectivity of human boundary delineation", "High inter-observer variance (Dice 65–75%)", "Subtle tumor contrast on Non-Contrast CT"],
            "color": BLUE_DARK, "bg": RGBColor(238, 242, 255)
        }
    ]

    for i, c in enumerate(cols_data):
        left = Inches(0.8 + i * 4.0)
        top = Inches(1.5)
        card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(3.7), Inches(2.2))
        card.fill.solid()
        card.fill.fore_color.rgb = c["bg"]
        card.line.color.rgb = c["color"]
        card.line.width = Pt(1.5)
        
        tf = card.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"{c['icon']} {c['header']}"
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = c["color"]
        p.alignment = PP_ALIGN.CENTER
        
        for item in c["items"]:
            pi = tf.add_paragraph()
            pi.text = f"• {item}"
            pi.font.size = Pt(12)
            pi.font.color.rgb = TEXT_MAIN
            pi.space_before = Pt(4)

    # Big Solution Box (Mod-Seg-SE2 Platform)
    sol_box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.0), Inches(11.7), Inches(1.8))
    sol_box.fill.solid()
    sol_box.fill.fore_color.rgb = WHITE
    sol_box.line.color.rgb = GREEN_DARK
    sol_box.line.width = Pt(2.0)
    
    tf_sol = sol_box.text_frame
    tf_sol.word_wrap = True
    p_sol = tf_sol.paragraphs[0]
    p_sol.text = "Mod-Seg-SE(2) Consensus Platform"
    p_sol.font.size = Pt(18)
    p_sol.font.bold = True
    p_sol.font.color.rgb = GREEN_DARK
    p_sol.alignment = PP_ALIGN.CENTER
    
    p_sol2 = tf_sol.add_paragraph()
    p_sol2.text = "Equivariant Group Theory Backbone (N=8 discrete rotations) + 6-Model Consensus Heatmap Engine"
    p_sol2.font.size = Pt(13)
    p_sol2.font.bold = True
    p_sol2.font.color.rgb = NAVY
    p_sol2.alignment = PP_ALIGN.CENTER
    p_sol2.space_before = Pt(4)
    
    p_sol3 = tf_sol.add_paragraph()
    p_sol3.text = "Solves rotational invariance at the mathematical group level • Eliminates radiologist inter-observer variability • Quantifies 3D multi-slice tumor volume"
    p_sol3.font.size = Pt(11)
    p_sol3.font.color.rgb = TEXT_MUTED
    p_sol3.alignment = PP_ALIGN.CENTER
    p_sol3.space_before = Pt(4)

    # Bottom Target Banner
    res_box = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(2.5), Inches(6.0), Inches(8.3), Inches(0.7))
    res_box.fill.solid()
    res_box.fill.fore_color.rgb = GREEN_DARK
    res_box.line.fill.background()
    
    tf_res = res_box.text_frame
    p_res = tf_res.paragraphs[0]
    p_res.text = "84.2% Dice | 73.1% IoU SOTA Segmentation & Quantitative 3D Volumetry"
    p_res.font.size = Pt(16)
    p_res.font.bold = True
    p_res.font.color.rgb = WHITE
    p_res.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 2: Framework Prototype (matching Reference Screenshot 1 - slide 6)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Framework Prototype", "A clinical brain tumor evaluation platform in a browser tab", slide_num=6)

    # "Live Demo" badge
    demo_badge = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.8), Inches(0.4), Inches(1.8), Inches(0.45))
    demo_badge.fill.solid()
    demo_badge.fill.fore_color.rgb = RGBColor(99, 102, 241)
    demo_badge.line.fill.background()
    p_db = demo_badge.text_frame.paragraphs[0]
    p_db.text = "Live Demo ⚡"
    p_db.font.size = Pt(13)
    p_db.font.bold = True
    p_db.font.color.rgb = WHITE
    p_db.alignment = PP_ALIGN.CENTER

    # Embedded Screenshot of the Dashboard
    mpr_img_path = "presentation_assets/dashboard_mpr_consensus_1792346200853.png"
    if os.path.exists(mpr_img_path):
        s2.shapes.add_picture(mpr_img_path, Inches(0.8), Inches(1.3), Inches(8.8), Inches(4.5))

    # Right Column Badges (Zero-install, Vendor-agnostic, Privacy by design)
    badges = [
        {
            "title": "Zero-install",
            "desc": "No GPU required on client,\nInstant browser access via Ngrok",
            "bg": GREEN_DARK, "fg": WHITE
        },
        {
            "title": "Vendor-agnostic",
            "desc": "Calibrated for NCCT & CTC\nacross Siemens, GE, Philips",
            "bg": AMBER_DARK, "fg": WHITE
        },
        {
            "title": "Privacy by design",
            "desc": "Local edge server inference,\nAutomated PACS & DOCX reports",
            "bg": RED_DARK, "fg": WHITE
        }
    ]

    for i, b in enumerate(badges):
        top_pos = Inches(1.3 + i * 1.55)
        b_card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.8), top_pos, Inches(2.7), Inches(1.35))
        b_card.fill.solid()
        b_card.fill.fore_color.rgb = b["bg"]
        b_card.line.fill.background()
        
        tf_b = b_card.text_frame
        tf_b.word_wrap = True
        p_bt = tf_b.paragraphs[0]
        p_bt.text = b["title"]
        p_bt.font.size = Pt(15)
        p_bt.font.bold = True
        p_bt.font.color.rgb = b["fg"]
        
        p_bd = tf_b.add_paragraph()
        p_bd.text = b["desc"]
        p_bd.font.size = Pt(11)
        p_bd.font.color.rgb = b["fg"]
        p_bd.space_before = Pt(4)

    # Bottom Pipeline Ribbon
    steps = [
        "2.5D Slice Triplet",
        "+ Mod-SE(2) Backbone (N=8)",
        "6-Model Consensus Engine",
        "High-Clarity MPR & 3D BBox",
        "Automated DOCX / PACS Report"
    ]
    step_colors = [CYAN_ACCENT, CYAN_ACCENT, BLUE_DARK, AMBER_DARK, GREEN_DARK]
    
    total_w = Inches(11.7)
    step_w = total_w / len(steps)
    for i, (st, col) in enumerate(zip(steps, step_colors)):
        sb = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i * 2.36), Inches(6.1), Inches(2.26), Inches(0.55))
        sb.fill.solid()
        sb.fill.fore_color.rgb = col
        sb.line.fill.background()
        
        p_s = sb.text_frame.paragraphs[0]
        p_s.text = st
        p_s.font.size = Pt(9.5)
        p_s.font.bold = True
        p_s.font.color.rgb = WHITE
        p_s.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 3: Advantage of Mod-Seg-SE(2) (matching Reference Screenshot 2 - slide 7)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Advantage of Mod-Seg-SE(2)", "Clinical Superiority Against Human Radiologists and Baseline CNNs", slide_num=7)

    adv_cards = [
        {
            "title": "Objective Tumor Boundary Delineation",
            "without_title": "Without",
            "without_text": "Same CT scan, 3 readers delineate different margins (inter-observer Dice 65–75%)",
            "with_title": "With Mod-Seg-SE(2)",
            "with_text": "One reproducible consensus outline + 6-model agreement heatmap (Dice 84.2%)"
        },
        {
            "title": "Higher Segmentation & Diagnostic Accuracy",
            "without_title": "Without",
            "without_text": "Standard U-Net: 74.2% Dice\nAttention U-Net: 76.5% Dice\nHigh false positive rate",
            "with_title": "With Mod-Seg-SE(2)",
            "with_text": "Mod-Seg-SE(2): 84.2% Dice | 73.1% IoU\nSOTA performance on challenging NCCT"
        },
        {
            "title": "Stable Delineation Under Head Tilt & Rotation",
            "without_title": "Without",
            "without_text": "Tilting head or gantry by 45° drops conventional CNN Dice by 12–18%",
            "with_title": "With Mod-Seg-SE(2)",
            "with_text": "Rotation-Equivariant (N=8 group convolutions): identical boundary at any angle"
        },
        {
            "title": "Consistent, Structured Volumetric Reporting",
            "without_title": "Without",
            "without_text": "Subjective free-text notes ('moderate mass effect in parietal lobe'), hard to compare",
            "with_title": "With Mod-Seg-SE(2)",
            "with_text": "Automated 3D volume (mL), min/max lesion diameter, risk tier & 1-click Word report"
        }
    ]

    for idx, card_info in enumerate(adv_cards):
        col = idx % 2
        row = idx // 2
        left = Inches(0.8 + col * 5.95)
        top = Inches(1.3 + row * 2.7)
        
        # Outer Card Box
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.75), Inches(2.5))
        card.fill.solid()
        card.fill.fore_color.rgb = BG_CARD
        card.line.color.rgb = BORDER_CARD
        card.line.width = Pt(1.5)
        
        tf = card.text_frame
        tf.word_wrap = True
        p_title = tf.paragraphs[0]
        p_title.text = card_info["title"]
        p_title.font.size = Pt(14)
        p_title.font.bold = True
        p_title.font.color.rgb = BLUE_DARK
        
        # Inner Left: Without Box
        w_box = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left + Inches(0.15), top + Inches(0.45), Inches(2.6), Inches(1.85))
        w_box.fill.solid()
        w_box.fill.fore_color.rgb = RED_LIGHT
        w_box.line.color.rgb = RED_DARK
        tf_w = w_box.text_frame
        tf_w.word_wrap = True
        pw1 = tf_w.paragraphs[0]
        pw1.text = f"✖ {card_info['without_title']}"
        pw1.font.size = Pt(11)
        pw1.font.bold = True
        pw1.font.color.rgb = RED_DARK
        pw2 = tf_w.add_paragraph()
        pw2.text = card_info["without_text"]
        pw2.font.size = Pt(10)
        pw2.font.color.rgb = TEXT_MAIN
        pw2.space_before = Pt(3)

        # Inner Right: With Box
        wt_box = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left + Inches(2.9), top + Inches(0.45), Inches(2.7), Inches(1.85))
        wt_box.fill.solid()
        wt_box.fill.fore_color.rgb = GREEN_LIGHT
        wt_box.line.color.rgb = GREEN_DARK
        tf_wt = wt_box.text_frame
        tf_wt.word_wrap = True
        pwt1 = tf_wt.paragraphs[0]
        pwt1.text = f"✔ {card_info['with_title']}"
        pwt1.font.size = Pt(11)
        pwt1.font.bold = True
        pwt1.font.color.rgb = GREEN_DARK
        pwt2 = tf_wt.add_paragraph()
        pwt2.text = card_info["with_text"]
        pwt2.font.size = Pt(10)
        pwt2.font.color.rgb = TEXT_MAIN
        pwt2.space_before = Pt(3)

    # =========================================================================
    # SLIDE 4: Market Opportunity (matching Reference Screenshot 3 - slide 8)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Current Market Opportunity & Clinical Positioning", 
               "AI in Neuroimaging market projection: US$14.2 B by 2032 (Growth 28.5%/year)", slide_num=8)

    # Comparison Table
    rows, cols = 7, 5
    table_shape = s4.shapes.add_table(rows, cols, Inches(0.8), Inches(1.3), Inches(8.3), Inches(5.0))
    table = table_shape.table
    table.columns[0].width = Inches(3.1)
    table.columns[1].width = Inches(1.3)
    table.columns[2].width = Inches(1.3)
    table.columns[3].width = Inches(1.3)
    table.columns[4].width = Inches(1.3)

    headers = ["Capability / Feature", "Stroke CAD\n(RapidAI/Viz)", "General AI\n(Aidoc)", "Academic SOTA\n(nnU-Net)", "Mod-Seg-SE(2)\n[OURS]"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY if j < 4 else GREEN_DARK
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    data_matrix = [
        ("Full 3D Multi-Planar Volumetry (Ax, Cor, Sag)", "⚪ No", "⚪ No", "🟡 Partial", "🟢 YES"),
        ("Non-Contrast CT (NCCT) Tumor Detection", "⚪ No", "🟡 Hemorrhage", "🟡 CT Only", "🟢 YES"),
        ("Group Equivariant Invariance (SE(2) N=8)", "⚪ No", "⚪ No", "⚪ No", "🟢 YES"),
        ("Multi-Model Consensus & Uncertainty Map", "⚪ No", "⚪ No", "⚪ No", "🟢 YES"),
        ("Zero-Install Browser Web & PACS Integration", "🟡 Scope-locked", "🟡 Cloud-locked", "⚪ Script only", "🟢 YES"),
        ("Automated Downloadable Clinical Word/DICOM", "⚪ No", "🟡 PDF only", "⚪ No", "🟢 YES")
    ]

    for i, row in enumerate(data_matrix):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BG_CARD if (i % 2 == 0) else WHITE
            if j == 4:
                cell.fill.fore_color.rgb = RGBColor(240, 253, 244) # highlighted column for ours
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            if j == 0:
                p.alignment = PP_ALIGN.LEFT
                p.font.bold = True
                p.font.color.rgb = TEXT_MAIN
            else:
                p.alignment = PP_ALIGN.CENTER
                p.font.color.rgb = GREEN_DARK if "YES" in val else (AMBER_DARK if "Partial" in val or "Only" in val or "locked" in val or "Hemorrhage" in val else TEXT_MUTED)

    # Right Callout Boxes: "The open space" and "Two buyers, one engine"
    r1 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.4), Inches(1.3), Inches(3.1), Inches(2.6))
    r1.fill.solid()
    r1.fill.fore_color.rgb = RGBColor(254, 242, 242)
    r1.line.color.rgb = RED_DARK
    tf_r1 = r1.text_frame
    tf_r1.word_wrap = True
    pr1 = tf_r1.paragraphs[0]
    pr1.text = "The Open Space"
    pr1.font.size = Pt(15)
    pr1.font.bold = True
    pr1.font.color.rgb = RED_DARK
    pr2 = tf_r1.add_paragraph()
    pr2.text = "No approved clinical AI software performs automated brain tumor segmentation on raw Non-Contrast CT with rotation-equivariant invariance and zero-install browser deployment.\n\nTraditional tools fail when heads are tilted; our platform fills this clinical void."
    pr2.font.size = Pt(10.5)
    pr2.font.color.rgb = TEXT_MAIN
    pr2.space_before = Pt(4)

    r2 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.4), Inches(4.1), Inches(3.1), Inches(2.2))
    r2.fill.solid()
    r2.fill.fore_color.rgb = RGBColor(240, 249, 255)
    r2.line.color.rgb = CYAN_ACCENT
    tf_r2 = r2.text_frame
    tf_r2.word_wrap = True
    pr3 = tf_r2.paragraphs[0]
    pr3.text = "Two Buyers, One Engine"
    pr3.font.size = Pt(15)
    pr3.font.bold = True
    pr3.font.color.rgb = BLUE_DARK
    pr4 = tf_r2.add_paragraph()
    pr4.text = "1. Hospital Routine Care: Rapid neuro-oncology triage, volumetric mass effect quantification, and surgical planning.\n\n2. Clinical Drug Trials: Automated, reproducible central reading without radiologist fatigue or rater bias."
    pr4.font.size = Pt(10.5)
    pr4.font.color.rgb = TEXT_MAIN
    pr4.space_before = Pt(4)

    # Status Note Banner
    st_banner = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    st_banner.fill.solid()
    st_banner.fill.fore_color.rgb = AMBER_LIGHT
    st_banner.line.color.rgb = AMBER_DARK
    p_sb = st_banner.text_frame.paragraphs[0]
    p_sb.text = "Present status: Working Web Platform Prototype (Verified on DGX-H100 & Taiwan Patient Cohort). Validated on 6 Baseline Architectures."
    p_sb.font.size = Pt(10)
    p_sb.font.color.rgb = AMBER_DARK
    p_sb.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 5: Mod-Seg-SE(2) Architecture (The 1 Extra Architecture Slide!)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Mod-Seg-SE(2) Architecture & Training Pipeline", 
               "Group Equivariant Convolutional Network (C8 / N=8 Rotations) for Robust Brain CT Segmentation", slide_num=9)

    # Left: Official Architecture Figure
    arch_img_path = "presentation_assets/Fig5_Mod_Seg_SE2_Architecture.png"
    if os.path.exists(arch_img_path):
        s5.shapes.add_picture(arch_img_path, Inches(0.8), Inches(1.3), Inches(5.6), Inches(5.2))

    # Right: Architecture Breakdown & Mathematical Specifications
    arch_card = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.6), Inches(1.3), Inches(5.9), Inches(5.2))
    arch_card.fill.solid()
    arch_card.fill.fore_color.rgb = BG_CARD
    arch_card.line.color.rgb = BORDER_CARD
    arch_card.line.width = Pt(1.5)
    
    tf_a = arch_card.text_frame
    tf_a.word_wrap = True
    
    sections = [
        ("1. 2.5D Spatial Stacking Input", "• Input tensor: [B, 3, 256, 256] stacking [img_prev, img_curr, img_next]\n• Provides 3D volumetric context without massive 3D-CNN compute overhead."),
        ("2. Group Equivariant R2Conv Encoder", "• Discrete rotation group C8 (N=8 rotations spaced by 45°)\n• Steerable filter banks guarantee f(g • x) = g • f(x)\n• Invariant to scanner couch indexing & patient head tilt."),
        ("3. Equivariant Bottleneck & Decoder", "• Pointwise Max Pooling & Equivariant Group Normalization\n• R2Upsampling + skip connection direct sum concatenation\n• Preserves fine sub-millimeter tumor boundary details."),
        ("4. Projection Head & Advanced Loss", "• Equivariant-to-Trivial projection (1x1 conv) -> 2-class softmax\n• AdvancedCombinedLoss: L = 2.0*Dice + 0.5*Focal + 0.5*Edge\n• Class weighting [1.0, 10.0] tackles extreme foreground/background tumor imbalance."),
        ("5. Validated Performance", "• Mod-Seg-SE(2) achieves 84.2% Dice & 73.1% IoU on challenging NCCT\n• Retains >82% Dice under severe 45°/90°/180° scan rotations.")
    ]
    
    for i, (title, content) in enumerate(sections):
        p_t = tf_a.paragraphs[0] if i == 0 else tf_a.add_paragraph()
        p_t.text = title
        p_t.font.size = Pt(12)
        p_t.font.bold = True
        p_t.font.color.rgb = BLUE_DARK
        p_t.space_before = Pt(6) if i > 0 else Pt(0)
        
        p_c = tf_a.add_paragraph()
        p_c.text = content
        p_c.font.size = Pt(10)
        p_c.font.color.rgb = TEXT_MAIN
        p_c.space_before = Pt(2)

    # Save presentation
    output_pptx = "Brain_CT_Seg_Client_Presentation.pptx"
    prs.save(output_pptx)
    print(f"Presentation saved successfully to: {output_pptx}")

if __name__ == "__main__":
    build_presentation()
