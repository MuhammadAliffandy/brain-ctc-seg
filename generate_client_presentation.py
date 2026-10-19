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
    PURPLE_DARK = RGBColor(124, 58, 237)
    PURPLE_LIGHT = RGBColor(245, 243, 255)

    logo_osl = "presentation_assets/logo_osl.png"
    logo_ntu = "presentation_assets/logo_ntu.png"

    def add_slide_chrome(slide, title_text, subtitle_text=None, slide_num=1):
        # 1. Top-Left OSL Logo (cleaned without black bar)
        if os.path.exists(logo_osl):
            slide.shapes.add_picture(logo_osl, Inches(0.6), Inches(0.35), Inches(1.8), Inches(0.75))
        else:
            tx_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(2.2), Inches(0.6))
            p = tx_box.text_frame.paragraphs[0]
            p.text = "OPTICAL SYSTEMS LAB"
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = NAVY

        # 2. Slide Title
        t_box = slide.shapes.add_textbox(Inches(2.7), Inches(0.35), Inches(9.8), Inches(0.85))
        tf = t_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(21)
        p.font.bold = True
        p.font.color.rgb = BLUE_DARK
        
        if subtitle_text:
            p_sub = tf.add_paragraph()
            p_sub.text = subtitle_text
            p_sub.font.size = Pt(12)
            p_sub.font.color.rgb = TEXT_MUTED

        # 3. Bottom-Left NTU Logo
        if os.path.exists(logo_ntu):
            slide.shapes.add_picture(logo_ntu, Inches(0.6), Inches(6.8), Inches(2.6), Inches(0.55))
        else:
            f_box = slide.shapes.add_textbox(Inches(0.6), Inches(6.9), Inches(5.0), Inches(0.4))
            p_foot = f_box.text_frame.paragraphs[0]
            p_foot.text = "National Taiwan University • NTU GSAT"
            p_foot.font.size = Pt(10)
            p_foot.font.color.rgb = TEXT_MUTED

        # 4. Bottom-Right Slide Number
        num_box = slide.shapes.add_textbox(Inches(12.2), Inches(6.85), Inches(0.6), Inches(0.4))
        p_num = num_box.text_frame.paragraphs[0]
        p_num.text = str(slide_num)
        p_num.font.size = Pt(12)
        p_num.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # SLIDE 1: Problem Formulation & Core Platform
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    add_slide_chrome(s1, "Brain-CT-Seg: AI for Brain Tumor Segmentation & Risk Volumetry",
                     "Overcoming Machine, Deep Learning, and Clinical Radiologist Barriers in Non-Contrast CT", slide_num=1)

    # 3 Top Columns: Machine, Software, Medical Doctor
    # Machine Column
    col1_x = Inches(1.0)
    s1.shapes.add_textbox(col1_x, Inches(1.3), Inches(3.4), Inches(0.35)).text_frame.paragraphs[0].text = "Machine / Acquisition"
    s1.shapes[-1].text_frame.paragraphs[0].font.size = Pt(14)
    s1.shapes[-1].text_frame.paragraphs[0].font.bold = True
    s1.shapes[-1].text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    ct_mach_path = "presentation_assets/ct_scanner_machine.jpg"
    if os.path.exists(ct_mach_path):
        s1.shapes.add_picture(ct_mach_path, col1_x + Inches(0.4), Inches(1.65), Inches(2.6), Inches(1.35))

    t1_box = s1.shapes.add_textbox(col1_x, Inches(3.05), Inches(3.4), Inches(0.7))
    tf1 = t1_box.text_frame
    tf1.word_wrap = True
    p1a = tf1.paragraphs[0]
    p1a.text = "• The rotation problem (head tilt & couch skew)"
    p1a.font.size = Pt(10)
    p1a.font.color.rgb = TEXT_MAIN
    p1b = tf1.add_paragraph()
    p1b.text = "• Multi-vendor variability (Siemens, GE, Philips)"
    p1b.font.size = Pt(10)
    p1b.font.color.rgb = TEXT_MAIN

    # Software Column
    col2_x = Inches(5.0)
    s1.shapes.add_textbox(col2_x, Inches(1.3), Inches(3.4), Inches(0.35)).text_frame.paragraphs[0].text = "Software / Deep Learning"
    s1.shapes[-1].text_frame.paragraphs[0].font.size = Pt(14)
    s1.shapes[-1].text_frame.paragraphs[0].font.bold = True
    s1.shapes[-1].text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    sw_path = "presentation_assets/ref_software.png"
    if os.path.exists(sw_path):
        s1.shapes.add_picture(sw_path, col2_x + Inches(0.75), Inches(1.65), Inches(1.9), Inches(1.35))

    t2_box = s1.shapes.add_textbox(col2_x, Inches(3.05), Inches(3.4), Inches(0.7))
    tf2 = t2_box.text_frame
    tf2.word_wrap = True
    p2a = tf2.paragraphs[0]
    p2a.text = "• Failure of standard CNNs on tilted/rotated scans"
    p2a.font.size = Pt(10)
    p2a.font.color.rgb = TEXT_MAIN
    p2b = tf2.add_paragraph()
    p2b.text = "• Severe 12–18% Dice drop under domain shift on NCCT"
    p2b.font.size = Pt(10)
    p2b.font.color.rgb = TEXT_MAIN

    # Medical Doctor Column
    col3_x = Inches(8.9)
    s1.shapes.add_textbox(col3_x, Inches(1.3), Inches(3.4), Inches(0.35)).text_frame.paragraphs[0].text = "Medical Doctor / Radiologist"
    s1.shapes[-1].text_frame.paragraphs[0].font.size = Pt(14)
    s1.shapes[-1].text_frame.paragraphs[0].font.bold = True
    s1.shapes[-1].text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    doc_path = "presentation_assets/slide1_doctor_visual.png"
    if os.path.exists(doc_path):
        s1.shapes.add_picture(doc_path, col3_x + Inches(0.5), Inches(1.65), Inches(2.4), Inches(1.35))

    t3_box = s1.shapes.add_textbox(col3_x, Inches(3.05), Inches(3.4), Inches(0.7))
    tf3 = t3_box.text_frame
    tf3.word_wrap = True
    p3a = tf3.paragraphs[0]
    p3a.text = "• Subjectivity of manual tumor boundary delineation"
    p3a.font.size = Pt(10)
    p3a.font.color.rgb = TEXT_MAIN
    p3b = tf3.add_paragraph()
    p3b.text = "• High inter-observer variance (radiologist Dice 65–75%)"
    p3b.font.size = Pt(10)
    p3b.font.color.rgb = TEXT_MAIN

    # Central Box (CT-SE(2) Platform)
    box_w = Inches(11.3)
    box_x = Inches(1.0)
    box_y = Inches(3.9)
    box_h = Inches(1.9)

    sol_box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, box_x, box_y, box_w, box_h)
    sol_box.fill.solid()
    sol_box.fill.fore_color.rgb = WHITE
    sol_box.line.color.rgb = GREEN_DARK
    sol_box.line.width = Pt(2.0)

    # Box title
    title_box = s1.shapes.add_textbox(box_x, box_y + Inches(0.1), box_w, Inches(0.4))
    pt = title_box.text_frame.paragraphs[0]
    pt.text = "CT-SE(2) Consensus Platform"
    pt.font.size = Pt(18)
    pt.font.bold = True
    pt.font.color.rgb = GREEN_DARK
    pt.alignment = PP_ALIGN.CENTER

    # Subtitle
    sub_box = s1.shapes.add_textbox(box_x, box_y + Inches(0.48), box_w, Inches(0.35))
    ps = sub_box.text_frame.paragraphs[0]
    ps.text = "Equivariant Group Theory Backbone (C8 / N=8 discrete rotations) + 6-Model Consensus Engine"
    ps.font.size = Pt(12)
    ps.font.bold = True
    ps.font.color.rgb = NAVY
    ps.alignment = PP_ALIGN.CENTER

    # 3 Feature Pills inside the solution box
    pills = [
        ("Equivariant R2Conv (C8)", "Guarantees f(g·x) = g·f(x). Boundary is identical at any scan angle."),
        ("6-Model Consensus Engine", "Combines CT-SE(2), HarmonicNet, nnU-Net with calibrated 0.80 threshold."),
        ("Automated 3D Volumetry", "Quantifies Tumor Volume (mL), Edema, Sphericity, and Midline Shift.")
    ]
    for p_idx, (p_title, p_desc) in enumerate(pills):
        p_left = box_x + Inches(0.3 + p_idx * 3.6)
        p_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, p_left, box_y + Inches(0.85), Inches(3.4), Inches(0.9))
        p_card.fill.solid()
        p_card.fill.fore_color.rgb = BG_CARD
        p_card.line.color.rgb = BORDER_CARD
        
        tf_pc = p_card.text_frame
        tf_pc.word_wrap = True
        pp1 = tf_pc.paragraphs[0]
        pp1.text = p_title
        pp1.font.size = Pt(11)
        pp1.font.bold = True
        pp1.font.color.rgb = BLUE_DARK
        
        pp2 = tf_pc.add_paragraph()
        pp2.text = p_desc
        pp2.font.size = Pt(9.2)
        pp2.font.color.rgb = TEXT_MAIN
        pp2.space_before = Pt(2)

    # Arrow and Final Score Banner
    res_box = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(2.2), Inches(5.95), Inches(8.9), Inches(0.65))
    res_box.fill.solid()
    res_box.fill.fore_color.rgb = GREEN_DARK
    res_box.line.fill.background()
    
    tfr = res_box.text_frame
    pr1 = tfr.paragraphs[0]
    pr1.text = "84.2% Dice | 73.1% IoU SOTA Segmentation & Quantitative 3D Volumetry"
    pr1.font.size = Pt(15.5)
    pr1.font.bold = True
    pr1.font.color.rgb = WHITE
    pr1.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 2: Framework Prototype (Live Web Application Dashboard)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_chrome(s2, "Framework Prototype", "A clinical brain tumor evaluation platform in a browser tab", slide_num=2)

    demo_badge = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.8), Inches(0.4), Inches(1.8), Inches(0.45))
    demo_badge.fill.solid()
    demo_badge.fill.fore_color.rgb = RGBColor(99, 102, 241)
    demo_badge.line.fill.background()
    p_db = demo_badge.text_frame.paragraphs[0]
    p_db.text = "Live Demo ⚡"
    p_db.font.size = Pt(12)
    p_db.font.bold = True
    p_db.font.color.rgb = WHITE
    p_db.alignment = PP_ALIGN.CENTER

    mpr_img_path = "presentation_assets/dashboard_mpr_consensus_updated.png"
    if os.path.exists(mpr_img_path):
        s2.shapes.add_picture(mpr_img_path, Inches(0.8), Inches(1.3), Inches(8.8), Inches(4.5))

    badges = [
        {"title": "Zero-install", "desc": "No GPU required on client,\nInstant browser access via Ngrok", "bg": GREEN_DARK, "fg": WHITE},
        {"title": "Vendor-agnostic", "desc": "Calibrated for NCCT & CTC\nacross Siemens, GE, Philips", "bg": AMBER_DARK, "fg": WHITE},
        {"title": "Privacy by design", "desc": "Local edge server inference,\nAutomated PACS & DOCX reports", "bg": RED_DARK, "fg": WHITE}
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
        p_bt.font.size = Pt(14)
        p_bt.font.bold = True
        p_bt.font.color.rgb = b["fg"]
        
        p_bd = tf_b.add_paragraph()
        p_bd.text = b["desc"]
        p_bd.font.size = Pt(10.5)
        p_bd.font.color.rgb = b["fg"]
        p_bd.space_before = Pt(3)

    steps = [
        "2.5D Slice Triplet",
        "+ CT-SE(2) Backbone (N=8)",
        "6-Model Consensus Engine",
        "High-Clarity MPR & 3D Brain",
        "Automated DOCX / PACS Report"
    ]
    step_colors = [CYAN_ACCENT, CYAN_ACCENT, BLUE_DARK, AMBER_DARK, GREEN_DARK]
    
    for i, (st, col) in enumerate(zip(steps, step_colors)):
        sb = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i * 2.36), Inches(6.05), Inches(2.26), Inches(0.55))
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
    # SLIDE 3: Advantage of CT-SE(2) (Visual Brain CT Datasets)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_chrome(s3, "Advantage of CT-SE(2)", "Clinical Superiority Against Human Radiologists and Baseline CNNs", slide_num=3)

    cards_meta = [
        {"title": "Objective tumor boundary delineation", "img": "presentation_assets/slide_cards/card1_full_visual.png"},
        {"title": "Higher diagnostic accuracy", "img": "presentation_assets/slide_cards/card2_full_visual.png"},
        {"title": "Stable delineation of borderline & tilted cases", "img": "presentation_assets/slide_cards/card3_full_visual.png"},
        {"title": "Consistent, structured reporting", "img": "presentation_assets/slide_cards/card4_full_visual.png"}
    ]

    for idx, card_info in enumerate(cards_meta):
        col = idx % 2
        row = idx // 2
        left = Inches(0.8 + col * 5.95)
        top = Inches(1.2 + row * 2.7)
        
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.75), Inches(2.55))
        card.fill.solid()
        card.fill.fore_color.rgb = BG_CARD
        card.line.color.rgb = BORDER_CARD
        card.line.width = Pt(1.5)
        
        tf = card.text_frame
        tf.word_wrap = True
        p_title = tf.paragraphs[0]
        p_title.text = card_info["title"]
        p_title.font.size = Pt(13)
        p_title.font.bold = True
        p_title.font.color.rgb = BLUE_DARK
        
        if os.path.exists(card_info["img"]):
            s3.shapes.add_picture(card_info["img"], left + Inches(0.12), top + Inches(0.38), Inches(5.51), Inches(2.05))

    # =========================================================================
    # SLIDE 4: Architecture of CT-SE(2) (Clean Native PPT Design with Real Scans)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_chrome(s4, "CT-SE(2) Architecture & Pipeline Flow", 
                     "Group Equivariant Convolutional Network (C8 / N=8 Rotations) with Multi-Model Consensus Engine", slide_num=4)

    # 5 Top Pipeline Stage Cards
    stages_data = [
        {
            "num": "1. 2.5D CT Triplet",
            "img": "presentation_assets/real_raw_ct_input.png",
            "desc": "• Raw CT [B, 3, 256, 256]\n• [z-1, z, z+1] Inter-slice\n• Real Patient NCCT scan",
            "color": CYAN_ACCENT, "bg": RGBColor(240, 249, 255)
        },
        {
            "num": "2. C8 Group R2Conv",
            "img": "presentation_assets/c8_rotation_roset.png",
            "desc": "• 8 Discrete Rotations (45°)\n• Pointwise MaxPool (2x2)\n• Channels: 24 -> 48 -> 96",
            "color": BLUE_DARK, "bg": RGBColor(239, 246, 255)
        },
        {
            "num": "3. Group Bottleneck",
            "img": "presentation_assets/deep_latent_feature.png",
            "desc": "• 192 Equivariant channels\n• Invariant representation\n• f(g · x) = g · f(x)",
            "color": PURPLE_DARK, "bg": PURPLE_LIGHT
        },
        {
            "num": "4. Equivariant Decoder",
            "img": "presentation_assets/decoder_feature.png",
            "desc": "• R2Upsampling (2x scale)\n• Direct-Sum Skip (⊕)\n• Sub-mm edge refinement",
            "color": BLUE_DARK, "bg": RGBColor(239, 246, 255)
        },
        {
            "num": "5. Output SOTA Mask",
            "img": "presentation_assets/real_se2_pred_output.png",
            "desc": "• 84.2% Dice | 73.1% IoU\n• Real Output from Platform\n• 3D Volumetric ROI mask",
            "color": GREEN_DARK, "bg": GREEN_LIGHT
        }
    ]

    card_w = Inches(2.22)
    card_h = Inches(3.25)
    card_y = Inches(1.4)
    start_x = Inches(0.66)
    gap = Inches(0.23)

    for i, st in enumerate(stages_data):
        cx = start_x + i * (card_w + gap)
        
        # Outer Card
        scard = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, card_y, card_w, card_h)
        scard.fill.solid()
        scard.fill.fore_color.rgb = st["bg"]
        scard.line.color.rgb = st["color"]
        scard.line.width = Pt(1.5)
        
        # Stage Header
        th = s4.shapes.add_textbox(cx, card_y + Inches(0.08), card_w, Inches(0.35))
        pth = th.text_frame.paragraphs[0]
        pth.text = st["num"]
        pth.font.size = Pt(11)
        pth.font.bold = True
        pth.font.color.rgb = st["color"]
        pth.alignment = PP_ALIGN.CENTER
        
        # Embedded Image
        if os.path.exists(st["img"]):
            s4.shapes.add_picture(st["img"], cx + Inches(0.21), card_y + Inches(0.45), Inches(1.8), Inches(1.65))
            
        # Description text below image
        td = s4.shapes.add_textbox(cx + Inches(0.05), card_y + Inches(2.15), card_w - Inches(0.1), Inches(1.0))
        td.text_frame.word_wrap = True
        for line_i, line_t in enumerate(st["desc"].split("\n")):
            p_line = td.text_frame.paragraphs[0] if line_i == 0 else td.text_frame.add_paragraph()
            p_line.text = line_t
            p_line.font.size = Pt(9.2)
            p_line.font.color.rgb = TEXT_MAIN
            if line_i > 0:
                p_line.space_before = Pt(2)

        # Arrow to next stage
        if i < 4:
            arr_box = s4.shapes.add_textbox(cx + card_w, card_y + Inches(1.2), gap, Inches(0.4))
            p_arr = arr_box.text_frame.paragraphs[0]
            p_arr.text = "➔"
            p_arr.font.size = Pt(14)
            p_arr.font.bold = True
            p_arr.font.color.rgb = TEXT_MUTED
            p_arr.alignment = PP_ALIGN.CENTER

    # Skip Connection Badge across Stages 2-4
    skip_badge = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.1), Inches(1.22), Inches(7.1), Inches(0.32))
    skip_badge.fill.solid()
    skip_badge.fill.fore_color.rgb = PURPLE_DARK
    skip_badge.line.fill.background()
    p_sk = skip_badge.text_frame.paragraphs[0]
    p_sk.text = "Equivariant Direct-Sum Skip Connections (⊕ Preserves Sub-Millimeter Boundaries)"
    p_sk.font.size = Pt(9.5)
    p_sk.font.bold = True
    p_sk.font.color.rgb = WHITE
    p_sk.alignment = PP_ALIGN.CENTER

    # 3 Bottom Highlight Cards
    b_y = Inches(4.85)
    b_h = Inches(1.85)
    b_w = Inches(3.85)
    
    bottom_data = [
        {
            "title": "Advanced Combined Loss Function",
            "lines": [
                "L_total = 2.0*Dice + 0.5*Focal + 0.5*Edge",
                "• Class Weighting [1.0, 10.0] tackles extreme foreground imbalance",
                "• Focal Loss (gamma=2.0) suppresses trivial brain background",
                "• Edge Loss penalizes fuzzy, indistinct tumor boundaries",
                "• Cosine Annealing Optimizer with Warm Restarts (LR 3e-5)"
            ],
            "color": NAVY, "bg": BG_CARD, "border": BORDER_CARD
        },
        {
            "title": "C8 Group Invariance Guarantee",
            "lines": [
                "ρ_out(g)[K ⋆ f](x) = [K ⋆ ρ_in(g)f](x)",
                "• 8 Rotational symmetries spaced by 45° around scanner axis",
                "• Steerable harmonic basis convolutions in R2Conv filter banks",
                "• Mathematical guarantee: tumor boundary is segmented identically",
                "  regardless of patient head tilt or scanner couch skew angle"
            ],
            "color": BLUE_DARK, "bg": RGBColor(239, 246, 255), "border": RGBColor(147, 197, 253)
        },
        {
            "title": "Multi-Model Consensus & 3D Volumetry",
            "lines": [
                "• 6 Benchmark Architectures Ensemble Voting",
                "  (CT-SE(2), HarmonicNet C4, nnU-Net, Attention U-Net)",
                "• CT-SE(2) SOTA: 84.2% Dice | 73.1% IoU on NCCT",
                "• Pixel-wise uncertainty agreement heatmap at 0.80 threshold",
                "• Automated 3D Volumetry: Tumor (mL), Edema (mL), Sphericity"
            ],
            "color": GREEN_DARK, "bg": RGBColor(240, 253, 244), "border": RGBColor(134, 239, 172)
        }
    ]

    for b_idx, b_info in enumerate(bottom_data):
        bx = Inches(0.66 + b_idx * 4.1)
        b_box = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, b_y, b_w, b_h)
        b_box.fill.solid()
        b_box.fill.fore_color.rgb = b_info["bg"]
        b_box.line.color.rgb = b_info["border"]
        b_box.line.width = Pt(1.5)
        
        tf_bb = b_box.text_frame
        tf_bb.word_wrap = True
        p_bt = tf_bb.paragraphs[0]
        p_bt.text = b_info["title"]
        p_bt.font.size = Pt(11)
        p_bt.font.bold = True
        p_bt.font.color.rgb = b_info["color"]
        
        for li, lt in enumerate(b_info["lines"]):
            p_lt = tf_bb.add_paragraph()
            p_lt.text = lt
            p_lt.font.size = Pt(9.0)
            p_lt.font.color.rgb = TEXT_MAIN
            p_lt.space_before = Pt(2.5) if li > 0 else Pt(4)
            if li == 0 and ("L_total" in lt or "ρ_out" in lt):
                p_lt.font.bold = True
                p_lt.font.color.rgb = b_info["color"]

    # Save presentation
    output_pptx = "Brain_CT_Seg_Client_Presentation.pptx"
    prs.save(output_pptx)
    print(f"Presentation saved successfully with EXACTLY 4 slides to: {output_pptx}")

if __name__ == "__main__":
    build_presentation()
