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

    def add_header(slide, title_text, subtitle_text=None, slide_num=1):
        # Lab Badge
        tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(2.5), Inches(0.6))
        tf = tx_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = "OPTICAL SYSTEMS LAB"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = NAVY
        
        # Slide Title
        t_box = slide.shapes.add_textbox(Inches(3.3), Inches(0.35), Inches(9.2), Inches(0.8))
        tf2 = t_box.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = title_text
        p2.font.size = Pt(21)
        p2.font.bold = True
        p2.font.color.rgb = BLUE_DARK
        
        if subtitle_text:
            p_sub = tf2.add_paragraph()
            p_sub.text = subtitle_text
            p_sub.font.size = Pt(12)
            p_sub.font.color.rgb = TEXT_MUTED

        # Footer
        f_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(6.0), Inches(0.35))
        p_foot = f_box.text_frame.paragraphs[0]
        p_foot.text = "National Taiwan University • NTU GSAT | Brain-CTC-Seg Consortium"
        p_foot.font.size = Pt(10)
        p_foot.font.color.rgb = TEXT_MUTED
        
        # Slide Number
        num_box = slide.shapes.add_textbox(Inches(12.0), Inches(7.05), Inches(0.8), Inches(0.35))
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
               "Overcoming Machine, Deep Learning, and Clinical Radiologist Barriers in Non-Contrast CT", slide_num=1)

    # 3 Top Columns: Machine, Software, Medical Doctor
    cols_data = [
        {
            "header": "Machine / Acquisition",
            "items": ["The rotation & head tilt problem", "Sparse 5mm slice thickness & couch gaps", "Cross-vendor variability (GE, Siemens, Philips)"],
            "color": RED_DARK, "bg": RED_LIGHT
        },
        {
            "header": "Software / AI",
            "items": ["Failure of standard CNNs on rotated scans", "Severe 12–18% Dice drop under domain shift", "Overconfident false positive tumor predictions"],
            "color": AMBER_DARK, "bg": AMBER_LIGHT
        },
        {
            "header": "Medical Doctor",
            "items": ["Subjectivity of human boundary delineation", "High inter-observer variance (Dice 65–75%)", "Subtle tumor contrast on Non-Contrast CT"],
            "color": BLUE_DARK, "bg": RGBColor(238, 242, 255)
        }
    ]

    for i, c in enumerate(cols_data):
        left = Inches(0.8 + i * 4.0)
        top = Inches(1.4)
        card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(3.7), Inches(2.2))
        card.fill.solid()
        card.fill.fore_color.rgb = c["bg"]
        card.line.color.rgb = c["color"]
        card.line.width = Pt(1.5)
        
        tf = card.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = c["header"]
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = c["color"]
        p.alignment = PP_ALIGN.CENTER
        
        for item in c["items"]:
            pi = tf.add_paragraph()
            pi.text = f"• {item}"
            pi.font.size = Pt(11)
            pi.font.color.rgb = TEXT_MAIN
            pi.space_before = Pt(4)

    # Big Solution Box (Mod-Seg-SE2 Platform)
    sol_box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.9), Inches(11.7), Inches(1.85))
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
    p_res.font.size = Pt(15.5)
    p_res.font.bold = True
    p_res.font.color.rgb = WHITE
    p_res.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 2: Framework Prototype (Live Web Application Dashboard)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Framework Prototype", "A clinical brain tumor evaluation platform in a browser tab", slide_num=2)

    # "Live Demo" badge
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
        p_bt.font.size = Pt(14)
        p_bt.font.bold = True
        p_bt.font.color.rgb = b["fg"]
        
        p_bd = tf_b.add_paragraph()
        p_bd.text = b["desc"]
        p_bd.font.size = Pt(10.5)
        p_bd.font.color.rgb = b["fg"]
        p_bd.space_before = Pt(3)

    # Bottom Pipeline Ribbon
    steps = [
        "2.5D Slice Triplet",
        "+ Mod-SE(2) Backbone (N=8)",
        "6-Model Consensus Engine",
        "High-Clarity MPR & 3D BBox",
        "Automated DOCX / PACS Report"
    ]
    step_colors = [CYAN_ACCENT, CYAN_ACCENT, BLUE_DARK, AMBER_DARK, GREEN_DARK]
    
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
    # SLIDE 3: Advantage of Mod-Seg-SE(2) (Visual 2x2 comparison with real scan representations)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Advantage of Mod-Seg-SE(2)", "Clinical Superiority Against Human Radiologists and Baseline CNNs", slide_num=3)

    cards_meta = [
        {"title": "Objective Tumor Boundary Delineation", "img": "presentation_assets/slide_cards/card1_full_visual.png"},
        {"title": "Higher Segmentation & Diagnostic Accuracy", "img": "presentation_assets/slide_cards/card2_full_visual.png"},
        {"title": "Stable Delineation Under Head Tilt & Rotation", "img": "presentation_assets/slide_cards/card3_full_visual.png"},
        {"title": "Consistent, Structured Volumetric Reporting", "img": "presentation_assets/slide_cards/card4_full_visual.png"}
    ]

    for idx, card_info in enumerate(cards_meta):
        col = idx % 2
        row = idx // 2
        left = Inches(0.8 + col * 5.95)
        top = Inches(1.25 + row * 2.8)
        
        # Outer Card Box
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.75), Inches(2.65))
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
        
        # Embed the generated visual card
        if os.path.exists(card_info["img"]):
            s3.shapes.add_picture(card_info["img"], left + Inches(0.12), top + Inches(0.42), Inches(5.51), Inches(2.13))

    # =========================================================================
    # SLIDE 4: Architecture of Mod-Seg-SE(2) (REPLACES MARKET SLIDE)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Mod-Seg-SE(2) Architecture & Pipeline Flow", 
               "Group Equivariant Convolutional Network (C8 / N=8 Rotations) with Multi-Model Consensus Engine", slide_num=4)

    arch_visual_path = "presentation_assets/slide_cards/arch_flow_visual_complete.png"
    if os.path.exists(arch_visual_path):
        s4.shapes.add_picture(arch_visual_path, Inches(0.8), Inches(1.3), Inches(11.73), Inches(5.5))

    # Save presentation
    output_pptx = "Brain_CT_Seg_Client_Presentation.pptx"
    prs.save(output_pptx)
    print(f"Presentation saved successfully with EXACTLY 4 slides to: {output_pptx}")

if __name__ == "__main__":
    build_presentation()
