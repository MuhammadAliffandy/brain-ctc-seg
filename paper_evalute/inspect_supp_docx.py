import docx

doc_path = "/Users/aliffandy/Documents/PukulEnam/brain-ctc-seg/Paper/jama/Supplemental_Material.docx"
try:
    doc = docx.Document(doc_path)
    print("Paragraphs in Supplemental_Material.docx:")
    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        if text:
            print(f"[{i}]: {text}")
        if 'Figure' in text or 'empty' in text.lower() or 'kosong' in text.lower():
            print(f"--- POTENTIAL FIGURE SPOT AT INDEX {i} ---")
except Exception as e:
    print(e)
