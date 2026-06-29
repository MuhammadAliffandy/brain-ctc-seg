import docx

doc_path = "/Users/aliffandy/Documents/PukulEnam/brain-ctc-seg/Paper/jama/Manuscript_JAMA_Final2_Updated.docx"
doc = docx.Document(doc_path)

print("=== STRUCTURE OF MANUSCRIPT_JAMA_FINAL2_UPDATED.docx ===\n")
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    style = p.style.name
    if text:
        print(f"[{i}] [{style}]: {text[:120]}")
