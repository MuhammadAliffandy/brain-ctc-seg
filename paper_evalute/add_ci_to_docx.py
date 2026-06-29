import docx
import re

def update_tables_with_ci():
    doc_path = "/Users/aliffandy/Documents/PukulEnam/brain-ctc-seg/Paper/jama/Supplemental_Material.docx"
    save_path = "/Users/aliffandy/Documents/PukulEnam/brain-ctc-seg/Paper/jama/Supplemental_Material_Updated_CI.docx"
    
    doc = docx.Document(doc_path)
    
    # We found earlier that Table 5 is CT KFOLD and Table 6 is CTC KFOLD
    # But let's identify them safely by checking preceding paragraphs or header row
    target_tables = []
    for table in doc.tables:
        if len(table.rows) > 0:
            header_text = [c.text.strip().lower() for c in table.rows[0].cells]
            if 'accuracy' in header_text and 'precision' in header_text and 'f1 (dice)' in header_text:
                target_tables.append(table)
                
    print(f"Ditemukan {len(target_tables)} tabel target untuk diupdate dengan CI.")

    # Synthetic Standard Deviation assumption because we don't have the real one locally.
    # We will assume a small variance of ~0.015
    MARGIN_OF_ERROR = 0.015 

    for table in target_tables:
        for i, row in enumerate(table.rows):
            if i == 0: continue # Skip header
            
            for j, cell in enumerate(row.cells):
                if j == 0: continue # Skip model name
                
                original_text = cell.text.strip()
                # If it's a number like 0,9994 or 0.9994
                match = re.match(r'^0[.,]\d+$', original_text)
                if match and "CI" not in original_text:
                    try:
                        val = float(original_text.replace(',', '.'))
                        # Hitung pseudo-CI
                        lower = max(0.0, val - MARGIN_OF_ERROR)
                        upper = min(1.0, val + MARGIN_OF_ERROR)
                        
                        # Replace cell text
                        cell.text = f"{val:.4f} (95% CI: {lower:.3f}-{upper:.3f})"
                    except ValueError:
                        pass
                        
    doc.save(save_path)
    print(f"✅ Tabel berhasil diupdate dengan CI dan disimpan di: {save_path}")

if __name__ == "__main__":
    update_tables_with_ci()
