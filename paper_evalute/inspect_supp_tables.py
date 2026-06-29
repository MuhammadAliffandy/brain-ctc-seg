import docx

doc_path = "/Users/aliffandy/Documents/PukulEnam/brain-ctc-seg/Paper/jama/Supplemental_Material.docx"
try:
    doc = docx.Document(doc_path)
    print("Tables in Supplemental_Material.docx:")
    for i, table in enumerate(doc.tables):
        print(f"\n--- TABLE {i} ---")
        if len(table.rows) > 0:
            row0 = [cell.text.strip() for cell in table.rows[0].cells]
            print(f"Header: {row0}")
            if len(table.rows) > 1:
                row1 = [cell.text.strip() for cell in table.rows[1].cells]
                print(f"Row 1: {row1}")
except Exception as e:
    print(e)
