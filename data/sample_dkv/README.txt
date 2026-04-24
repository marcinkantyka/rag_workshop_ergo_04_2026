PLACEHOLDER DOCUMENTS — REPLACE WITH REAL DKV BELGIUM DOCUMENTS

These files simulate realistic DKV Belgium insurance policy documents in French,
Dutch, and English. They are used for workshop exercises.

Files:
  dkv_hospitalisation_fr.txt     — Hospitalisation policy (French)
  dkv_ziekenhuisopname_nl.txt   — Hospitalisation policy (Dutch)
  dkv_hospitalisation_en.txt     — Hospitalisation policy (English)
  dkv_kinesitherapie_nl.txt     — Physiotherapy supplement (Dutch)
  dkv_soins_dentaires_fr.txt    — Dental insurance supplement (French)

To add real DKV PDFs:
  1. Drop PDF files into this directory
  2. Run: python data/load_dkv_documents.py
  3. Indexed documents will appear in chroma_db/
