# backend/rag/ingest_doc.py
from .ocr_mistral import run_mistral_ocr
from .chunker import make_chunks
import os, json, uuid

def ingest(file_path: str, out_dir="rag/store") -> str:
    os.makedirs(out_dir, exist_ok=True)
    blocks = run_mistral_ocr(file_path)
    chunks = make_chunks(blocks)
    doc_id = os.path.splitext(os.path.basename(file_path))[0] + "-" + uuid.uuid4().hex[:6]
    out_path = os.path.join(out_dir, f"{doc_id}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return out_path
