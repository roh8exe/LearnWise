from sentence_transformers import SentenceTransformer
import faiss, json, glob, os
import numpy as np

EMB = SentenceTransformer("BAAI/bge-base-en-v1.5")
DIM = 768

def load_store() -> tuple[list[str], list[dict]]:
    docs, meta = [], []
    for p in glob.glob("rag/store/*.jsonl"):
        for line in open(p):
            rec = json.loads(line)
            docs.append(rec["text"])
            meta.append({"file": os.path.basename(p), "page": rec["page"],
                         "section": rec["section"], "i": rec["i"]})
    return docs, meta

def build():
    docs, meta = load_store()
    if not docs:
        raise SystemExit("No docs in rag/store; run OCR ingest first.")
    vecs = EMB.encode(docs, normalize_embeddings=True)
    vecs = vecs.astype("float32")
    index = faiss.IndexFlatIP(DIM)
    index.add(vecs)
    faiss.write_index(index, "rag/index.faiss")
    json.dump(meta, open("rag/meta.json", "w"))

if __name__ == "__main__":
    build()
