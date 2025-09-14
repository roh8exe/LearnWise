from typing import List, Dict, Tuple
import re

def _normalize(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def group_blocks(blocks: List[Dict]) -> List[Dict]:
    """Group by (page, section). If no section in OCR, group by page."""
    groups = {}
    for b in blocks:
        key = (b.get("page", 0), b.get("section") or f"page-{b.get('page',0)}")
        groups.setdefault(key, []).append(b)
    # sort each group by bbox top-left (if bbox present)
    grouped = []
    for (page, section), items in groups.items():
        items = sorted(items, key=lambda x: (x.get("bbox",[0,0,0,0])[1], x.get("bbox",[0,0,0,0])[0]))
        text = _normalize(" ".join(i.get("text","") for i in items))
        grouped.append({"page": page, "section": section, "text": text})
    return grouped

def token_chunks(text: str, max_tokens: int=350, overlap: int=60) -> List[str]:
    """
    Simple token-ish splitter by words. For production, use a tokenizer.
    max_tokens and overlap emulate context windows for LLM grounding.
    """
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        j = min(i + max_tokens, len(words))
        chunk = " ".join(words[i:j])
        out.append(chunk)
        if j == len(words): break
        i = max(0, j - overlap)
    return out

def make_chunks(blocks: List[Dict], max_tokens=350, overlap=60) -> List[Dict]:
    grouped = group_blocks(blocks)
    chunks = []
    for g in grouped:
        parts = token_chunks(g["text"], max_tokens=max_tokens, overlap=overlap)
        for k, p in enumerate(parts):
            chunks.append({
                "page": g["page"],
                "section": g["section"],
                "i": k,
                "text": p
            })
    return chunks
