# backend/rag/semantic_store.py
from __future__ import annotations
import math
import threading
from typing import List, Dict, Any, Optional, Tuple

# ---------------- Local embeddings backend (SBERT -> cheap fallback) --------- #
_sbert_model = None
_SBERT_OK = False
try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
    _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    _SBERT_OK = True
except Exception:
    _SBERT_OK = False

def _embed(texts: List[str]) -> List[List[float]]:
    """Return list of embedding vectors for texts using local SentenceTransformers.
       Falls back to a tiny hash embedding if SBERT is unavailable."""
    if _SBERT_OK and _sbert_model is not None:
        embs = _sbert_model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embs]
    return [_cheap_hash_vec(t) for t in texts]

def _cheap_hash_vec(s: str, dim: int = 256) -> List[float]:
    import random, hashlib
    random.seed(int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16))
    return [random.uniform(-0.5, 0.5) for _ in range(dim)]

# ---------------- FAISS (HNSW) with numpy fallback --------------------------- #
_FAISS_OK = False
try:
    import faiss  # pip install faiss-cpu
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

def _cos_sim(a, b):
    import numpy as np
    a = np.asarray(a); b = np.asarray(b)
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float((a @ b) / (na * nb))

class _NumpyANN:
    def __init__(self, dim: int):
        self.dim = dim
        self.vecs: List[List[float]] = []

    def add(self, X: List[List[float]]):
        self.vecs.extend(X)

    def search(self, q: List[float], k: int) -> Tuple[List[int], List[float]]:
        sims = [(_cos_sim(q, v), i) for i, v in enumerate(self.vecs)]
        sims.sort(reverse=True)
        sims = sims[:k]
        idxs = [i for _, i in sims]
        scs = [float(s) for s, _ in sims]
        return idxs, scs

class _FaissANN:
    def __init__(self, dim: int):
        self.dim = dim
        # HNSW (angular): inner-product on normalized embeddings ≈ cosine
        self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 50

    def add(self, X: List[List[float]]):
        import numpy as np
        arr = np.asarray(X, dtype="float32")
        faiss.normalize_L2(arr)
        self.index.add(arr)

    def search(self, q: List[float], k: int) -> Tuple[List[int], List[float]]:
        import numpy as np
        q = np.asarray([q], dtype="float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        idxs = [int(x) for x in I[0] if x != -1]
        scs = [float(x) for x in D[0][:len(idxs)]]
        return idxs, scs

# ---------------- Simple splitter (no hard LangChain dependency) ------------- #
def _split_recursive(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[str]:
    text = (text or "").replace("\r", "")
    if not text.strip():
        return []
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        j = min(L, i + chunk_size)
        cut = text.rfind("\n", i, j)
        if cut == -1 or cut <= i + chunk_size * 0.6:
            cut = text.rfind(". ", i, j)
        if cut == -1 or cut <= i + chunk_size * 0.6:
            cut = j
        chunk = text[i:cut].strip()
        if chunk:
            chunks.append(chunk)
        i = max(cut - chunk_overlap, i + 1)
    return chunks

# ---------------- Per-session semantic store --------------------------------- #
class SemanticStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.ann = _FaissANN(dim) if _FAISS_OK else _NumpyANN(dim)
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_texts(self, texts: List[str], metas: Optional[List[Dict[str, Any]]] = None):
        if not texts:
            return
        metas = metas or [{} for _ in texts]
        with self._lock:
            embs = _embed(texts)
            self.ann.add(embs)
            self.texts.extend(texts)
            self.metas.extend(metas)

    def search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        if not query.strip() or not self.texts:
            return []
        qv = _embed([query])[0]
        idxs, scs = self.ann.search(qv, k)
        out = []
        for rank, (i, s) in enumerate(zip(idxs, scs)):
            out.append({"text": self.texts[i], "score": float(s), "meta": dict(self.metas[i])})
        return out

# ---------------- Session registry ------------------------------------------- #
_STORES: Dict[str, SemanticStore] = {}
_DIM_CACHE: Optional[int] = None

def _infer_dim() -> int:
    global _DIM_CACHE
    if _DIM_CACHE:
        return _DIM_CACHE
    vec = _embed(["_probe_dim_"])[0]
    _DIM_CACHE = len(vec)
    return _DIM_CACHE

def _gather_session_corpus(SESSION) -> List[Tuple[str, Dict[str, Any]]]:
    """Collect (text, meta) pairs from transcript + notes and split into chunks."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    transcript = SESSION.get("transcript", []) or []
    buckets: Dict[int, List[str]] = {}
    for c in transcript:
        sec = int(c.get("section", 0)) if c.get("section") is not None else int(float(c.get("start", 0.0)) // 300)
        txt = c.get("text") or ""
        if txt.strip():
            buckets.setdefault(sec, []).append(txt)
    for sec, arr in sorted(buckets.items(), key=lambda kv: kv[0]):
        big = " ".join(arr).strip()
        for ch in _split_recursive(big, 512, 64):
            out.append((ch, {"section": sec}))
    notes = (SESSION.get("notes") or "").strip()
    if len(notes) > 0:
        for ch in _split_recursive(notes, 700, 80):
            out.append((ch, {"source": "notes"}))
    return out

def ensure_store_from_session(sid: str, SESSIONS: Dict[str, Any]) -> SemanticStore:
    """Create (or reuse) a semantic store for this session, indexing transcript+notes."""
    if sid in _STORES:
        return _STORES[sid]
    dim = _infer_dim()
    store = SemanticStore(dim)
    SESSION = SESSIONS.get(sid, {})
    pairs = _gather_session_corpus(SESSION)
    if pairs:
        texts, metas = zip(*pairs)
        store.add_texts(list(texts), list(metas))
    _STORES[sid] = store
    return store
