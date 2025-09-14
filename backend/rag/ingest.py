# backend/rag/ingest.py
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

# Try sklearn TF-IDF, fall back to a tiny built-in vectorizer if unavailable
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except Exception:
    TfidfVectorizer = None  # we’ll use _MiniTfidf below


# ---------------------- simple utilities ----------------------

_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z0-9(])")

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    # collapse whitespace and strip weird NBSP etc.
    s = s.replace("\xa0", " ")
    s = _WS_RE.sub(" ", s)
    return s.strip()

def _split_sentences(text: str) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []
    # conservative sentence split
    parts = _SENT_SPLIT_RE.split(text)
    # keep short “sentences” if they carry numbers/symbols
    out = [p.strip() for p in parts if p and p.strip()]
    return out

def _window_by_words(sentences: List[str], target_words: int = 120, overlap_words: int = 24) -> List[str]:
    """
    Build chunks ~target_words with ~20% overlap. Sentence-boundary aware.
    """
    if not sentences:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    count = 0
    for s in sentences:
        w = len(s.split())
        if count + w > target_words and buf:
            # flush current
            chunk = " ".join(buf).strip()
            if chunk:
                chunks.append(chunk)

            # start next with overlap tail
            tail = " ".join(" ".join(buf).split()[-overlap_words:]) if overlap_words > 0 else ""
            buf = [tail] if tail else []
            count = len(tail.split()) if tail else 0

        buf.append(s)
        count += w

    if buf:
        chunk = " ".join(buf).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ---------------------- fallback TF-IDF ----------------------

class _MiniTfidf:
    """
    Minimal TF-IDF implementation used only if sklearn is not available.
    Vocabulary is whitespace/letter-digit word-based; lowercase; no stemming.
    """
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        # keep simple words with letters/digits and minimum length 2
        return [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}", text.lower())]

    def fit(self, docs: List[str]):
        N = len(docs)
        df: Dict[str, int] = {}
        for d in docs:
            toks = set(self._tokenize(d))
            for t in toks:
                df[t] = df.get(t, 0) + 1
        self.vocab = {t: i for i, (t, _) in enumerate(sorted(df.items(), key=lambda x: x[0]))}
        # idf = log( (1+N) / (1+df) ) + 1
        idf_vec = np.zeros(len(self.vocab), dtype=np.float32)
        for t, i in self.vocab.items():
            idf_vec[i] = np.log((1.0 + N) / (1.0 + df[t])) + 1.0
        self.idf = idf_vec
        self._fitted = True

    def transform(self, docs: List[str]) -> np.ndarray:
        assert self._fitted and self.idf is not None
        X = np.zeros((len(docs), len(self.vocab)), dtype=np.float32)
        for r, d in enumerate(docs):
            toks = self._tokenize(d)
            if not toks:
                continue
            tf: Dict[int, int] = {}
            for t in toks:
                idx = self.vocab.get(t)
                if idx is not None:
                    tf[idx] = tf.get(idx, 0) + 1
            if not tf:
                continue
            # l2-normalized tf-idf row
            row = np.zeros(len(self.vocab), dtype=np.float32)
            for idx, cnt in tf.items():
                row[idx] = cnt
            # tf -> tfidf
            row = row * self.idf
            norm = np.linalg.norm(row) + 1e-8
            X[r, :] = row / norm
        return X


# ---------------------- in-memory RAG store ----------------------

class _SessionIndex:
    def __init__(self):
        self.chunks: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.vectorizer = None
        self.matrix = None  # np.ndarray or scipy sparse matrix

    def _ensure_fitted(self):
        texts = self.chunks
        if not texts:
            self.vectorizer = None
            self.matrix = None
            return

        if TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            vec = _MiniTfidf()
            vec.fit(texts)
            self.vectorizer = vec
            self.matrix = vec.transform(texts)

    def add_chunk_batch(self, chunk_texts: List[str], base_meta: Optional[Dict[str, Any]] = None):
        base_meta = base_meta or {}
        start_idx = len(self.chunks)
        for i, c in enumerate(chunk_texts):
            c = _normalize_text(c)
            if not c:
                continue
            self.chunks.append(c)
            m = dict(base_meta)
            m["i"] = start_idx + i
            self.metas.append(m)
        self._ensure_fitted()

    def search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        query = _normalize_text(query)
        if not query or self.vectorizer is None or self.matrix is None or len(self.chunks) == 0:
            return []

        if TfidfVectorizer is not None:
            qv = self.vectorizer.transform([query])
            # cosine similarity for l2-normalized rows; matrix is TF-IDF (l2 by default in sklearn)
            scores = (qv @ self.matrix.T).toarray()[0]
        else:
            qv = self.vectorizer.transform([query])
            scores = (qv @ self.matrix.T).astype(np.float32)  # both dense
            scores = np.asarray(scores)[0]

        idxs = np.argsort(-scores)[:max(1, k)]
        results: List[Dict[str, Any]] = []
        for idx in idxs:
            results.append({
                "text": self.chunks[int(idx)],
                "score": float(scores[int(idx)]),
                "meta": self.metas[int(idx)] if int(idx) < len(self.metas) else {}
            })
        return results


# Store per session; choose a “current” one for modules that don’t pass sid
_SESS_STORE: Dict[str, _SessionIndex] = defaultdict(_SessionIndex)
_CURRENT_SID: Optional[str] = None


# ---------------------- public API ----------------------

def set_active_session(session_id: str) -> None:
    """Mark a session as active; retrieve() will use this if no sid provided."""
    global _CURRENT_SID
    _CURRENT_SID = session_id

def reset_session(session_id: str) -> None:
    """Delete all chunks/index for a session."""
    if session_id in _SESS_STORE:
        del _SESS_STORE[session_id]
    global _CURRENT_SID
    if _CURRENT_SID == session_id:
        _CURRENT_SID = None

def add_document(session_id: str, text: str, meta: Optional[Dict[str, Any]] = None,
                 target_words: int = 120, overlap_words: int = 24) -> int:
    """
    Add ONE long document; we’ll chunk it sentence-aware with overlap and index it.
    Returns number of chunks added.
    """
    meta = meta or {}
    sents = _split_sentences(text or "")
    chunks = _window_by_words(sents, target_words=target_words, overlap_words=overlap_words)
    if not chunks:
        return 0
    idx = _SESS_STORE[session_id]
    idx.add_chunk_batch(chunks, base_meta=meta)
    set_active_session(session_id)
    return len(chunks)

def add_documents(session_id: str, docs: List[Dict[str, Any]],
                  target_words: int = 120, overlap_words: int = 24) -> int:
    """
    Add MANY docs. Each item can be:
      { "text": "...", "meta": {...} }  OR  a plain string (text).
    Returns total chunks added.
    """
    total = 0
    idx = _SESS_STORE[session_id]
    for d in docs or []:
        if isinstance(d, str):
            text = d
            meta = {}
        else:
            text = (d.get("text") or d.get("page_content") or "").strip()
            meta = dict(d.get("meta") or d.get("metadata") or {})
        if not text:
            continue
        sents = _split_sentences(text)
        chunks = _window_by_words(sents, target_words=target_words, overlap_words=overlap_words)
        if not chunks:
            continue
        idx.add_chunk_batch(chunks, base_meta=meta)
        total += len(chunks)
    set_active_session(session_id)
    return total

def retrieve(query: str, k: int = 8, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search the TF-IDF index for the active session (or a provided one) and
    return top-k chunks as dicts: {text, score, meta}.
    """
    sid = session_id or _CURRENT_SID
    if sid is None or sid not in _SESS_STORE:
        return []
    return _SESS_STORE[sid].search(query, k=k)
