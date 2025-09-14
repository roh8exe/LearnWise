# backend/agents/explainer.py

from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple, Optional
from ..registry import SESSIONS, get_client
from ..rag.ingest import retrieve
from ..my_llm import generate

# Optional cross-encoder re-ranker
try:
    from sentence_transformers import CrossEncoder
    _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    _RERANKER = None

# Utilities for managing session state

def _ensure_chat_buffer(sid: str):
    if "chat_history" not in SESSIONS[sid]:
        SESSIONS[sid]["chat_history"] = []

def _recent_transcript_window(transcript: List[Dict[str, Any]], max_chunks: int = 50) -> str:
    if not transcript:
        return ""
    window = transcript[-max_chunks:]
    return " ".join(x.get("text", "") for x in window if x.get("text"))

def _group_by_section(transcript: List[Dict[str, Any]]):
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for c in transcript:
        sec = int(c.get("section", 0))
        buckets.setdefault(sec, []).append(c)
    for arr in buckets.values():
        arr.sort(key=lambda x: float(x.get("start", 0.0)))
    return sorted(buckets.items(), key=lambda kv: kv[0])

def _truncate_at_boundary(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    cut = s.rfind(" ", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return s[:cut].rstrip()

def _make_sections_text(transcript: List[Dict[str, Any]], max_sections: int = 6, max_chars_per_section: int = 800) -> str:
    sections = _group_by_section(transcript)
    out_blocks = []
    # most recent sections first
    for sec, items in sections[-max_sections:]:
        text = " ".join(x.get("text", "") for x in items if x.get("text"))
        text = _truncate_at_boundary(text, max_chars_per_section)
        out_blocks.append(f"[Section {sec}] {text}")
    return "\n\n".join(out_blocks)

def _dedupe_by_text(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        t = (r.get("text") or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(r)
    return out

def _normalize_retrieve_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm = []
    for r in rows or []:
        text = r.get("text") or r.get("page_content") or ""
        score = r.get("score", None)
        meta = r.get("meta", r.get("metadata", {}))
        norm.append({"text": text, "score": score, "meta": meta})
    return norm

def _rerank(q: str, candidates: List[Dict[str, Any]], top_k: int = 6) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    if _RERANKER is None:
        def key(r):
            s = r.get("score")
            return (float(s) if s is not None else 0.0, len(r.get("text", "")))
        return sorted(candidates, key=key, reverse=True)[:top_k]
    pairs = [(q, c.get("text", "")) for c in candidates]
    scores = _RERANKER.predict(pairs)
    ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda z: z[0], reverse=True)]
    return ranked[:top_k]

def _format_references(refs: List[Dict[str, Any]], max_chars_each: int = 700, max_total_chars: int = 3000) -> Tuple[str, List[Dict[str, Any]]]:
    total = 0
    lines = []
    out_refs = []
    for i, r in enumerate(refs, start=1):
        t = (r.get("text") or "").strip()
        t = _truncate_at_boundary(t, max_chars_each)
        if not t:
            continue
        if total + len(t) > max_total_chars:
            break
        total += len(t)
        meta = r.get("meta") or {}
        tag = ""
        if meta:
            f = meta.get("file")
            sec = meta.get("section")
            idx = meta.get("i")
            loc = []
            if f: loc.append(f)
            if sec is not None: loc.append(f"sec {sec}")
            if idx is not None: loc.append(f"chunk {idx}")
            if loc:
                tag = f" ({', '.join(loc)})"
        lines.append(f"[Ref {i}]{tag}: {t}")
        out_refs.append({"id": i, "text": t, "meta": meta})
    return "\n\n".join(lines), out_refs

def _chat_turns_to_md(history: List[Dict[str, str]], limit: int = 3) -> str:
    if not history:
        return ""
    hx = history[-limit:]
    rows = []
    for t in hx:
        role = t.get("role")
        text = (t.get("text") or "").strip()
        if not text:
            continue
        if role == "user":
            rows.append(f"**Student:** {text}")
        else:
            rows.append(f"**Tutor:** {text}")
    return "\n\n".join(rows)

def _adaptive_mix(q: str) -> str:
    lower = (q or "").lower()
    local_markers = ("this", "just said", "earlier", "above", "previous", "right now", "last slide")
    return "local" if any(m in lower for m in local_markers) else "global"

#Scalar extractor (direct answer) 

_SCALAR_Q_RE = re.compile(r"\b(what\s+is|what's|how\s+much|how\s+many|what\s+are)\b", re.I)
_NUMERIC_CUES_RE = re.compile(r"(\$|%|\d|billion|million|thousand|bn|mn|tn|k|cagr|yoy|mom|eps|pe)", re.I)

# Terms that imply we want a number in the answer.
_SCALAR_TERMS = [
    "revenue","sales","turnover","income","price","growth","increase","market cap","market capitalization",
    "valuation","profit","loss","margin","rate","percentage","percent","deal","contract","worth","projected",
    "projection","forecast","guidance","cagr","yoy","mom","eps","pe","multiple"
]

# Generic pattern used only as a fallback.
_VALUE_SPAN_RE = re.compile(
    r"""(?ix)
    (?:revenue|sales|turnover|income|makes|earns|price|growth|increase|market\s*cap|
       valuation|profit|loss|margin|rate|percentage|deal|contract|worth|forecast|projected)
    [^$%\d]{0,40}
    (\$?\s?\d[\d,\.]*\s?(?:trillion|billion|million|thousand|tn|bn|mn|m|k)?\s?%?)
    """
)

def _wants_numeric(q: str) -> bool:
    ql = (q or "").lower()
    if _NUMERIC_CUES_RE.search(ql):
        return True
    return any(term in ql for term in _SCALAR_TERMS)

def _classify_scalar_question(q: str) -> bool:
    # Require both a generic question opener and numeric intent.
    return bool(_SCALAR_Q_RE.search(q or "")) and _wants_numeric(q)

def _kw_patterns_for_question(q: str) -> List[re.Pattern]:
    ql = (q or "").lower()
    kws = [kw for kw in _SCALAR_TERMS if kw in ql]
    pats: List[re.Pattern] = []
    for kw in sorted(kws, key=len, reverse=True):
        kw_esc = re.escape(kw)
        pats.append(
            re.compile(
                rf"(?i)\b{kw_esc}\b[^$%\d]{{0,40}}(\$?\s?\d[\d,\.]*\s?(?:trillion|billion|million|thousand|tn|bn|mn|m|k)?\s?%?)"
            )
        )
    return pats

def _mine_scalar_from_text(q: str, texts: List[str]) -> Optional[Tuple[str, int]]:
    if not texts:
        return None
    patterns = _kw_patterns_for_question(q)
    # If question had numeric cues but no explicit keyword, allow generic fallback.
    if not patterns and _NUMERIC_CUES_RE.search((q or "")):
        patterns = [_VALUE_SPAN_RE]

    if not patterns:
        return None

    for idx, t in enumerate(texts):
        if not t:
            continue
        for pat in patterns:
            m = pat.search(t)
            if m:
                val = m.group(1)
                val = re.sub(r"\s+", " ", val).strip()
                if val and any(ch.isdigit() for ch in val):
                    return (val, idx)
    return None

# Main entry 

async def answer(sid: str, q: str) -> str:
    client = get_client(sid)
    _ensure_chat_buffer(sid)

    transcript: List[Dict[str, Any]] = SESSIONS[sid].get("transcript", []) or []

    # Query expansion (kept minimal to avoid drift on scalar Qs)
    expansions = [q]

    # Build transcript context
    recent_ctx = _recent_transcript_window(transcript, max_chunks=50)
    hier_ctx = _make_sections_text(transcript, max_sections=6, max_chars_per_section=800)

    # Retrieval (RAG store) for each expansion and for anchored query
    rag_candidates: List[Dict[str, Any]] = []
    for subq in expansions:
        try:
            rr = _normalize_retrieve_rows(retrieve(subq, k=10, session_id=sid))
            rag_candidates.extend(rr)
        except Exception:
            pass
    if recent_ctx:
        try:
            rr_recent = _normalize_retrieve_rows(retrieve(q + "\n" + recent_ctx[:1500], k=10, session_id=sid))
            rag_candidates.extend(rr_recent)
        except Exception:
            pass

    # Deduplicate + rerank
    rag_candidates = _dedupe_by_text(rag_candidates)
    reranked = _rerank(q, rag_candidates, top_k=8)

    # Adaptive mixing
    mode = _adaptive_mix(q)
    if mode == "local":
        refs_for_prompt = reranked[:4]
        transcript_snippet = "\n".join([recent_ctx[:1800], hier_ctx[:1600]]).strip()
    else:
        refs_for_prompt = reranked[:8]
        transcript_snippet = "\n".join([hier_ctx[:1600], recent_ctx[:800]]).strip()

    # If we have absolutely nothing, bail early—keeps answers precise
    if not refs_for_prompt and not transcript_snippet:
        return 'I don’t have evidence.'

    # References block
    refs_block, refs_index = _format_references(refs_for_prompt, max_chars_each=700, max_total_chars=3000)
    SESSIONS[sid]["last_refs"] = refs_index

    # Fast path: direct scalar extraction
    direct_answer = None
    if _classify_scalar_question(q):
        texts = [r.get("text", "") for r in refs_for_prompt]
        mined = _mine_scalar_from_text(q, texts)
        if mined:
            val, ref_idx = mined
            one_based = ref_idx + 1 if 0 <= ref_idx < len(texts) else 1
            answer_text = f"- {val} [Ref {one_based}]"
            SESSIONS[sid]["chat_history"].append({"role": "user", "text": q})
            SESSIONS[sid]["chat_history"].append({"role": "ai", "text": answer_text})
            return answer_text

    #If no direct hit, go to LLM with strict style

    chat_md = _chat_turns_to_md(SESSIONS[sid].get("chat_history", []), limit=3)

    vague_qs = {"what is video about", "what is this about", "summarize", "overview"}
    is_vague = (q.strip().lower() in vague_qs) or (len(q.strip().split()) <= 5 and not _classify_scalar_question(q))

    if is_vague:
        PROMPT = f"""You are an expert tutor. The student asked a broad question: "{q}".
Summarize the video’s main themes and important details. Be concise and concrete.

Transcript excerpts:
{transcript_snippet or "(no transcript available)"}

Retrieved References:
{refs_block or "(no references retrieved)"}

Write 4–7 bullets. Every non-obvious factual claim must include a [Ref i] citation. Avoid speculation.
"""
    else:
        PROMPT = f"""You are an expert tutor. Answer the student's question using ONLY the materials below.
If the answer is not supported by the materials, reply exactly: "I don't have evidence."

STYLE:
- Start with ONE short line that directly answers the question (no preamble).
- If helpful, add at most 3 compact bullets for context.
- Quote numbers/units exactly as they appear.
- Add [Ref i] citations for every factual line.
- Do NOT invent missing pieces.

Previous chat (last few turns, if any):
{chat_md or "(none)"}

Student Question:
{q}

Transcript (structured excerpts):
{transcript_snippet or "(no transcript available)"}

Retrieved References:
{refs_block or "(no references retrieved)"}

Your answer (short, on-point, with [Ref i] citations):
"""

    try:
        answer_text = await generate(client, PROMPT, max_new_tokens=320)
    except Exception as e:
        answer_text = f"_(Answer generation failed: {e})_"

    # Save to conversation memory
    SESSIONS[sid]["chat_history"].append({"role": "user", "text": q})
    SESSIONS[sid]["chat_history"].append({"role": "ai", "text": answer_text})

    return answer_text
