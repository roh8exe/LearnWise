# backend/agents/tedlium_eval.py
import time
import re
from typing import Optional, Dict, Any
import numpy as np
import torch
from datasets import load_dataset, Audio
from transformers import pipeline

from ..registry import SESSIONS
from ..bus import publish

HF_WHISPER_MODEL = "openai/whisper-small"

# ---------- text normalization ----------
_re_punct = re.compile(r"[^a-z0-9' ]+")
_re_spaces = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = _re_punct.sub(" ", s)
    s = _re_spaces.sub(" ", s).strip()
    return s

# ---------- WER with counts via Levenshtein ----------
def wer_counts(ref: str, hyp: str):
    """Return dict with wer, errors, ref_len, hyp_len (word-level)."""
    r = ref.split()
    h = hyp.split()
    R, H = len(r), len(h)

    # dp[i][j] = min ops to transform r[:i] -> h[:j]
    dp = [[0]*(H+1) for _ in range(R+1)]
    for i in range(R+1):
        dp[i][0] = i
    for j in range(H+1):
        dp[0][j] = j
    for i in range(1, R+1):
        for j in range(1, H+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,        # deletion
                dp[i][j-1] + 1,        # insertion
                dp[i-1][j-1] + cost,   # substitution/match
            )
    errors = dp[R][H]
    return {
        "wer": errors / max(1, R),
        "errors": errors,
        "ref_len": R,
        "hyp_len": H,
    }

def _get_text_key(example: Dict[str, Any]) -> str:
    for k in ("text", "transcription", "normalized_text"):
        if k in example:
            return k
    raise KeyError("No reference text field found in example.")

def _should_score(example: Dict[str, Any]) -> bool:
    """
    Return False for TEDLIUM 'gap' segments that must be ignored in scoring.
    """
    spk = (example.get("speaker") or example.get("speaker_id") or "").strip().lower()
    if spk == "inter_segment_gap":
        return False

    try:
        ref_raw = (example[_get_text_key(example)] or "").strip().lower()
    except KeyError:
        return False  # no usable text field

    if ref_raw == "ignore_time_segment_in_scoring" or ref_raw == "":
        return False

    return True


# ---------- main ----------
async def run_tedlium_eval(session_id: str, limit: Optional[int] = None, config: str = "release1"):
    """
    Evaluate Whisper on LIUM/tedlium (release1) test split, stream per-utterance WER and final summary.

    Publishes:
      - WER_PROGRESS: {i, utt_id, speaker, ref, hyp, wer, errors, ref_len, hyp_len}
      - WER_DONE: {corpus_wer, total_err, total_ref, total_hyp, runtime_sec}
    Stores in SESSIONS[sid]["wer_rows"] and ["wer_summary"].
    """
    SESSIONS[session_id]["wer_rows"] = []
    SESSIONS[session_id]["wer_summary"] = None
    t0 = time.time()

    ds = load_dataset("LIUM/tedlium", config, split="test")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # NEW: drop gap rows that should not be scored
    ds = ds.filter(_should_score)

    n_total = len(ds)  # after filtering
    n = n_total if limit is None else min(limit, n_total)
    print(f"🔍 TEDLIUM {config} test: evaluating {n} utterances (of {n_total} after filtering)")

    # 2) ASR pipeline
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline(
        "automatic-speech-recognition",
        model=HF_WHISPER_MODEL,
        chunk_length_s=30,
        device=device,
        generate_kwargs={"language": "en"},
    )

    # 3) Loop & compute
    total_err = 0
    total_ref = 0
    total_hyp = 0

    for i in range(n):
        ex = ds[i]
        audio = ex["audio"]
        arr = audio["array"]
        sr = audio["sampling_rate"]

        utt_id = ex.get("id") or ex.get("path") or f"utt_{i}"
        speaker = ex.get("speaker_id") or ex.get("speaker") or ""

        # reference and hypothesis
        ref_raw = ex[_get_text_key(ex)] or ""
        arr = np.asarray(arr, dtype=np.float32)
        sr = int(sr)

        audio_input = {"array": arr, "sampling_rate": sr}
        out = asr(audio_input)  # no sampling_rate kwarg here
        hyp_raw = (out.get("text") or "").strip()

        ref = normalize_text(ref_raw)
        hyp = normalize_text(hyp_raw)

        stats = wer_counts(ref, hyp)

        row = {
            "i": i,
            "utt_id": utt_id,
            "speaker": speaker,
            "ref": ref_raw,   # keep originals for display if needed
            "hyp": hyp_raw,
            "wer": stats["wer"],
            "errors": stats["errors"],
            "ref_len": stats["ref_len"],
            "hyp_len": stats["hyp_len"],
        }

        # stream to frontend
        await publish(session_id, "WER_PROGRESS", row)

        # keep cache (most recent 500)
        cache = SESSIONS[session_id]["wer_rows"]
        SESSIONS[session_id]["wer_rows"] = [row] + cache[:499]

        total_err += stats["errors"]
        total_ref += stats["ref_len"]
        total_hyp += stats["hyp_len"]

    summary = {
        "corpus_wer": total_err / max(1, total_ref),
        "total_err": int(total_err),
        "total_ref": int(total_ref),
        "total_hyp": int(total_hyp),
        "runtime_sec": time.time() - t0,
    }

    SESSIONS[session_id]["wer_summary"] = summary
    await publish(session_id, "WER_DONE", summary)
    print(f"✅ TEDLIUM eval done. corpus WER={summary['corpus_wer']:.4f} over {n} utts")
