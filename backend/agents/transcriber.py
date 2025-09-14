# backend/agents/transcriber.py
import os
import uuid
import json
from pathlib import Path
from typing import List
from urllib.parse import urlparse, parse_qs

from fastapi import BackgroundTasks
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
import yt_dlp

from transformers import pipeline
import torch

from .. import bus
from ..bus import publish
from ..rag.ingest import add_documents

# Hugging Face Whisper model
HF_WHISPER_MODEL = "openai/whisper-small"
if torch.cuda.is_available():
    device = 0   # first GPU
    print("✅ Using GPU for Whisper")
else:
    device = -1  # CPU
    print("⚠️ No GPU detected, falling back to CPU")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=HF_WHISPER_MODEL,
    chunk_length_s=30,
    device=device,
    generate_kwargs={"language": "en"}
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
RAG_STORE = Path(__file__).resolve().parent.parent / "rag" / "store"

for d in [AUDIO_DIR, TRANSCRIPT_DIR, RAG_STORE]:
    d.mkdir(parents=True, exist_ok=True)


def _clean_youtube_url(url: str) -> str:
    """Normalize a YouTube URL to only include the video ID."""
    if "youtube.com" in url:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        video_id = qs.get("v", [None])[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url


def _extract_video_id(url: str) -> str:
    """Extract just the YouTube video ID."""
    if "youtube.com/watch?v=" in url:
        return url.split("v=")[-1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return url


def _download_audio_with_yt_dlp(url: str, output_dir: Path) -> str:
    """Download audio from YouTube directly as .m4a (no ffmpeg needed)."""
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "noprogress": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return str(Path(ydl.prepare_filename(info)))


async def _transcribe_local(path: str, session_id: str) -> List[dict]:
    """Transcribe audio using Hugging Face Whisper pipeline and stream chunks."""
    print(f"🎙️ Running Whisper transcription on {path} ...")
    
    # Ensure path is a string
    result = asr_pipeline(str(path), return_timestamps=True)


    transcript = []
    chunks = result.get("chunks", [])
    if not chunks:  # fallback if only "text" available
        seg = {
            "text": result.get("text", ""),
            "start": 0.0,
            "end": 0.0,
            "section": 0,
            "i": 0,
        }
        await publish(session_id, "TRANSCRIPT_CHUNK", seg)
        transcript.append(seg)
        return transcript

    for i, seg in enumerate(chunks):
        start, end = seg.get("timestamp", (0.0, None))
        start = float(start) if start is not None else 0.0
        end = float(end) if end is not None else start
        rec = {
            "i": i,
            "text": seg.get("text", "").strip(),
            "start": start,
            "end": end,
            "section": int(start // 120),
        }
        transcript.append(rec)

        # stream each chunk to frontend
        await publish(session_id, "TRANSCRIPT_CHUNK", rec)


    print(f"✅ Generated {len(transcript)} segments with Whisper")
    return transcript


async def start_transcription_job(file_path: str = None, url: str = None, session_id: str = None):
    transcript = None
    video_id = None

    if url:
        try:
            clean_url = _clean_youtube_url(url)
            video_id = _extract_video_id(clean_url)
            loader = YoutubeLoader.from_youtube_url(
                clean_url,
                add_video_info=True,
                language=["en"],
                translation="en",
                transcript_format=TranscriptFormat.CHUNKS,
                chunk_size_seconds=30,
            )
            docs = loader.load()
            transcript = []
            for i, doc in enumerate(docs):
                meta = doc.metadata or {}
                start = float(meta.get("start_time", 0.0) or 0.0)
                end = float(meta.get("end_time", start) or start)
                rec = {
                    "i": i,
                    "text": doc.page_content.strip(),
                    "start": start,
                    "end": end,
                    "section": int(start // 120),
                }
                transcript.append(rec)
                await publish(session_id, "TRANSCRIPT_CHUNK", rec)
            print(f"✅ Transcript fetched via YoutubeLoader for video {video_id}")
        except Exception as e:
            print(f"⚠️ YoutubeLoader failed: {e}\n⚠️ Falling back to yt-dlp + local Whisper...")
            try:
                local_audio = _download_audio_with_yt_dlp(url, AUDIO_DIR)
                video_id = _extract_video_id(url) or Path(local_audio).stem
                transcript = await _transcribe_local(str(local_audio), session_id)
            except Exception as e:
                print(f"❌ yt-dlp/HF Whisper fallback failed: {e}")
                transcript = []

    elif file_path:
        try:
            transcript = await _transcribe_local(str(file_path), session_id)
            video_id = Path(file_path).stem
            print(f"✅ Transcribed local file {file_path} using HF Whisper")
        except Exception as e:
            print(f"❌ Local HF Whisper transcription failed: {e}")
            return
    else:
        print("❌ No file_path or url provided for transcription.")
        return

    if not transcript or not video_id:
        print("❌ No transcript data was produced, skipping save/publish.")
        return

    # Save transcript
    transcript_data = [
    {
        "i": rec["i"],
        "page": rec["start"],
        "section": rec["section"],
        "text": rec["text"],
    }
    for rec in transcript
    ]

    # Save full transcript for UI
    ui_path = TRANSCRIPT_DIR / f"{video_id}.jsonl"
    rag_path = RAG_STORE / f"{video_id}.jsonl"

    # Save UI copy
    with open(ui_path, "w") as f:
        for rec in transcript_data:
            f.write(json.dumps(rec) + "\n")

    # Save RAG copy
    with open(rag_path, "w") as f:
        for rec in transcript_data:
            f.write(json.dumps(rec) + "\n")

    print(f"💾 Transcript saved to {ui_path} and {rag_path}")

    # Add to RAG index
    # Build docs in the shape ingest.add_documents() expects
    docs = [
        {
            "text": rec["text"],
            "meta": {
                "file": f"{video_id}.jsonl",
                "page": rec["start"],
                "section": rec["section"],
                "i": rec["i"],
            },
        }
        for rec in transcript
    ]

    # Use keyword args so the window sizes are ints (and not accidentally a list)
    add_documents(
        session_id=session_id,
        docs=docs,
        target_words=120,   # tune if you want larger/smaller chunks
        overlap_words=24,
    )

    from ..registry import SESSIONS
    SESSIONS[session_id]["transcript"] = transcript
    print(f"📢 Publishing TRANSCRIPT_READY for {session_id}, transcript length={len(transcript)}")
    await publish(session_id, "TRANSCRIPT_READY", {"transcript": transcript})
