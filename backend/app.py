# backend/app.py
import asyncio
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import uuid
import json
from .bus import subscribe
from inspect import iscoroutinefunction
from pydantic import BaseModel
from typing import Optional
from .registry import SESSIONS, set_model
from .rag.ingest import set_active_session, add_document
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestReq(BaseModel):
    url: Optional[str] = None
    selected_model: str

class QAReq(BaseModel):
    session_id: str
    query: str

class EvalReq(BaseModel):
    selected_model: Optional[str] = None  # in case you switch models later
    limit: Optional[int] = None           # quick run: e.g. 50

@app.post("/eval/tedlium")
async def eval_tedlium(req: EvalReq):
    try:
        sid = str(uuid.uuid4())
        SESSIONS[sid] = {
            "url": None,
            "files": [],
            "model": req.selected_model or "llama-3-8b",
            "transcript": [],   # not used here, but keep structure
            "notes": "",
            "quiz": [],
            "chat_history": [],
            "wer_rows": [],
            "wer_summary": None,
        }
        from .agents.tedlium_eval import run_tedlium_eval
        # run in background
        asyncio.create_task(run_tedlium_eval(sid, limit=req.limit or None, config="release1"))
        return {"session_id": sid}
    except Exception as e:
        print(" TEDLIUM eval failed:", e)
        raise HTTPException(status_code=500, detail=f"Eval failed: {str(e)}")

@app.websocket("/ws/wer/{sid}")
async def ws_wer(websocket: WebSocket, sid: str):
    await websocket.accept()
    if sid not in SESSIONS:
        await websocket.close(code=1008, reason="Session not found")
        return

    # Send cached (summary + first rows) immediately, so the UI isn't empty
    try:
        if SESSIONS[sid].get("wer_summary"):
            await websocket.send_json({"type": "WER_SNAPSHOT", "payload": {
                "summary": SESSIONS[sid]["wer_summary"],
                "rows": SESSIONS[sid]["wer_rows"],
            }})
    except Exception as e:
        print(f" Error sending initial WER snapshot: {e}")

    async def forward(topic: str):
        async for msg in subscribe(sid, topic):
            try:
                await websocket.send_json({"type": topic, "payload": msg})
            except Exception as e:
                print(f" Error sending {topic}: {e}")
                break

    try:
        await asyncio.gather(
            forward("WER_PROGRESS"),
            forward("WER_DONE"),
        )
    except WebSocketDisconnect:
        print(f"📞 WER WebSocket disconnected for session {sid}")


@app.post("/ingest")
async def ingest(req: IngestReq):
    try:
        sid = str(uuid.uuid4())
        SESSIONS[sid] = {
            "url": req.url,
            "files": [],
            "model": req.selected_model,
            "transcript": [],
            "notes": "",
            "quiz": [],
            "chat_history": []
        }
        set_model(sid, req.selected_model)
        set_active_session(sid)

        # Lazy imports to avoid cold-start overhead if unused
        from .agents.transcriber import start_transcription_job
        from .agents import summarizer, quiz

        # Start agents
        asyncio.create_task(summarizer.start_for_session(sid))
        asyncio.create_task(quiz.start_for_session(sid))

        # Kick off transcription
        asyncio.create_task(start_transcription_job(session_id=sid, url=req.url))

        return {"session_id": sid}

    except Exception as e:
        print(" Ingest setup failed:", e)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")

@app.post("/upload")
async def uploadDoc(selected_model: str = Form(...), file: UploadFile = File(...)):
    """
    Upload a single document (e.g., .txt or .md) and index it for RAG within a fresh session.
    """
    try:
        sid = str(uuid.uuid4())
        SESSIONS[sid] = {
            "url": None,
            "files": [file.filename],
            "model": selected_model,
            "transcript": [],
            "notes": "",
            "quiz": [],
            "chat_history": []
        }
        set_model(sid, selected_model)
        set_active_session(sid)

        # Start agents (summary/quiz can still operate over RAG-docs if your agents support it)
        from .agents import summarizer, quiz
        asyncio.create_task(summarizer.start_for_session(sid))
        asyncio.create_task(quiz.start_for_session(sid))

        # Basic text handling; extend with PDF/Doc parsers as needed.
        raw = await file.read()
        text: Optional[str] = None
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = None

        if text and text.strip():
            meta = {"file": file.filename, "source": "upload"}
            # Chunk + index for this session
            add_document(session_id=sid, text=text, meta=meta, target_words=120, overlap_words=24)
        else:
            # No text extracted; leave indexed store empty
            pass

        return {"session_id": sid}
    except Exception as e:
        print(" Upload failed:", e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.websocket("/ws/transcript/{sid}")
@app.websocket("/ws/transcript/{sid}")
async def ws_transcript(websocket: WebSocket, sid: str):
    await websocket.accept()

    if sid not in SESSIONS:
        await websocket.close(code=1008, reason="Session not found")
        return

    # Send cached transcript immediately
    try:
        initial = SESSIONS.get(sid, {}).get("transcript", [])
        if initial:
            await websocket.send_json({"transcript": initial})
    except Exception as e:
        print(f" Error sending initial transcript: {e}")

    async def forward(topic: str):
        async for msg in subscribe(sid, topic):
            try:
                await websocket.send_json(msg)
            except Exception as e:
                print(f" Error sending {topic}: {e}")
                continue

    try:
        await asyncio.gather(
            forward("TRANSCRIPT_READY"),
            forward("TRANSCRIPT_CHUNK"),
        )
    except WebSocketDisconnect:
        print(f" Transcript WebSocket disconnected for session {sid}")

@app.websocket("/ws/summary/{sid}")
async def ws_summary(websocket: WebSocket, sid: str):
    await websocket.accept()

    if sid not in SESSIONS:
        await websocket.close(code=1008, reason="Session not found")
        return

    # Send cached summary immediately
    try:
        initial_notes = SESSIONS.get(sid, {}).get("notes")
        if initial_notes:
            await websocket.send_json({"notes_md": initial_notes})
    except Exception as e:
        print(f" Error sending initial summary: {e}")

    try:
        async for msg in subscribe(sid, "SUMMARY_READY"):
            try:
                await websocket.send_json(msg)
            except Exception as e:
                print(f"❌ Error sending summary: {e}")
                break
    except WebSocketDisconnect:
        print(f" Summary WebSocket disconnected for session {sid}")

@app.websocket("/ws/quiz/{sid}")
async def ws_quiz(websocket: WebSocket, sid: str):
    await websocket.accept()

    if sid not in SESSIONS:
        await websocket.close(code=1008, reason="Session not found")
        return

    # Send cached quiz immediately
    try:
        initial_q = SESSIONS.get(sid, {}).get("quiz")
        if initial_q:
            await websocket.send_json({"questions": initial_q})
    except Exception as e:
        print(f" Error sending initial quiz: {e}")

    try:
        async for msg in subscribe(sid, "QUIZ_READY"):
            try:
                await websocket.send_json(msg)
            except Exception as e:
                print(f" Error sending quiz: {e}")
                continue
    except WebSocketDisconnect:
        print(f" Quiz WebSocket disconnected for session {sid}")

@app.post("/qa")
async def qa(payload: QAReq):
    try:
        sid = payload.session_id
        if sid not in SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")
        from .agents.explainer import answer
        if iscoroutinefunction(answer):
            result = await answer(sid, payload.query)
        else:
            result = answer(sid, payload.query)
        if not result:
            result = "Sorry, I couldn’t generate an answer."
        return {"answer": result}
    except HTTPException:
        raise
    except Exception as e:
        print(f" QA error: {e}")
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "sessions": len(SESSIONS)}
