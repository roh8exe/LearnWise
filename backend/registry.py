# backend/registry.py
from .my_llm import load_client

SESSIONS = {}

def set_model(session_id: str, model_id: str):
    client = load_client(model_id)
    SESSIONS[session_id]["client"] = client
    SESSIONS[session_id]["model"] = model_id

def get_model(session_id: str):
    return SESSIONS[session_id]["model"]

def get_client(session_id: str):
    return SESSIONS[session_id]["client"]
