# backend/rag/ocr_mistral.py
import os, subprocess, tempfile, json
from typing import List, Dict
from dotenv import load_dotenv

# load API key if required by the mistral CLI
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

def run_mistral_ocr(file_path: str, out_dir: str="rag/ocr_out") -> str:
    """
    Run mistral-ocr CLI on a document, save output with the same base name.
    Returns path to the saved OCR file.
    """
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(out_dir, f"{base}.json")

    # Call CLI – adjust flags depending on how your mistral-ocr is installed
    # Example if it supports API key env:
    env = os.environ.copy()
    if API_KEY:
        env["MISTRAL_API_KEY"] = API_KEY

    # Run mistral-ocr and get JSON instead of output.md
    cmd = ["mistral-ocr", "--json", file_path]
    with open(out_path, "w") as f:
        subprocess.run(cmd, env=env, check=True, stdout=f)

    return out_path

def parse_blocks(out_path: str) -> List[Dict]:
    """
    Parse the saved OCR JSON into blocks schema: {page, section, text, bbox}
    """
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    blocks = []
    for blk in data.get("blocks", []):
        blocks.append({
            "page": blk.get("page", 1),
            "section": blk.get("section"),
            "text": blk.get("text", "").strip(),
            "bbox": blk.get("bbox")
        })
    return blocks
