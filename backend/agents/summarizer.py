from ..bus import publish, subscribe
from ..registry import get_client, SESSIONS
from ..rag.ingest import retrieve
from ..my_llm import generate

async def start_for_session(sid: str):
    print(f"🚀 Summarizer agent started for session {sid}")
    client = get_client(sid)

    PROMPT_TEMPLATE = """You are an academic note-maker.
Transcript window:
{trans}

Retrieved references:
{refs}

Write structured notes (Markdown) with headings, bullets, and LaTeX for equations.
"""

    async for m in subscribe(sid, "TRANSCRIPT_READY"):
        print(f"📩 Summarizer received TRANSCRIPT_READY for {sid}, keys={list(m.keys())}")

        # Get transcript from event or session
        transcript_list = m.get("transcript") or SESSIONS[sid].get("transcript", [])
        if transcript_list:
            SESSIONS[sid]["transcript"] = transcript_list

        window = " ".join(x.get("text", "") for x in transcript_list[-80:])
        try:
            refs = retrieve(window, k=3) if window else []
        except FileNotFoundError:
            print(f"⚠️ No RAG store found for session {sid}, skipping references")
            refs = []

        prompt = PROMPT_TEMPLATE.format(
            trans=window or "(no transcript text available)",
            refs="\n\n".join(r["text"][:800] for r in refs) if refs else "(no references)"
        )

        try:
              notes = await generate(client, prompt, max_new_tokens=500)
        except Exception as e:
            print(f"❌ Summarizer generate() failed for {sid}: {e}")
            notes = "_(Summarizer failed to generate notes.)_"

        SESSIONS[sid]["notes"] = notes
        await publish(sid, "SUMMARY_READY", {"notes_md": notes})
        print(f"✅ Summary published for {sid}, length={len(notes)}")
        print(f"--- Summary Preview ---\n{notes[:300]}...\n")
