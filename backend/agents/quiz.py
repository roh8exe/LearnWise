# backend/agents/quiz.py
import json
import re
import ast
import hashlib
import string
import random
from typing import Optional, List, Any, Dict

from ..bus import publish, subscribe
from ..registry import get_client, SESSIONS
from ..my_llm import generate

#  helpers: robust JSON extraction 

CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

def _strip_code_fence(s: str) -> str:
    if not s:
        return ""
    m = CODE_FENCE_RE.search(s)
    return m.group(1) if m else s

def _extract_first_json_array(s: str) -> Optional[str]:
    """
    Extract the first top-level JSON array using bracket balancing, ignoring brackets inside strings.
    Works even if there is prose or wrong fences.
    """
    text = _strip_code_fence(s or "")
    depth = 0
    start = -1
    in_str = False
    esc = False

    for i, ch in enumerate(text):
        if ch == '"' and not esc:
            in_str = not in_str
        if ch == "\\" and not esc:
            esc = True
            continue
        esc = False
        if in_str:
            continue
        if ch == '[':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ']':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start : i + 1]
    return None
def contains_proper_names(text: str) -> bool:
    """Check if text contains what looks like proper names"""
    return bool(re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", text))
def _normalize_smart_quotes(s: str) -> str:
    return (s or "").replace("\u201c", '"').replace("\u201d", '"') \
                    .replace("\u2018", "'").replace("\u2019", "'")

def parse_questions(raw: str) -> Any:
    """
    Parse LLM output into Python list/dict.
    Try: (1) balanced array → json, (2) raw json, (3) literal_eval.
    Also accept {"questions":[...]} by unwrapping.
    """
    if not raw:
        raise ValueError("Empty output")

    candidates: List[str] = []
    arr = _extract_first_json_array(raw)
    if arr:
        candidates.append(arr)
    candidates.append(raw.strip())

    for cand in candidates:
        cand = _normalize_smart_quotes(cand).strip()
        cand = _strip_code_fence(cand).strip()
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict) and "questions" in parsed and isinstance(parsed["questions"], list):
                parsed = parsed["questions"]
            return parsed
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(cand)
            if isinstance(parsed, dict) and "questions" in parsed and isinstance(parsed["questions"], list):
                parsed = parsed["questions"]
            return parsed
        except Exception:
            pass

    raise ValueError("Unable to parse JSON array from model output")

# normalization 
def _letter_to_index(s: str) -> Optional[int]:
    if not isinstance(s, str):
        return None
    a = s.strip()
    if len(a) == 1 and a.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return ord(a.upper()) - 65
    m = re.match(r"^\s*([A-Za-z])\s*[\.\)\-]?\s*$", a)
    if m:
        return ord(m.group(1).upper()) - 65
    return None

def normalize_questions(parsed: Any) -> List[Dict[str, Any]]:
    """
    Normalize shapes & values:
      - unify type
      - choices -> options
      - options dict -> list
      - answer index/letter -> option text
      - drop empty/placeholder rows
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(parsed, list):
        parsed = [parsed]

    for q in parsed:
        if not isinstance(q, dict):
            continue
        q = dict(q)

        qtype = (q.get("type") or q.get("question_type") or ("MCQ" if "options" in q or "choices" in q else "Application"))
        qtype = str(qtype).strip()
        q["type"] = "MCQ" if qtype.upper().startswith("MCQ") else ("Application" if qtype.lower().startswith("app") else qtype)

        if "choices" in q and "options" not in q:
            q["options"] = q.pop("choices")

        opts = q.get("options")
        if isinstance(opts, dict):
            keys = sorted(opts.keys())
            opts = [str(opts[k]).strip() for k in keys]
            q["options"] = opts
        elif isinstance(opts, list):
            q["options"] = [str(x).strip() for x in opts if x is not None and str(x).strip() != ""]

        question_txt = (q.get("question") or "").strip()
        if not question_txt or question_txt == "...":
            continue
        if isinstance(q.get("options"), list) and q["options"] and all(o in ("A","B","C","D") for o in q["options"]):
            continue

        ans = q.get("answer")
        opts = q.get("options") if isinstance(q.get("options"), list) else None
        if ans is not None and opts:
            if isinstance(ans, int) and 0 <= ans < len(opts):
                q["answer"] = opts[ans]
            elif isinstance(ans, str):
                a = ans.strip()
                idx = _letter_to_index(a)
                if idx is not None and 0 <= idx < len(opts):
                    q["answer"] = opts[idx]
                else:
                    for opt in opts:
                        if opt.strip().lower() == a.lower():
                            q["answer"] = opt
                            break

        out.append(q)

    # keep at most 2
    return out[:2]
def _force_two_questions(norm: List[Dict[str, Any]], notes_md: str) -> List[Dict[str, Any]]:
    """Ensure we have exactly one MCQ and one Application; fill gaps via fallback."""
    norm = [q for q in (norm or []) if isinstance(q, dict)]

    mcq = next((q for q in norm if str(q.get("type","")).upper().startswith("MCQ")), None)
    app = next((q for q in norm if str(q.get("type","")).lower().startswith("app")), None)

    def valid_mcq(q):
        return (
            q and isinstance(q.get("options"), list) and len(q["options"]) == 4
            and isinstance(q.get("answer"), str) and q["answer"] in q["options"]
        )

    if not valid_mcq(mcq) or not app:
        fb = build_rule_based_fallback(notes_md or "")
        if not valid_mcq(mcq):
            mcq = next((q for q in fb["questions"] if q.get("type") == "MCQ"), mcq)
        if not app:
            app = next((q for q in fb["questions"] if q.get("type") == "Application"), app)

    final = []
    if mcq: final.append(mcq)
    if app: final.append(app)
    return final[:2]

#  quality gate & refinement

_STOPWORDS = set("""
the a an and or of to in on for with from by as is are was were be being been at it this that these those into over under
about after against all am any as at because but by can cannot could did do does doing done down during each few for from
further had has have having he her here hers herself him himself his how i if in into is it its itself me more most my
myself no nor not now of off on once only or other ought our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through to too under until up very was we were what
when where which while who whom why will with would you your yours yourself yourselves
""".split())

# Extra stopwords tuned for lecture notes
_STOPWORDS |= {
    "note", "notes", "video", "lecture", "slide", "slides", "topic", "example",
    "method", "approach", "key", "idea", "primary", "concept", "definition",
    "question", "answer", "option", "choice", "select", "choose", "based",
    "according", "per", "refer", "discuss", "state", "content", "material"
}

PLACEHOLDER_BAN = [
    "aligns best", "apply a key idea", "apply the primary method", 
    "based on the notes", "which statement is correct", "no examples",
    "option a", "option b", "option c", "option d", "generic placeholder",
    "select the correct", "choose the best", "what is the main", 
    "according to the text", "per the material", "as discussed in",
    "refer to the", "based on this passage", "the notes state",
    "this is a placeholder", "example question", "sample question",
    "no examples or explanations", "random unrelated facts"
] + [f"option {chr(i)}" for i in range(65, 69)]  # A, B, C, D

def _clean_text(s: str) -> str:
    return (s or "").replace("\xa0", " ").strip()

def split_sentences(md: str) -> List[str]:
    """
    Split markdown 'notes' into sentence-like units.
    Keeps bullets and numbered items as separate candidates.
    """
    md = _clean_text(md)
    lines = [l.strip() for l in md.splitlines() if l.strip()]
    segs: List[str] = []
    for ln in lines:
        if re.match(r"^(\d+[\.\)]\s+|[-*+]\s+|#{1,6}\s+)", ln):
            segs.append(ln)
        else:
            # Better sentence splitting that handles academic content
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", ln)
            for part in parts:
                if part.strip() and len(part.strip()) > 10:  # Filter out very short fragments
                    segs.append(part.strip())
    return segs

def extract_keywords(text: str, top_k: int = 20) -> List[str]:
    """Better keyword extraction focusing on meaningful terms"""
    # Remove common patterns that aren't real keywords
    text = re.sub(r"\$\d+[\d,.]*\b", "", text)  # Remove dollar amounts
    text = re.sub(r"\b\d+[\d,.]*\b", "", text)  # Remove plain numbers
    
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text)
    freq: Dict[str, int] = {}
    
    for w in words:
        lw = w.lower()
        if lw in _STOPWORDS or len(lw) < 4:
            continue
        # Weight proper nouns and capitalized terms higher
        if w[0].isupper():
            freq[lw] = freq.get(lw, 0) + 3
        else:
            freq[lw] = freq.get(lw, 0) + 1
    
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in items[:top_k]]

def extract_real_formulas(text: str) -> List[str]:
    """Extract only meaningful formulas/code, not random dollar amounts"""
    # Mathematical formulas (LaTeX)
    latex_formulas = re.findall(r"\$\$([^$]{5,100})\$\$", text)
    # Code blocks or inline code
    code_blocks = re.findall(r"`([^`]{5,100})`", text)
    # Actual mathematical expressions (with operators)
    math_expressions = re.findall(r"\b([A-Za-z0-9]+\s*[=+\-*/^]\s*[A-Za-z0-9]+)", text)
    
    return latex_formulas + code_blocks + math_expressions

def extract_focus(notes_md: str, top_k: int = 10) -> Dict[str, List[str]]:
    """
    Pulls focus anchors from notes: headings, code/formulas, capitalized terms, top keywords.
    """
    notes_md = _clean_text(notes_md)
    headings = re.findall(r"^#{1,6}\s+(.+)$", notes_md, flags=re.MULTILINE)
    codebits = extract_real_formulas(notes_md)
    caps = re.findall(r"\b([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,3})\b", notes_md)
    kws = extract_keywords(notes_md, top_k=30)

    def dedup_take(xs, n):
        seen = set()
        out = []
        for x in xs:
            k = x.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(x.strip())
            if len(out) >= n:
                break
        return out

    return {
        "headings": dedup_take(headings, 5),
        "symbols": dedup_take(codebits, 5),
        "caps": dedup_take(caps, 5),
        "keywords": dedup_take(kws, 10),
    }

def _has_placeholder(q: Dict[str, Any]) -> bool:
    def contains_placeholder(s: str) -> bool:
        if not s: return False
        s_lower = s.lower()
        return any(pattern in s_lower for pattern in PLACEHOLDER_BAN)
    
    if contains_placeholder(q.get("question", "")):
        return True
    if contains_placeholder(q.get("rationale", "")):
        return True
        
    opts = q.get("options") or []
    if isinstance(opts, list):
        # Check for single-letter options or placeholder patterns
        if all(len(o.strip()) == 1 and o.strip().upper() in "ABCD" for o in opts):
            return True
        if any(contains_placeholder(o) for o in opts):
            return True
            
    return False

def _is_grounded(q: Dict[str, Any], kws: List[str]) -> bool:
    if not kws:
        return False  # Changed from True to False - if no keywords, it's not grounded!
    
    def count_hits(s: str) -> int:
        if not s: return 0
        s_lower = s.lower()
        return sum(1 for kw in kws if kw.lower() in s_lower)
    
    question_hits = count_hits(q.get("question", ""))
    answer_hits = count_hits(q.get("answer", ""))
    
    if q.get("type") == "MCQ":
        opts = q.get("options") or []
        option_hits = [count_hits(o) for o in opts]
        rationale_hits = count_hits(q.get("rationale", ""))
        
        # Require at least 1 keyword in question, 2 options with keywords, and answer grounded
        return (question_hits >= 1 and 
                sum(h > 0 for h in option_hits) >= 2 and 
                answer_hits >= 1 and
                (rationale_hits + answer_hits) >= 2)
    else:
        # Application questions need stronger grounding
        return question_hits >= 2 and answer_hits >= 2

def needs_refine(qs: List[Dict[str, Any]], kws: List[str]) -> bool:
    if len(qs) != 2:
        print("❌ Refine needed: Wrong number of questions")
        return True
        
    types = [q.get("type") for q in qs]
    if "MCQ" not in types or "Application" not in types:
        print(f"❌ Refine needed: Missing question types. Got: {types}")
        return True
        
    for i, q in enumerate(qs):
        if _has_placeholder(q):
            print(f"❌ Refine needed: Question {i+1} has placeholder")
            return True
        if not _is_grounded(q, kws):
            print(f"❌ Refine needed: Question {i+1} not grounded in keywords: {kws[:5]}")
            return True
            
        if q.get("type") == "MCQ":
            opts = q.get("options")
            if not (isinstance(opts, list) and len(opts) == 4):
                print(f"❌ Refine needed: MCQ options invalid: {opts}")
                return True
                
            ans = q.get("answer")
            if not isinstance(ans, str) or ans not in opts:
                print(f"❌ Refine needed: Answer not in options. Answer: {ans}, Options: {opts}")
                return True
                
    return False

def build_rule_based_fallback(notes_md: str) -> Dict[str, Any]:
    """
    Build two grounded questions from the notes when the model fails.
    Completely rewritten to create meaningful, context-aware questions.
    """
    notes_md = notes_md or ""
    sentences = split_sentences(notes_md)
    kws = extract_keywords(notes_md, top_k=15)
    
    # Filter out very short, header, or problematic sentences
    content_sentences = [
        s for s in sentences 
        if (len(s) > 25 and len(s) < 200 and 
            not re.match(r"^#", s) and 
            not any(ban in s.lower() for ban in PLACEHOLDER_BAN) and
            not re.search(r"\$\d+", s) and  # Exclude dollar amounts
            not re.search(r"^[A-Z][a-z]*\s+[A-Z][a-z]*\s+[A-Z][a-z]*", s)  # Exclude names
        )
    ]
    
    if not content_sentences:
        # Ultimate fallback if no good sentences found
        return {
            "questions": [
                {
                    "type": "MCQ",
                    "question": "What appears to be the main topic or concept discussed in these notes?",
                    "options": [
                        "The notes discuss a specific case study or example",
                        "The content focuses on theoretical concepts without practical application",
                        "A business or technical scenario is being analyzed",
                        "The material covers historical context or background information"
                    ],
                    "answer": "The notes discuss a specific case study or example",
                    "rationale": "Based on the content structure, this appears to be analyzing a specific scenario or case study."
                },
                {
                    "type": "Application",
                    "question": "How would you summarize the key takeaways or main points from this content?",
                    "answer": "Identify the core concepts, explain their significance, and describe how they might be applied in a relevant context."
                }
            ]
        }
    
    # --- Find the BEST sentence for MCQ ---
    # Prioritize sentences that contain keywords and look like factual statements
    best_sentences = []
    for s in content_sentences:
        score = 0
        # Score based on keyword presence
        if any(kw in s.lower() for kw in kws[:5]):
            score += 3
        # Score based on being a factual statement (not question, not command)
        if not s.endswith('?') and not s.startswith(('How', 'Why', 'What', 'When', 'Where')):
            score += 2
        # Penalize sentences that are too specific or contain names
        if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", s):  # Likely contains names
            score -= 2
        if len(s) > 100:  # Too long
            score -= 1
            
        if score > 0:
            best_sentences.append((s, score))
    
    # Sort by score and take the best
    best_sentences.sort(key=lambda x: x[1], reverse=True)
    correct_answer = best_sentences[0][0] if best_sentences else content_sentences[0]
    
    # --- Create meaningful distractors ---
    distractors = []
    
    # Strategy 1: Find contrasting sentences from the notes
    other_good_sentences = [s for s, score in best_sentences[1:4] if s != correct_answer]
    distractors.extend(other_good_sentences[:2])
    
    # Strategy 2: Create semantic opposites
    opposite_pairs = [
        (r"\bis\b", "is not"), (r"\bdoes\b", "does not"), (r"\bcan\b", "cannot"),
        (r"\bwill\b", "will not"), (r"\bshould\b", "should not"), 
        (r"\bincreases?\b", "decreases"), (r"\bdecreases?\b", "increases"),
        (r"\bhigher\b", "lower"), (r"\blower\b", "higher"),
        (r"\bmore\b", "less"), (r"\bless\b", "more"),
        (r"\bbetter\b", "worse"), (r"\bworse\b", "better")
    ]
    
    for pattern, replacement in opposite_pairs:
        if re.search(pattern, correct_answer, re.IGNORECASE):
            distractor = re.sub(pattern, replacement, correct_answer, flags=re.IGNORECASE)
            if distractor != correct_answer and len(distractor) < 150:
                distractors.append(distractor)
                break
    
    # Strategy 3: General but plausible alternatives
    general_distractors = [
        "This represents an exception rather than the general rule",
        "The opposite perspective is supported by alternative evidence",
        "This conclusion requires additional context to be fully accurate",
        "Modern research suggests a different interpretation"
    ]
    
    # Fill with general distractors if needed
    while len(distractors) < 3:
        for distractor in general_distractors:
            if distractor not in distractors:
                distractors.append(distractor)
                if len(distractors) >= 3:
                    break
    
    options = [correct_answer] + distractors[:3]
    random.shuffle(options)
    
    # --- Create context-aware MCQ question ---
    # Extract the main topic from keywords
    main_topic = " ".join(kws[:2]) if kws else "the main concept"
    mcq_question = f"Which statement most accurately represents the discussion of {main_topic} in these notes?"
    
    # --- Application Question Generation ---
    # Analyze the content type to create appropriate application question
    s = notes_md or ""
    has_technical   = bool(re.search(r"\b(algorithm|method|process|system|model)\b", s, re.I))
    has_business    = bool(re.search(r"\b(business|company|market|product|strategy)\b", s, re.I))
    has_theoretical = bool(re.search(r"\b(concept|theory|principle|framework|model)\b", s, re.I))

    
    if has_technical:
        app_q = "Describe how you would implement or apply the technical approach discussed in these notes to a real-world scenario."
        app_a = "First, identify the key technical components or methods. Then outline a step-by-step implementation plan, considering practical constraints and potential challenges. Finally, explain how you would evaluate the success of this application."
    elif has_business:
        app_q = "How would you apply the business insights or strategies from these notes to a contemporary organizational challenge?"
        app_a = "Identify a relevant business context, analyze how the discussed strategies would apply, develop an implementation plan with specific actions, and describe the expected outcomes and metrics for success."
    elif has_theoretical:
        app_q = "How would you use the theoretical framework or concepts from these notes to analyze a practical problem or case study?"
        app_a = "Select an appropriate case or scenario, apply the key theoretical concepts to analyze it, identify insights or solutions based on this analysis, and explain the limitations or assumptions of this approach."
    else:
        # Generic but better application question
        app_q = "Based on the content, how would you apply the main ideas or insights to address a relevant practical challenge?"
        app_a = "Identify a specific context where these ideas would be applicable, outline a concrete plan for implementation, describe the expected outcomes and benefits, and discuss potential obstacles or limitations."
    
    return {
        "questions": [
            {
                "type": "MCQ",
                "question": mcq_question,
                "options": options,
                "answer": correct_answer,
                "rationale": "This statement directly reflects the specific content and terminology used in the source material."
            },
            {
                "type": "Application",
                "question": app_q,
                "answer": app_a
            }
        ]
    }

async def refine_questions(client, notes: str, qs: List[Dict[str, Any]], kws: List[str]) -> List[Dict[str, Any]]:
    """
    Ask the model to rewrite the SAME two questions so they are grounded in the notes.
    Enforce that MCQ options include keywords from the notes.
    """
    skeleton = json.dumps(qs, ensure_ascii=False)
    focus = extract_focus(notes)
    all_terms = focus["headings"] + focus["caps"] + focus["symbols"] + focus["keywords"]
    unique_terms = list(dict.fromkeys(all_terms))  # Preserve order while deduping
    focus_terms = ", ".join(unique_terms[:8]) or "key terms from the notes"
    kw_hint = ", ".join(kws[:10]) or focus_terms

    REWRITE_PROMPT = string.Template("\n".join([
        "Rewrite and improve these TWO questions so they are SPECIFIC to the notes.",
        "Return ONLY a JSON array (you MAY wrap it in ```json fences).",
        "",
        "Rules:",
        "- Keep exactly TWO items: one MCQ and one Application.",
        "- MCQ: four plausible options; correct answer must exactly match one option string.",
        "- Ground everything in the notes; no placeholders.",
        "- Avoid banned phrases (aligns best, based on the notes, apply the primary method, Option A/B/C/D).",
        "- Use at least TWO of these focus terms: $kw",
        "- Include a brief rationale for the MCQ.",
        "",
        "Original (to rewrite):",
        "$skeleton",
        "",
        "Notes:",
        "$notes",
    ])).substitute(skeleton=skeleton, notes=notes, kw=kw_hint)

    out = await generate(client, REWRITE_PROMPT, max_new_tokens=700)
    if isinstance(out, dict):
        out = out.get("text") or out.get("content") or json.dumps(out)
    parsed = parse_questions(out)
    norm = normalize_questions(parsed)
    return norm

# ---------- main agent ----------

async def start_for_session(sid: str):
    print(f"📝 Quiz agent started for session {sid}")
    client = get_client(sid)

    # Primary prompt: ask directly for JSON array (2 items) with stronger rules
    Q_PROMPT_TPL = string.Template("\n".join([
        "You are a strict tutor. Based ONLY on the notes, create exactly TWO questions:",
        "• One **MCQ** with four plausible options (include the correct answer TEXT and a 1–2 sentence rationale).",
        "• One **Application** (free response) with a concise answer.",
        "",
        "HARD RULES:",
        "- Ground both questions in the notes; USE concrete terms and quantities from the notes.",
        "- Every MCQ option must be specific and unique; avoid trivial negations.",
        "- Avoid banned phrases: 'aligns best with the notes', 'based on the notes', 'apply the primary method', 'Option A/B/C/D'.",
        "- Include at least TWO of these focus terms across the question/options/rationale: $focus",
        "- If the notes include steps, algorithms, or formulas, reflect them in the Application item.",
        "- Output ONLY a JSON array of exactly TWO items. No prose or headings.",
        "- If you include anything outside the array, your answer will be discarded.",
        "",
        "OUTPUT FORMAT (ONLY a JSON array, you MAY wrap it in ```json fences):",
        '[{',
        '  "type": "MCQ",',
        '  "question": string,',
        '  "options": [string, string, string, string],',
        '  "answer": string,',
        '  "rationale": string',
        '},{',
        '  "type": "Application",',
        '  "question": string,',
        '  "answer": string',
        '}]',
        "",
        "Notes:",
        "$notes",
    ]))

    # Repair prompt: convert arbitrary text to our JSON array
    REPAIR_PROMPT_TPL = string.Template("\n".join([
        "Convert the content below into a VALID JSON array of TWO question objects.",
        "Return ONLY the JSON array (you MAY wrap it in ```json fences).",
        "Each item schema:",
        '{"type":"MCQ"|"Application","question":string,"options":string[]?,"answer":string,"rationale":string?}',
        "",
        "Content to convert:",
        "$bad",
    ]))

    async for msg in subscribe(sid, "SUMMARY_READY"):
        print(f"❓ Quiz agent received SUMMARY_READY for {sid}")

        notes_md = (
            msg.get("notes_md")
            or msg.get("notes")
            or msg.get("summary")
            or msg.get("text")
            or ""
        ).strip()

        if not notes_md:
            print("ℹ️ No notes content; skipping quiz generation.")
            continue

        key = hashlib.md5(notes_md.encode("utf-8")).hexdigest()
        if SESSIONS.get(sid, {}).get("_last_quiz_key") == key:
            print("↩️ Same notes as last time; skipping duplicate quiz generation.")
            continue

        brief = notes_md if len(notes_md) <= 2000 else notes_md[:2000] + "..."
        keywords = extract_keywords(brief)
        focus = extract_focus(brief)
        all_terms = focus["headings"] + focus["caps"] + focus["symbols"] + focus["keywords"]
        unique_terms = list(dict.fromkeys(all_terms))
        focus_terms = ", ".join(unique_terms[:8]) or "key terms from the notes"

        try:
            # --- primary generation ---
            prompt = Q_PROMPT_TPL.substitute(notes=brief, focus=focus_terms)
            output = await generate(client, prompt, max_new_tokens=650)
            if isinstance(output, dict):
                output = output.get("text") or output.get("content") or json.dumps(output)
            raw = (output or "").strip()
            print(f"📋 Raw LLM output (first 200): {raw[:200]}...")

            # parse or repair
            try:
                parsed = parse_questions(raw)
            except Exception as e1:
                print(f"🔧 First parse failed ({e1}); attempting JSON repair…")
                repair_prompt = REPAIR_PROMPT_TPL.substitute(bad=raw[:4000])
                repaired = await generate(client, repair_prompt, max_new_tokens=700)
                if isinstance(repaired, dict):
                    repaired = repaired.get("text") or repaired.get("content") or json.dumps(repaired)
                parsed = parse_questions(repaired)

            norm = normalize_questions(parsed)
            norm = _force_two_questions(norm, brief)
            # quality gate: must be grounded + no placeholders, answer must match options
            if needs_refine(norm, keywords):
                print("♻️ Refining low-quality questions to ground in notes…")
                norm = await refine_questions(client, brief, norm, keywords)
                norm = _force_two_questions(norm, brief)
                # final guard: if still weak, try once more
                if needs_refine(norm, keywords):
                    print("♻️ Second refinement…")
                    norm = await refine_questions(client, brief, norm, keywords)

            # if STILL unusable, raise to trigger final fallback
            if needs_refine(norm, keywords):
                raise ValueError("Questions remained low-quality after refinement.")

            # ensure exactly two, MCQ first then Application
            mcq = next((q for q in norm if q.get("type") == "MCQ"), None)
            app = next((q for q in norm if q.get("type") == "Application"), None)
            final = []
            if mcq: final.append(mcq)
            if app: final.append(app)
            final = final[:2]

            payload = {"questions": final}

        except Exception as e:
            print(f"⚠️ Quiz generation/parse failed for {sid}: {e}")
            # grounded rule-based fallback (guarded)
            try:
                payload = build_rule_based_fallback(brief or "")
                if not isinstance(payload, dict) or "questions" not in payload:
                    raise ValueError("Fallback returned invalid payload")
            except Exception as e2:
                print(f"🚑 Fallback builder crashed: {e2}")
                payload = {
                    "questions": [
                        {
                            "type":"MCQ",
                            "question":"Which statement best matches the main idea discussed?",
                            "options":[
                                "The notes analyze a concrete case in detail",
                                "They only list definitions without context",
                                "They focus on unrelated historical trivia",
                                "They present random facts without structure"
                            ],
                            "answer":"The notes analyze a concrete case in detail",
                            "rationale":"Chosen because it fits the structure and emphasis of the material."
                        },
                        {
                            "type":"Application",
                            "question":"Apply the core idea from the notes to a relevant real-world scenario.",
                            "answer":"Outline a concrete context, the steps to apply the idea, and the success criteria."
                        }
                    ]
                }


        # cache + de-dupe key
        try:
            SESSIONS[sid]["quiz"] = payload["questions"]
            SESSIONS[sid]["_last_quiz_key"] = key
        except Exception:
            pass

        await publish(sid, "QUIZ_READY", payload)
        print(f"✅ Quiz published for {sid}, count={len(payload['questions'])}")
        for i, q in enumerate(payload["questions"]):
            print(f"Q{i+1}: {q.get('question', 'No question')}")
            if q.get("options"):
                print(f"   Options: {q['options']}")