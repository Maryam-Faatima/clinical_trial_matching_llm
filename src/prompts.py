"""Shared prompt/target/parsing logic (used by training, eval and the app)."""
import json
import re
from typing import Dict, List

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}

SYSTEM_PROMPT = (
    "You are a clinical trial eligibility screening assistant for an educational "
    "prototype. Judge ONE criterion against the patient note using only the note. "
    "Never assume a missing fact is true or false. Answer with JSON only."
)


def split_note(note: str) -> Dict[int, str]:
    """Parse 'N. sentence' lines into {id: sentence}; join wrapped lines."""
    sents: Dict[int, str] = {}
    last = None
    for line in note.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\.\s*(.*)$", line)
        if m:
            last = int(m.group(1))
            sents[last] = m.group(2)
        elif last is not None:
            sents[last] += " " + line
    return sents


def format_note(sentences: Dict[int, str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in sorted(sentences.items()))


def build_messages(sentences: Dict[int, str], criterion_type: str, criterion_text: str) -> List[dict]:
    ctype = criterion_type.strip().lower()
    labels = LABELS[ctype]
    user = (
        f"Patient note (sentences are numbered):\n{format_note(sentences)}\n\n"
        f"Criterion type: {ctype}\n"
        f"Criterion: {criterion_text.strip()}\n\n"
        f"Allowed labels: {', '.join(labels)}\n"
        'Return JSON: {"label": <one allowed label>, "evidence": [<sentence ids>]}. '
        "Use an empty list if no sentence supports the decision."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def format_target(label: str, evidence: List[int]) -> str:
    return json.dumps({"label": label, "evidence": sorted(int(e) for e in evidence)})


def parse_output(text: str, criterion_type: str) -> dict:
    """Robustly parse model output. Invalid output => valid=False (scored wrong)."""
    out = {"label": None, "evidence": [], "valid": False}
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
        label = str(obj.get("label", "")).strip().lower()
        ev = obj.get("evidence", [])
        if label in LABELS[criterion_type.strip().lower()] and isinstance(ev, list):
            out["label"] = label
            out["evidence"] = sorted({int(e) for e in ev})
            out["valid"] = True
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return out