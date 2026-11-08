"""Prompts and answer parsers shared by E2 (monolithic VLM) and the E3 heavy path.

The same cell prompt is used in both, so the VLM is asked the same question
whether it runs alone (E2) or after the router (E3).
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, Optional

from src.utils.cells import ABSENT, DRY, FLOODED

CELL_PROMPT = (
    "This is a crop from a post-disaster aerial (UAV) photograph. Look only at this crop.\n"
    "Answer two questions.\n"
    '"buildings": "none" if no building is visible; "flooded" if the visible buildings stand in or are '
    'surrounded by floodwater; otherwise "not_flooded".\n'
    '"roads": "none" if no road is visible; "flooded" if the visible road is covered by floodwater; '
    'otherwise "not_flooded".\n'
    'Reply with JSON only, for example {"buildings": "not_flooded", "roads": "none"}.'
)
PROMPT_SHA = hashlib.sha256(CELL_PROMPT.encode()).hexdigest()[:12]

_NEG = re.compile(r"\b(not|non|un)[\s_-]*flooded\b|\bdry\b|\bunflooded\b")
_NONE = re.compile(r"\b(none|no|absent|n/?a|nothing)\b")


def parse_state(text) -> Optional[int]:
    """Map one answer word/phrase to ABSENT / DRY / FLOODED; None if it is not understood."""
    if not isinstance(text, str):
        return None
    t = text.strip().lower().replace('"', "")
    if _NEG.search(t):                      # check negations first: "not flooded" contains "flooded"
        return DRY
    if "flooded" in t or "inundated" in t:
        return FLOODED
    if _NONE.search(t):
        return ABSENT
    return None


def parse_flood_word(text) -> Optional[bool]:
    """True = flooded, False = not flooded, None = unclear."""
    s = parse_state(text)
    return None if s in (None, ABSENT) else s == FLOODED


def parse_cell_answer(text: str) -> Dict[str, Optional[int]]:
    """{'building': state|None, 'road': state|None} from the VLM's reply.

    Tries JSON first, then a regex per key, so slightly malformed replies still count.
    A key that cannot be read comes back as None; callers count those as parse failures.
    """
    out: Dict[str, Optional[int]] = {"building": None, "road": None}
    keys = {"building": ("buildings", "building"), "road": ("roads", "road")}
    obj = None
    m = re.search(r"\{.*?\}", text or "", re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    for task, names in keys.items():
        for name in names:
            if isinstance(obj, dict) and name in obj:
                out[task] = parse_state(str(obj[name]))
                break
            mm = re.search(rf'["\']?{name}["\']?\s*[:=]\s*["\']?([A-Za-z_ \-]+)', text or "", re.I)
            if mm:
                out[task] = parse_state(mm.group(1))
                break
    return out
