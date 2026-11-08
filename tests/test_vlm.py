import numpy as np
import pytest

from src.utils.cells import ABSENT, DRY, FLOODED
from src.vlm.backends import MockVLM
from src.vlm.prompts import CELL_PROMPT, PROMPT_SHA, parse_cell_answer, parse_flood_word, parse_state


@pytest.mark.parametrize("text,expected", [
    ("flooded", FLOODED), ("Flooded.", FLOODED), ("not_flooded", DRY), ("not flooded", DRY),
    ("non-flooded", DRY), ("Not Flooded", DRY), ("dry", DRY), ("unflooded", DRY),
    ("none", ABSENT), ("No", ABSENT), ('"flooded"', FLOODED), ("inundated", FLOODED),
    ("maybe", None), ("", None), (None, None),
])
def test_parse_state(text, expected):
    assert parse_state(text) == expected


def test_parse_flood_word():
    assert parse_flood_word("flooded") is True and parse_flood_word("not flooded") is False
    assert parse_flood_word("none") is None and parse_flood_word("unsure") is None


@pytest.mark.parametrize("text,b,r", [
    ('{"buildings": "flooded", "roads": "none"}', FLOODED, ABSENT),
    ('Sure! {"buildings": "not_flooded", "roads": "flooded"} hope that helps', DRY, FLOODED),
    ('```json\n{"buildings":"none","roads":"not_flooded"}\n```', ABSENT, DRY),
    ("{'buildings': 'flooded', 'roads': 'not_flooded'}", FLOODED, DRY),          # single quotes: regex fallback
    ('buildings: flooded, roads: none', FLOODED, ABSENT),
    ('{"building": "flooded", "road": "none"}', FLOODED, ABSENT),
])
def test_parse_cell_answer(text, b, r):
    out = parse_cell_answer(text)
    assert out == {"building": b, "road": r}


def test_unreadable_answers_come_back_as_none():
    assert parse_cell_answer("I cannot tell") == {"building": None, "road": None}
    assert parse_cell_answer('{"buildings": "flooded"}') == {"building": FLOODED, "road": None}
    assert parse_cell_answer("") == {"building": None, "road": None}


def test_prompt_fingerprint_is_stable_and_mock_records_calls():
    assert len(PROMPT_SHA) == 12 and PROMPT_SHA == PROMPT_SHA.lower()
    vlm = MockVLM(lambda img, p: '{"buildings":"none","roads":"none"}')
    vlm.generate(np.zeros((5, 7, 3), np.uint8), CELL_PROMPT)
    assert vlm.calls == [((5, 7), CELL_PROMPT)]
