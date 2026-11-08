import json

import numpy as np
import pytest

from src.state import (Disposition, RoadGraphSummary, Route, RoutingDecision, TileInput,
                       TileState, UncertaintyReport, Verdict)


def make_state():
    return TileState(TileInput("t1", 300, 400, "a.jpg", "test"))


def test_step_records_timing_and_summary():
    st = make_state()
    with st.step("route") as rec:
        rec["summary"] = "fast"
    assert st.trajectory[0]["tool"] == "route" and st.trajectory[0]["step"] == 1
    assert st.trajectory[0]["duration_s"] >= 0 and st.trajectory[0]["summary"] == "fast"


def test_failed_step_is_logged_and_reraised():
    st = make_state()
    with pytest.raises(ValueError):
        with st.step("boom"):
            raise ValueError("bad")
    assert "ValueError: bad" in st.trajectory[0]["error"] and st.errors == ["boom: ValueError: bad"]


def test_json_roundtrip_with_arrays_enums_and_nan():
    st = make_state()
    st.arrays["alpha"] = np.ones((10, 4, 4), np.float32)
    st.routing = RoutingDecision(Route.VLM, "entropy", float("nan"), 0.4, [[0, 0, 5, 5]])
    st.uncertainty = UncertaintyReport("vacuity", 0.4, np.float32(0.25), 0.9, 0.5)
    st.verdict = Verdict(Disposition.HUMAN_REVIEW, ["x"])
    d = json.loads(st.to_json())
    assert d["arrays"]["alpha"] == {"shape": [10, 4, 4], "dtype": "float32"}
    assert d["routing"]["route"] == "vlm" and d["routing"]["score"] is None
    assert d["verdict"]["disposition"] == "human_review"
    assert d["uncertainty"]["tile_uncertainty"] == 0.25


def test_import_steps_and_jsonl(tmp_path):
    st = make_state()
    with st.step("a"):
        pass
    st.import_steps([{"step": 1, "tool": "skeletonize", "summary": "s", "duration_s": 0.1}], "geospatial")
    assert [s["tool"] for s in st.trajectory] == ["a", "geospatial.skeletonize"]
    assert st.trajectory[1]["step"] == 2
    p = tmp_path / "logs" / "t.jsonl"
    st.append_jsonl(str(p))
    st.append_jsonl(str(p))
    assert len(p.read_text(encoding="utf-8").splitlines()) == 2


def test_road_summary_from_geospatial_stats():
    stats = dict(n_nodes=5, n_edges=4, n_components=1, n_components_passable=1, n_blocked_edges=0,
                 passable_length_frac=1.0, network_continuity=1.0, spurs_removed=0, total_length_px=9.0)
    assert RoadGraphSummary.from_stats(stats).n_edges == 4
