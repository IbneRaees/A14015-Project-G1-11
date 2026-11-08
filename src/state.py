"""Shared state passed between the router, the agents and the orchestrator.

One `TileState` follows one UAV tile through the pipeline. Large arrays
(label maps, Dirichlet parameters, entropy maps) live in `TileState.arrays`
and are serialised as shape/dtype descriptors only; everything else is plain
data that goes straight into JSON logs for trajectory analysis.

The classes are plain dataclasses, so they work as LangGraph / AutoGen state
without extra adapters.
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class Route(str, Enum):
    FAST = "fast"   # fine-tuned YOLOv8x-Seg head
    VLM = "vlm"     # fine-tuned Qwen2-VL


class Disposition(str, Enum):
    AUTO_REPORT = "auto_report"
    HUMAN_REVIEW = "human_review"


@dataclass
class TileInput:
    tile_id: str
    height: int
    width: int
    image_path: Optional[str] = None
    split: Optional[str] = None


@dataclass
class RoutingDecision:
    route: Route
    mode: str                                   # entropy | always_fast | always_vlm
    score: Optional[float] = None               # tile entropy score that was compared
    threshold: Optional[float] = None
    regions: List[List[int]] = field(default_factory=list)   # [x0, y0, x1, y1] in original px
    reason: str = ""


@dataclass
class VisionOutput:
    route: Route
    model: str
    latency_s: float
    n_instances: int = 0
    notes: str = ""
    vlm_calls: int = 0
    vlm_time_s: float = 0.0


@dataclass
class RoadGraphSummary:
    n_nodes: int
    n_edges: int
    n_components: int
    n_components_passable: int
    n_blocked_edges: int
    passable_length_frac: float
    network_continuity: float

    @classmethod
    def from_stats(cls, stats: dict) -> "RoadGraphSummary":
        """Build from `RoadGraphResult.stats` (geospatial agent)."""
        return cls(**{f.name: stats[f.name] for f in dataclasses.fields(cls)})


@dataclass
class UncertaintyReport:
    measure: str                                # vacuity | entropy | epistemic
    threshold: float
    tile_uncertainty: float
    mean_confidence: float
    relevant_mass: float                        # share of the tile that is building/road content
    flagged_regions: List[List[int]] = field(default_factory=list)
    n_cells_flagged: int = 0
    uncertain_edges: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Verdict:
    disposition: Disposition
    reasons: List[str] = field(default_factory=list)


@dataclass
class TileState:
    tile: TileInput
    routing: Optional[RoutingDecision] = None
    vision: Optional[VisionOutput] = None
    road_graph: Optional[RoadGraphSummary] = None
    uncertainty: Optional[UncertaintyReport] = None
    verdict: Optional[Verdict] = None
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @contextmanager
    def step(self, tool: str):
        """Record one pipeline step (name, duration, summary, error).

            with state.step("route") as rec:
                ...
                rec["summary"] = "fast, score=0.12"
        """
        rec: Dict[str, Any] = {"step": len(self.trajectory) + 1, "tool": tool, "summary": ""}
        t0 = time.perf_counter()
        try:
            yield rec
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            self.errors.append(f"{tool}: {rec['error']}")
            raise
        finally:
            rec["duration_s"] = round(time.perf_counter() - t0, 4)
            self.trajectory.append(rec)

    def import_steps(self, steps: List[Dict[str, Any]], prefix: str) -> None:
        """Append steps logged by a sub-agent (e.g. the geospatial agent), renumbered."""
        for s in steps:
            # nested: already counted inside the parent step's duration
            self.trajectory.append({**s, "step": len(self.trajectory) + 1,
                                    "tool": f"{prefix}.{s['tool']}", "nested": True})

    def to_dict(self) -> dict:
        return _plain(self)

    def to_json(self, path: Optional[str] = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, allow_nan=False)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text

    def append_jsonl(self, path: str) -> None:
        """One compact line per tile, for logs/ trajectory files."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), allow_nan=False) + "\n")


def _plain(obj: Any) -> Any:
    """JSON-safe copy. Arrays become {shape, dtype}; NaN/inf become None."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):
        return _plain(obj.item())
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
