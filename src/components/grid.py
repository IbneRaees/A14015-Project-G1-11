"""Block-grid helpers shared by the router and the verifier."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def cell_scores(score_map: np.ndarray, grid: int) -> np.ndarray:
    """Mean of a 2-D score map inside each cell of a grid x grid layout."""
    if score_map.ndim != 2:
        raise ValueError(f"score_map must be 2-D, got {score_map.shape}")
    return cv2.resize(np.asarray(score_map, np.float32), (grid, grid), interpolation=cv2.INTER_AREA)


def cell_boxes(cells: np.ndarray, threshold: float, shape: Tuple[int, int],
               max_regions: Optional[int] = None) -> List[List[int]]:
    """[x0, y0, x1, y1] boxes (pixels of an image of `shape`) for cells >= threshold, highest score first."""
    gh, gw = cells.shape
    h, w = shape
    idx = np.argwhere(cells >= threshold)
    order = np.argsort(-cells[idx[:, 0], idx[:, 1]], kind="stable")
    boxes = [[int(round(c * w / gw)), int(round(r * h / gh)),
              int(round((c + 1) * w / gw)), int(round((r + 1) * h / gh))] for r, c in idx[order]]
    return boxes[:max_regions] if max_regions else boxes
