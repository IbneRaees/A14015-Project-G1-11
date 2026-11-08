import numpy as np

from src.utils.segmentation import instances_to_semantic, mask_to_polygons


def sq(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def test_higher_confidence_wins_overlap():
    polys = [sq(0, 0, 20, 20), sq(10, 10, 30, 30)]
    sem = instances_to_semantic(polys, [3, 4], [0.9, 0.4], (40, 40))
    assert sem[15, 15] == 3 and sem[25, 25] == 4 and sem[5, 5] == 3 and sem[35, 35] == 0
    sem2 = instances_to_semantic(polys, [3, 4], [0.4, 0.9], (40, 40))
    assert sem2[15, 15] == 4


def test_empty_and_degenerate_polygons():
    assert instances_to_semantic([], [], [], (8, 8)).sum() == 0
    assert instances_to_semantic([np.array([[1, 1], [2, 2]])], [1], [0.5], (8, 8)).sum() == 0


def test_mask_to_polygons_normalised_and_filters_specks():
    m = np.zeros((100, 200), np.uint8)
    m[20:60, 40:120] = 1
    m[90, 5] = 1                                   # 1-pixel speck, dropped
    polys = mask_to_polygons(m, min_area=20)
    assert len(polys) == 1
    p = polys[0]
    assert p[:, 0].min() == 40 / 200 and p[:, 0].max() <= 119 / 200 + 1e-9
    assert p.min() >= 0 and p.max() <= 1
