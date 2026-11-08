import numpy as np
import pytest
from PIL import Image


def make_floodnet_tree(root, counts, size=(64, 48), start=1000):
    """Tiny fake dataset in the official FloodNet layout. Returns {split: [ids]}."""
    ids, n = {}, start
    for split, c in counts.items():
        img_dir = root / split / f"{split}-org-img"
        lab_dir = root / split / f"{split}-label-img"
        img_dir.mkdir(parents=True)
        lab_dir.mkdir(parents=True)
        ids[split] = []
        for _ in range(c):
            w, h = size
            Image.fromarray(np.full((h, w, 3), 90, np.uint8)).save(img_dir / f"{n}.jpg")
            lab = np.zeros((h, w), np.uint8)
            lab[5:20, 5:30] = 2      # building-non-flooded
            lab[30:36, :] = 4        # road-non-flooded
            lab[30:36, 40:] = 3      # road-flooded
            Image.fromarray(lab).save(lab_dir / f"{n}_lab.png")
            ids[split].append(str(n))
            n += 1
    return ids


@pytest.fixture
def tiny_counts():
    return {"train": 5, "val": 3, "test": 4}
