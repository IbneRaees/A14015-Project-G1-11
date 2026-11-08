import numpy as np
import pytest
from PIL import Image

from data import floodnet as fn
from tests.conftest import make_floodnet_tree


def test_official_counts_are_frozen():
    assert fn.EXPECTED_COUNTS == {"train": 1445, "val": 450, "test": 448}


def test_wrong_counts_are_rejected(tmp_path, tiny_counts):
    make_floodnet_tree(tmp_path, tiny_counts)
    with pytest.raises(fn.SplitError, match="expected 1445"):
        fn.FloodNetDataset(tmp_path, "train")          # strict default


def test_loader_reads_pairs(tmp_path, tiny_counts):
    make_floodnet_tree(tmp_path, tiny_counts)
    ds = fn.FloodNetDataset(tmp_path, "test", expected=tiny_counts)
    assert len(ds) == 4 and not ds.strict
    item = ds[0]
    assert item["image"].shape == (48, 64, 3) and item["image"].dtype == np.uint8
    assert item["label"].shape == (48, 64) and set(np.unique(item["label"])) == {0, 2, 3, 4}


def test_missing_label_is_an_error(tmp_path, tiny_counts):
    make_floodnet_tree(tmp_path, tiny_counts)
    next((tmp_path / "val" / "val-label-img").iterdir()).unlink()
    with pytest.raises(fn.SplitError, match="no label"):
        fn.index_all(tmp_path, tiny_counts)


def test_overlapping_ids_are_an_error(tmp_path, tiny_counts):
    ids = make_floodnet_tree(tmp_path, tiny_counts)
    dup = ids["train"][0]
    Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(tmp_path / "test" / "test-org-img" / f"{dup}.jpg")
    Image.fromarray(np.zeros((48, 64), np.uint8)).save(tmp_path / "test" / "test-label-img" / f"{dup}_lab.png")
    with pytest.raises(fn.SplitError, match="appears in both"):
        fn.index_all(tmp_path, {**tiny_counts, "test": 5})


def test_manifest_freezes_split(tmp_path, tiny_counts):
    make_floodnet_tree(tmp_path / "d", tiny_counts)
    mpath = tmp_path / "m.json"
    fn.build_manifest(tmp_path / "d", mpath, expected=tiny_counts)
    idx = fn.index_all(tmp_path / "d", tiny_counts)
    assert fn.verify_manifest(idx, mpath)
    tid = idx["test"][0].image_id
    (tmp_path / "d" / "test" / "test-org-img" / f"{tid}.jpg").rename(tmp_path / "d" / "test" / "test-org-img" / "9999.jpg")
    (tmp_path / "d" / "test" / "test-label-img" / f"{tid}_lab.png").rename(tmp_path / "d" / "test" / "test-label-img" / "9999_lab.png")
    with pytest.raises(fn.SplitError, match="has changed"):
        fn.verify_manifest(fn.index_all(tmp_path / "d", tiny_counts), mpath)
    assert not fn.verify_manifest(idx, tmp_path / "none.json")


def test_rgb_palette_labels_are_refused(tmp_path):
    p = tmp_path / "x_lab.png"
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[..., 0] = 200
    Image.fromarray(rgb).save(p)
    with pytest.raises(ValueError, match="colour-coded"):
        fn.read_label(p)


def test_out_of_range_ids_are_refused(tmp_path):
    p = tmp_path / "y_lab.png"
    Image.fromarray(np.full((4, 4), 12, np.uint8)).save(p)
    with pytest.raises(ValueError, match="class id 12"):
        fn.read_label(p)


def test_task_masks():
    lab = np.array([[0, 1, 2], [3, 4, 5]], np.uint8)
    assert fn.road_mask(lab).sum() == 2
    assert fn.blocked_road_mask(lab).sum() == 1
    assert fn.building_mask(lab).sum() == 2
