import numpy as np
import pytest

from src.components.yolo_features import (downsample_labels_stride, fused_channels, resize_for_yolo,
                                          resize_label)


def test_resize_for_yolo_gives_multiples_of_32_and_keeps_long_side():
    for hw in ((3000, 4000), (3072, 4592), (4000, 3000), (480, 640)):
        img = np.zeros((*hw, 3), np.uint8)
        out, (h, w) = resize_for_yolo(img, 1024)
        assert out.shape == (h, w, 3) and h % 32 == 0 and w % 32 == 0
        assert max(h, w) == 1024
        assert abs(h / w - hw[0] / hw[1]) < 0.03          # aspect ratio barely changes


def test_label_resize_never_creates_new_ids():
    lab = np.zeros((300, 400), np.uint8)
    lab[:150] = 3
    lab[150:] = 9
    out = resize_label(lab, (96, 128))
    assert set(np.unique(out)) == {3, 9} and out.shape == (96, 128)


def test_stride_downsample_is_centre_sampled():
    a = np.arange(16).reshape(4, 4)
    assert downsample_labels_stride(a, 1) is a
    assert downsample_labels_stride(a, 2).tolist() == [[5, 7], [13, 15]]


def test_fused_channels():
    assert fused_channels([320, 640, 640], (0, 1)) == 960
    assert fused_channels([320, 640, 640], (2,)) == 640
