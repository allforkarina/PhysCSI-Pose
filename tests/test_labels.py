import numpy as np

from dataset.labels import detect_gt_coord_format, normalize_gt_sequence


def test_pixel_coordinates_map_to_target_range():
    gt = np.zeros((297, 17, 3), dtype=np.float32)
    gt[..., 0] = 1920.0
    gt[..., 1] = 1080.0
    gt[..., 2] = 1.0
    y, conf, stats = normalize_gt_sequence(gt)
    assert stats["coord_format"] == "pixel_1920x1080"
    assert np.allclose(y[..., 0], 0.8)
    assert np.allclose(y[..., 1], 0.8)
    assert np.allclose(conf, 1.0)


def test_unit_coordinates_map_to_target_range():
    gt = np.zeros((297, 17, 3), dtype=np.float32)
    gt[..., 0] = 0.5
    gt[..., 1] = 1.0
    gt[..., 2] = 0.7
    y, conf, stats = normalize_gt_sequence(gt)
    assert stats["coord_format"] == "unit_norm_0_1"
    assert np.allclose(y[..., 0], 0.0)
    assert np.allclose(y[..., 1], 0.8)
    assert np.allclose(conf, 0.7)


def test_target_norm_clamps_only():
    gt = np.zeros((297, 17, 3), dtype=np.float32)
    gt[..., 0] = -0.9
    gt[..., 1] = 0.9
    gt[..., 2] = 2.0
    y, conf, stats = normalize_gt_sequence(gt)
    assert stats["coord_format"] == "target_norm_-0.8_0.8"
    assert np.allclose(y[..., 0], -0.8)
    assert np.allclose(y[..., 1], 0.8)
    assert np.allclose(conf, 1.0)


def test_invalid_xy_and_conf_are_zeroed():
    gt = np.ones((297, 17, 3), dtype=np.float32)
    gt[0, 0, 0] = np.nan
    gt[0, 0, 1] = 5.0
    gt[0, 1, :2] = 0.0
    gt[0, 2, 2] = np.inf
    y, conf, stats = normalize_gt_sequence(gt)
    assert np.allclose(y[0, 0], [0.0, 0.0])
    assert conf[0, 0] == 0.0
    assert np.allclose(y[0, 1], [0.0, 0.0])
    assert conf[0, 1] == 0.0
    assert conf[0, 2] == 0.0
    assert stats["invalid_keypoints"] >= 3


def test_detect_gt_coord_format():
    assert detect_gt_coord_format(-1.0, 100.0, 100.0) == "pixel_1920x1080"
    assert detect_gt_coord_format(0.0, 1.0, 1.0) == "unit_norm_0_1"
    assert detect_gt_coord_format(-0.7, 0.7, 0.7) == "target_norm_-0.8_0.8"
