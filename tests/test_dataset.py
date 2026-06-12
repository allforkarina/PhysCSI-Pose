import numpy as np
import pytest

from dataset.memmap_dataset import MemmapPoseDataset


def write_fake_cache(root):
    n = 5
    x = np.zeros((n, 12, 10, 114), dtype=np.float32)
    for channel in range(12):
        x[:, channel, :, :] = float(channel)
    np.save(root / "X_all.npy", x)
    np.save(root / "Y_all.npy", np.zeros((n, 17, 2), dtype=np.float32))
    np.save(root / "Conf_all.npy", np.ones((n, 17), dtype=np.float32))
    np.savez(
        root / "meta.npz",
        global_idx=np.arange(n, dtype=np.int64),
        env_id=np.array([1, 1, 1, 2, 2], dtype=np.uint8),
        subject_id=np.array([1, 8, 10, 11, 20], dtype=np.uint8),
        action_id=np.array([1, 1, 1, 1, 1], dtype=np.uint8),
        frame_id=np.array([0, 0, 0, 0, 0], dtype=np.uint16),
        seq_id=np.array([0, 189, 243, 270, 513], dtype=np.uint16),
    )


def test_source_only_env01_train_filters_subjects(tmp_path):
    write_fake_cache(tmp_path)
    ds = MemmapPoseDataset(tmp_path, protocol="source_only", env_id=1, split="train")
    assert len(ds) == 1
    item = ds[0]
    assert item["x"].shape == (12, 10, 114)
    assert item["y"].shape == (17, 2)
    assert item["conf"].shape == (17,)
    assert item["meta"]["subject_id"] == 1


def test_feature_selection_returns_requested_channel_groups(tmp_path):
    write_fake_cache(tmp_path)
    l_norm_ds = MemmapPoseDataset(tmp_path, protocol="source_only", env_id=1, split="train", features=["l_norm"])
    l_norm_item = l_norm_ds[0]
    assert l_norm_item["x"].shape == (3, 10, 114)
    assert np.all(l_norm_item["x"][:, 0, 0] == np.array([0.0, 1.0, 2.0], dtype=np.float32))

    combo_ds = MemmapPoseDataset(tmp_path, protocol="source_only", env_id=1, split="train", features=["f_sub", "c_ant"])
    combo_item = combo_ds[0]
    assert combo_item["x"].shape == (6, 10, 114)
    assert np.all(combo_item["x"][:, 0, 0] == np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32))


def test_source_only_env02_test_filters_subjects(tmp_path):
    write_fake_cache(tmp_path)
    ds = MemmapPoseDataset(tmp_path, protocol="source_only", env_id=2, split="test")
    assert len(ds) == 1
    item = ds[0]
    assert item["meta"]["subject_id"] == 20


def test_invalid_feature_selection_raises(tmp_path):
    write_fake_cache(tmp_path)
    with pytest.raises(ValueError, match="unknown feature"):
        MemmapPoseDataset(tmp_path, protocol="source_only", env_id=1, split="train", features=["raw_amp"])


def test_finetune_protocol_is_not_implemented(tmp_path):
    write_fake_cache(tmp_path)
    with pytest.raises(NotImplementedError, match="finetune"):
        MemmapPoseDataset(tmp_path, protocol="finetune", env_id=2, split="train")
