import numpy as np
import pytest

from dataset.meta import (
    build_meta_arrays,
    env_id_from_subject,
    global_index,
    sequence_id,
)
from dataset.splits import source_only_subjects


def test_env_sequence_and_global_index_reference_points():
    assert env_id_from_subject(1) == 1
    assert env_id_from_subject(10) == 1
    assert env_id_from_subject(11) == 2
    assert env_id_from_subject(40) == 4

    assert sequence_id(1, 1) == 0
    assert sequence_id(1, 2) == 1
    assert sequence_id(40, 27) == 1079

    assert global_index(1, 1, 0) == 0
    assert global_index(1, 1, 296) == 296
    assert global_index(1, 2, 0) == 297
    assert global_index(40, 27, 296) == 320759


def test_build_meta_arrays_shapes_and_dtypes():
    meta = build_meta_arrays(num_subjects=40, num_actions=27, num_frames=297)
    assert meta["global_idx"].shape == (320760,)
    assert meta["env_id"].dtype == np.uint8
    assert meta["subject_id"].dtype == np.uint8
    assert meta["action_id"].dtype == np.uint8
    assert meta["frame_id"].dtype == np.uint16
    assert meta["seq_id"].dtype == np.uint16
    assert meta["global_idx"][0] == 0
    assert meta["env_id"][0] == 1
    assert meta["subject_id"][320759] == 40
    assert meta["action_id"][320759] == 27
    assert meta["frame_id"][320759] == 296
    assert meta["seq_id"][320759] == 1079


def test_source_only_split_subjects():
    assert source_only_subjects(env_id=1, split="train") == [1, 2, 3, 4, 5, 6, 7]
    assert source_only_subjects(env_id=1, split="val") == [8, 9]
    assert source_only_subjects(env_id=1, split="test") == [10]
    assert source_only_subjects(env_id=2, split="train") == [11, 12, 13, 14, 15, 16, 17]
    assert source_only_subjects(env_id=4, split="test") == [40]


def test_invalid_split_inputs_raise():
    with pytest.raises(ValueError, match="env_id"):
        source_only_subjects(env_id=0, split="train")
    with pytest.raises(ValueError, match="split"):
        source_only_subjects(env_id=1, split="dev")
