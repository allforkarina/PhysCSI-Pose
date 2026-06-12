from __future__ import annotations

import numpy as np


def env_id_from_subject(subject_id: int, subjects_per_env: int = 10) -> int:
    if subject_id < 1:
        raise ValueError(f"subject_id must be positive, got {subject_id}")
    return (subject_id - 1) // subjects_per_env + 1


def sequence_id(subject_id: int, action_id: int, num_actions: int = 27) -> int:
    if subject_id < 1:
        raise ValueError(f"subject_id must be positive, got {subject_id}")
    if not 1 <= action_id <= num_actions:
        raise ValueError(f"action_id must be in 1..{num_actions}, got {action_id}")
    return (subject_id - 1) * num_actions + (action_id - 1)


def global_index(
    subject_id: int,
    action_id: int,
    frame_id: int,
    num_actions: int = 27,
    num_frames: int = 297,
) -> int:
    if not 0 <= frame_id < num_frames:
        raise ValueError(f"frame_id must be in 0..{num_frames - 1}, got {frame_id}")
    return sequence_id(subject_id, action_id, num_actions=num_actions) * num_frames + frame_id


def build_meta_arrays(
    num_subjects: int = 40,
    num_actions: int = 27,
    num_frames: int = 297,
    subjects_per_env: int = 10,
) -> dict[str, np.ndarray]:
    total = num_subjects * num_actions * num_frames
    global_idx = np.empty(total, dtype=np.int64)
    env_id = np.empty(total, dtype=np.uint8)
    subject_id_arr = np.empty(total, dtype=np.uint8)
    action_id_arr = np.empty(total, dtype=np.uint8)
    frame_id_arr = np.empty(total, dtype=np.uint16)
    seq_id_arr = np.empty(total, dtype=np.uint16)

    cursor = 0
    for subject_id in range(1, num_subjects + 1):
        env = env_id_from_subject(subject_id, subjects_per_env=subjects_per_env)
        for action_id in range(1, num_actions + 1):
            seq = sequence_id(subject_id, action_id, num_actions=num_actions)
            for frame_id in range(num_frames):
                idx = seq * num_frames + frame_id
                global_idx[cursor] = idx
                env_id[cursor] = env
                subject_id_arr[cursor] = subject_id
                action_id_arr[cursor] = action_id
                frame_id_arr[cursor] = frame_id
                seq_id_arr[cursor] = seq
                cursor += 1

    return {
        "global_idx": global_idx,
        "env_id": env_id,
        "subject_id": subject_id_arr,
        "action_id": action_id_arr,
        "frame_id": frame_id_arr,
        "seq_id": seq_id_arr,
    }
