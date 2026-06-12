from __future__ import annotations


VALID_SPLITS = {"train", "val", "test"}


def source_only_subjects(env_id: int, split: str, subjects_per_env: int = 10) -> list[int]:
    if not 1 <= env_id <= 4:
        raise ValueError(f"env_id must be in 1..4, got {env_id}")
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

    start = (env_id - 1) * subjects_per_env + 1
    subjects = list(range(start, start + subjects_per_env))
    if split == "train":
        return subjects[:7]
    if split == "val":
        return subjects[7:9]
    return subjects[9:]
