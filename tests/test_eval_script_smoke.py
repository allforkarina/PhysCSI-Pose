from __future__ import annotations

from pathlib import Path

from eval import build_eval_dir, make_eval_name


def test_make_eval_name_uses_checkpoint_parent_and_split():
    checkpoint = Path("runs/exp001/checkpoints/best.pt")

    name = make_eval_name(checkpoint=checkpoint, split="test")

    assert name.startswith("exp001_test_")


def test_build_eval_dir_creates_directory(tmp_path: Path):
    eval_dir = build_eval_dir(tmp_path, eval_name="exp001_test")

    assert eval_dir == tmp_path / "exp001_test"
    assert eval_dir.is_dir()
