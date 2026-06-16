from __future__ import annotations

from pathlib import Path

from eval import _format_eval_log, build_eval_dir, make_eval_name


def test_make_eval_name_uses_checkpoint_parent_and_split():
    checkpoint = Path("runs/exp001/checkpoints/best.pt")

    name = make_eval_name(checkpoint=checkpoint, split="test")

    assert name.startswith("exp001_test_")


def test_build_eval_dir_creates_directory(tmp_path: Path):
    eval_dir = build_eval_dir(tmp_path, eval_name="exp001_test")

    assert eval_dir == tmp_path / "exp001_test"
    assert eval_dir.is_dir()


def test_format_eval_log_includes_debug_metrics():
    message = _format_eval_log(
        split="test",
        metrics={
            "test_loss": 0.25,
            "mpjpe_norm": 0.3,
            "pck_0.05": 0.1,
            "pck_0.10": 0.2,
            "pck_0.20": 0.3,
            "pck_0.50": 0.4,
            "mean_joint_std": 0.05,
            "min_joint_std": 0.01,
            "gt_mean_joint_std": 0.2,
        },
        elapsed_sec=5.67,
    )

    assert "split=test" in message
    assert "test_loss=0.250000" in message
    assert "mpjpe_norm=0.300000" in message
    assert "pck@0.05=0.1000" in message
    assert "pck@0.50=0.4000" in message
    assert "joint_std_mean=0.050000" in message
    assert "gt_joint_std_mean=0.200000" in message
    assert "time=5.7s" in message
