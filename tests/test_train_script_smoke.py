from __future__ import annotations

from pathlib import Path

from train import _format_epoch_log, build_run_dir, make_run_name, resolve_device


def test_make_run_name_includes_protocol_env_and_window():
    name = make_run_name(protocol="source_only", env_id=3, window_length=4)

    assert name.startswith("source_only_env03_L4_")


def test_build_run_dir_creates_expected_subdirectories(tmp_path: Path):
    run_dir = build_run_dir(tmp_path, run_name="exp001")

    assert run_dir == tmp_path / "exp001"
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "window_index").is_dir()


def test_resolve_device_auto_returns_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    device = resolve_device("auto")

    assert device.type == "cpu"


def test_format_epoch_log_includes_debug_metrics():
    message = _format_epoch_log(
        epoch=3,
        epochs=100,
        metrics={
            "lr": 1.0e-4,
            "train_loss": 0.5,
            "val_loss": 0.4,
            "mpjpe_norm": 0.3,
            "pck_0.05": 0.1,
            "pck_0.10": 0.2,
            "pck_0.20": 0.3,
            "pck_0.50": 0.4,
            "mean_joint_std": 0.05,
            "min_joint_std": 0.01,
        },
        monitor="val_loss",
        best_metric=0.4,
        stale_epochs=2,
        elapsed_sec=12.34,
    )

    assert "epoch=003/100" in message
    assert "lr=0.000100" in message
    assert "train_loss=0.500000" in message
    assert "val_loss=0.400000" in message
    assert "mpjpe_norm=0.300000" in message
    assert "pck@0.05=0.1000" in message
    assert "joint_std_mean=0.050000" in message
    assert "best_val_loss=0.400000" in message
    assert "stale=2" in message
    assert "time=12.3s" in message
