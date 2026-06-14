from __future__ import annotations

from pathlib import Path

from train import build_run_dir, make_run_name, resolve_device


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
