from pathlib import Path

import yaml

from train import deep_update, load_config_with_overrides


def test_deep_update_preserves_unspecified_nested_values():
    base = {"train": {"epochs": 100, "batch_size": 32}, "model": {"token_dim": 128}}
    override = {"train": {"batch_size": 64}}

    merged = deep_update(base, override)

    assert merged["train"]["epochs"] == 100
    assert merged["train"]["batch_size"] == 64
    assert merged["model"]["token_dim"] == 128


def test_cli_overrides_win_over_yaml(tmp_path: Path):
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"experiment": {"env_id": 1}, "train": {"batch_size": 32}}),
        encoding="utf-8",
    )

    cfg = load_config_with_overrides(
        cfg_path,
        overrides={"experiment.env_id": 3, "train.batch_size": 16},
    )

    assert cfg["experiment"]["env_id"] == 3
    assert cfg["train"]["batch_size"] == 16
