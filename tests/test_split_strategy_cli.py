from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval as eval_module
import train
from train import TrainConfig


def test_train_config_defaults_to_subject_split() -> None:
    assert TrainConfig(dataset_root="data").split_strategy == "subject"


@pytest.mark.parametrize("strategy", ("subject", "frame_random"))
def test_train_parser_accepts_split_strategy(monkeypatch, strategy: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--mode",
            "source_only",
            "--dataset-root",
            "data",
            "--split-strategy",
            strategy,
        ],
    )

    assert train.parse_args().split_strategy == strategy


def test_train_parser_rejects_unknown_split_strategy(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--mode",
            "source_only",
            "--dataset-root",
            "data",
            "--split-strategy",
            "unknown",
        ],
    )

    with pytest.raises(SystemExit):
        train.parse_args()


@pytest.mark.parametrize("strategy", ("subject", "frame_random"))
def test_eval_parser_accepts_split_strategy(monkeypatch, strategy: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval.py",
            "--dataset-root",
            "data",
            "--checkpoint",
            "model.pth",
            "--split-strategy",
            strategy,
        ],
    )

    assert eval_module.parse_args().split_strategy == strategy


def test_eval_parser_rejects_unknown_split_strategy(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval.py",
            "--dataset-root",
            "data",
            "--checkpoint",
            "model.pth",
            "--split-strategy",
            "unknown",
        ],
    )

    with pytest.raises(SystemExit):
        eval_module.parse_args()
