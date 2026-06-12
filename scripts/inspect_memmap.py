from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)

    x = np.load(root / "X_all.npy", mmap_mode="r")
    y = np.load(root / "Y_all.npy", mmap_mode="r")
    conf = np.load(root / "Conf_all.npy", mmap_mode="r")
    meta = np.load(root / "meta.npz")
    manifest = json.loads((root / "meta_build.json").read_text(encoding="utf-8"))

    print(f"X_all: shape={x.shape}, dtype={x.dtype}")
    print(f"Y_all: shape={y.shape}, dtype={y.dtype}")
    print(f"Conf_all: shape={conf.shape}, dtype={conf.dtype}")
    for key in meta.files:
        print(f"meta[{key}]: shape={meta[key].shape}, dtype={meta[key].dtype}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
