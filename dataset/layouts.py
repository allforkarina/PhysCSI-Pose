from __future__ import annotations


RAW_CSI_MAT_LAYOUT = "antenna,subcarrier,time10"
MODEL_INPUT_LAYOUT = "sample,antenna,subcarrier,time64"
POSE_TARGET_LAYOUT = "sample,joint,xy"

RAW_CSI_MAT_SHAPE = (3, 114, 10)
MODEL_INPUT_SHAPE = (3, 114, 64)
POSE_TARGET_SHAPE = (17, 2)
