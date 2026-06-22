from __future__ import annotations


SKELETON_NAME = "human36m17"

H36M17_JOINT_NAMES = (
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

H36M17_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)

H36M17_JOINT_GROUPS = {
    "torso": (0, 7, 8, 9, 10),
    "proximal": (1, 4, 11, 14),
    "middle": (2, 5, 12, 15),
    "distal": (3, 6, 13, 16),
    "wrist": (13, 16),
    "ankle": (3, 6),
}

H36M17_JOINT_TO_INDEX = {name: index for index, name in enumerate(H36M17_JOINT_NAMES)}
