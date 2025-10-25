# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
常数定义

定义实时姿态预测系统中使用的常数和数据结构。
"""

from dataclasses import dataclass


# EMG信号采样率（Hz）
EMG_SAMPLE_RATE = 2000

# 关节数量
NUM_JOINTS = 20

# 手指名称
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# 关节群组
PD_GROUPS = ["proximal", "mid", "distal"]


@dataclass
class Joint:
    """关节定义"""
    name: str
    index: int
    groups: list[str]


# AA: abduction, adduction (外展/内收)
# FE: flexion, extension (屈曲/伸展)
JOINTS: list[Joint] = [
    Joint("THUMB_CMC_FE", 0, ["thumb", "proximal"]),
    Joint("THUMB_CMC_AA", 1, ["thumb", "proximal"]),
    Joint("THUMB_MCP_FE", 2, ["thumb", "mid"]),
    Joint("THUMB_IP_FE", 3, ["thumb", "distal"]),
    Joint("INDEX_MCP_AA", 4, ["index", "proximal"]),
    Joint("INDEX_MCP_FE", 5, ["index", "proximal"]),
    Joint("INDEX_PIP_FE", 6, ["index", "mid"]),
    Joint("INDEX_DIP_FE", 7, ["index", "distal"]),
    Joint("MIDDLE_MCP_AA", 8, ["middle", "proximal"]),
    Joint("MIDDLE_MCP_FE", 9, ["middle", "proximal"]),
    Joint("MIDDLE_PIP_FE", 10, ["middle", "mid"]),
    Joint("MIDDLE_DIP_FE", 11, ["middle", "distal"]),
    Joint("RING_MCP_AA", 12, ["ring", "proximal"]),
    Joint("RING_MCP_FE", 13, ["ring", "proximal"]),
    Joint("RING_PIP_FE", 14, ["ring", "mid"]),
    Joint("RING_DIP_FE", 15, ["ring", "distal"]),
    Joint("PINKY_MCP_AA", 16, ["pinky", "proximal"]),
    Joint("PINKY_MCP_FE", 17, ["pinky", "proximal"]),
    Joint("PINKY_PIP_FE", 18, ["pinky", "mid"]),
    Joint("PINKY_DIP_FE", 19, ["pinky", "distal"]),
]


@dataclass
class Landmark:
    """地标点定义"""
    name: str
    index: int
    groups: list[str]


LANDMARKS: list[Landmark] = [
    Landmark("THUMB_FINGERTIP", 0, ["thumb", "fingertip"]),
    Landmark("INDEX_FINGER_FINGERTIP", 1, ["index", "fingertip"]),
    Landmark("MIDDLE_FINGER_FINGERTIP", 2, ["middle", "fingertip"]),
    Landmark("RING_FINGER_FINGERTIP", 3, ["ring", "fingertip"]),
    Landmark("PINKY_FINGER_FINGERTIP", 4, ["pinky", "fingertip"]),
    Landmark("WRIST_JOINT", 5, ["wrist"]),
    Landmark("THUMB_INTERMEDIATE_FRAME", 6, ["thumb"]),
    Landmark("THUMB_DISTAL_FRAME", 7, ["thumb"]),
    Landmark("INDEX_PROXIMAL_FRAME", 8, ["index"]),
    Landmark("INDEX_INTERMEDIATE_FRAME", 9, ["index"]),
    Landmark("INDEX_DISTAL_FRAME", 10, ["index"]),
    Landmark("MIDDLE_PROXIMAL_FRAME", 11, ["middle"]),
    Landmark("MIDDLE_INTERMEDIATE_FRAME", 12, ["middle"]),
    Landmark("MIDDLE_DISTAL_FRAME", 13, ["middle"]),
    Landmark("RING_PROXIMAL_FRAME", 14, ["ring"]),
    Landmark("RING_INTERMEDIATE_FRAME", 15, ["ring"]),
    Landmark("RING_DISTAL_FRAME", 16, ["ring"]),
    Landmark("PINKY_PROXIMAL_FRAME", 17, ["pinky"]),
    Landmark("PINKY_INTERMEDIATE_FRAME", 18, ["pinky"]),
    Landmark("PINKY_DISTAL_FRAME", 19, ["pinky"]),
    Landmark("PALM_CENTER", 20, ["palm"]),
]

# 以下地标点不会移动，因为手腕不移动，因此在地标指标中将其屏蔽
NO_MOVEMENT_LANDMARKS = [
    "INDEX_PROXIMAL_FRAME",
    "MIDDLE_PROXIMAL_FRAME",
    "RING_PROXIMAL_FRAME",
    "PINKY_PROXIMAL_FRAME",
]
