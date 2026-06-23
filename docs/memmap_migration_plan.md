# WiFlow × NPY Memmap 数据集后端对接 — 修改计划 (v2)

> **状态**: 待审核  
> **版本**: v2 — 基于用户 8 条修改建议全面修订  
> **创建日期**: 2026-05-11  
> **最后更新**: 2026-05-14

---

## 目录

1. [设计决策总览](#1-设计决策总览)
2. [接口差异分析](#2-接口差异分析)
3. [模型架构变更设计](#3-模型架构变更设计)
4. [数据加载流程设计](#4-数据加载流程设计)
5. [需要修改的核心模块及具体代码位置](#5-需要修改的核心模块及具体代码位置)
6. [分阶段实施步骤](#6-分阶段实施步骤)
7. [兼容性测试标准](#7-兼容性测试标准)
8. [关键代码片段预览](#8-关键代码片段预览)
9. [文件变更清单](#9-文件变更清单)
10. [风险识别与缓解](#10-风险识别与缓解)

---

## 1. 设计决策总览

### 1.1 用户 8 条修改建议 → 设计决策映射

| # | 用户要求 | 设计决策 |
|---|----------|----------|
| 1 | 解码端保留 joint query + FC 回归，不用 heatmap | `WiFlowJointDecoder` / `WiFlowHierarchicalJointDecoder` 保持 FC 回归头；`heatmap_msfn` decoder 保留为 ablation 但不作为默认 |
| 2 | `build_memmap.py` 为参考实现，`memmap_dataset.py` 为标准后端 | `scripts/build_memmap.py` 保留不动；`data/memmap_dataset.py` 作为唯一 Dataset 类 |
| 3 | 输入 `(64, 3, 114)`，仅 amplitude，3 天线 | `input_channels=3`，`num_features=1`，移除 phase 处理链路 |
| 4 | SpatialEncoder 输出 `[B, 128, 29, 16]`，解码端 18 个 joint query | SpatialEncoder 时间轴同步下采样（stride 2×2）；decoder `num_queries=18` |
| 5 | 移除 `action_env`，统一 `frame_random`，删除参数 | 删除 `split_scheme` 参数及所有相关代码 |
| 6 | 取消适配层，模型架构层面直接调整 | 删除 `data/memmap_adapter.py` 计划；collate_fn 仅做简单 stack |
| 7 | dataloader 仅保留 NPY，删除 HDF5/NPZ | 重写 `dataloader.py`，移除 `MMFiPoseDataset` 及所有 HDF5 代码 |
| 8 | 所有模块自动对齐接口 | 逐文件审查并更新形状假设、常量引用 |

### 1.2 架构变更总图

```
旧架构 (HDF5, COCO17, 6ch):
  CSI (3,114,10) × 2 features → [B,6,114,10]
    → SpatialEncoder(stride 1×2, 1×2, 1×1) → [B,128,29,10]
    → AxialEncoder → [B,256,29,10]
    → JointDecoder(17 queries) → [B,17,2]

新架构 (NPY, OpenPose18, 3ch):
  CSI (64,3,114) → permute → [B,3,114,64]
    → SpatialEncoder(stride 2×2, 2×2, 1×1) → [B,128,29,16]
    → AxialEncoder → [B,256,29,16]
    → JointDecoder(18 queries) → [B,18,2]
```

---

## 2. 接口差异分析

### 2.1 数据形状差异

| 维度 | 旧后端 (HDF5) | 新后端 (NPY Memmap) | 处理方式 |
|------|---------------|---------------------|----------|
| CSI 张量形状 | `(3, 114, 10)` | `(64, 3, 114)` | collate_fn 中 `permute(0, 2, 3, 1)` → `(B, 3, 114, 64)` |
| 关键点格式 | COCO17 `(17, 2)` | OpenPose18 `(18, 2)` | 全链路使用 OpenPose18，不做转换 |
| CSI 通道数 | 6（amp 3ch + phase_cos 3ch） | 3（仅 amplitude） | `input_channels=3`, `num_features=1` |
| 相位信息 | 有 | 无 | 移除 `csi_phase` / `csi_phase_cos` 处理 |
| 归一化 | 在线（HDF5 存 raw） | 预归一化（3 种变体可选） | 零开销读取 |
| Split 机制 | `action_env` / `frame_random` | 按 subject 动态划分 | 统一 `frame_random` 语义 |

### 2.2 API 差异

| 接口 | 旧后端 | 新后端 |
|------|--------|--------|
| Dataset 类 | `MMFiPoseDataset` | `MemmapDataset` |
| `__getitem__` 返回 | `{keypoints(17,2), csi_amplitude(3,114,10), csi_phase, csi_phase_cos, action, sample, environment, frame_id}` | `{csi(64,3,114), kpts18(18,2), meta: {env, subject, action, frame_idx}}` |
| DataLoader 工厂 | `create_data_loaders()` | `create_memmap_data_loaders()` |
| 序列支持 | `sequence_length` + `_build_sequence_frame_indices` | 时间维度已从 10 上采样至 64，天然在单样本内，无需序列模式 |

### 2.3 关键点格式差异：COCO17 vs OpenPose18

```
COCO17 (旧):                          OpenPose18 (新):
  0: nose                               0: nose
  1: left_eye                           1: neck          ← 新增
  2: right_eye                          2: right_shoulder
  3: left_ear                           3: right_elbow
  4: right_ear                          4: right_wrist
  5: left_shoulder                      5: left_shoulder
  6: right_shoulder                     6: left_elbow
  7: left_elbow                         7: left_wrist
  8: right_elbow                        8: right_hip
  9: left_wrist                         9: right_knee
 10: right_wrist                       10: right_ankle
 11: left_hip                          11: left_hip
 12: right_hip                         12: left_knee
 13: left_knee                         13: left_ankle
 14: right_knee                        14: right_eye
 15: left_ankle                        15: left_eye
 16: right_ankle                       16: right_ear
                                       17: left_ear
```

**PCK torso 参考点变化**：
- 旧：`RIGHT_SHOULDER_INDEX=6`, `LEFT_HIP_INDEX=11`
- 新：`RIGHT_SHOULDER_INDEX=2`, `LEFT_HIP_INDEX=11`

---

## 3. 模型架构变更设计

### 3.1 SpatialEncoder：时间轴同步下采样

**当前实现** (`AsymmetricResidualDownsampleBlock`):

```python
# resblock1: stride=(1, 2) → 时间不变, 子载波减半
# resblock2: stride=(1, 2) → 时间不变, 子载波减半
# resblock3: stride=(1, 1) → 不变
```

**修改后**:

```python
# resblock1: stride=(2, 2) → 时间减半, 子载波减半
# resblock2: stride=(2, 2) → 时间减半, 子载波减半
# resblock3: stride=(1, 1) → 不变
```

**形状流转**:

```
输入:  [B, 3, 114, 64]
_to_conv_layout:    [B, 3, 64, 114]     (permute 0,1,3,2)
_apply_feature_stems: [B, 32, 64, 114]  (num_features=1, 单 stem)
resblock1(stride 2×2): [B, 64, 32, 57]
resblock2(stride 2×2): [B, 128, 16, 29]
resblock3(stride 1×1): [B, 128, 16, 29]
_to_model_layout:   [B, 128, 29, 16]    (transpose 2,3)
```

**关键修改点**：`AsymmetricResidualDownsampleBlock` 的 `stride` 参数从 `(1, spatial_stride)` 改为 `(spatial_stride, spatial_stride)`，即时间轴和子载波轴使用相同的 stride。由于该类的语义从"仅子载波下采样"变为"时空同步下采样"，类名和文档字符串也需更新。

### 3.2 AxialEncoder：自动适配时间维度

`WiFlowAxialEncoder` 的注意力机制对空间和时间 token 数量是透明的——它通过 `permute` + `reshape` 操作动态处理任意维度。当前实现假设输入 `[B, 128, 29, 10]`，修改后接收 `[B, 128, 29, 16]`，无需代码变更。

**验证**：`_prepare_spatial_attention_input` 中 `temporal` 变量自动取 `x.shape[3]`，`_prepare_temporal_attention_input` 同理。

### 3.3 JointDecoder：18 个 joint query

| 组件 | 旧值 | 新值 |
|------|------|------|
| `num_queries` | `NUM_COCO_KEYPOINTS = 17` | `NUM_OPENPOSE_KEYPOINTS = 18` |
| `joint_queries` | `(17, 256)` | `(18, 256)` |
| `flatten_tokens` 输入 | `[B, 256, 29, 10]` → 290 tokens | `[B, 256, 29, 16]` → 464 tokens |
| `coordinate_head` 输出 | `[B, 17, 2]` | `[B, 18, 2]` |

### 3.4 HierarchicalJointDecoder：stage_indices 重映射

当前 `stage_indices` 基于 COCO17 拓扑：

```python
self.stage_indices = (
    (0, 5, 6, 11, 12),           # torso: nose, l_shoulder, r_shoulder, l_hip, r_hip
    (1, 2, 3, 4, 7, 8, 13, 14),  # limbs: eyes, ears, elbows, knees
    (9, 10, 15, 16),              # extremities: wrists, ankles
)
```

需重映射为 OpenPose18 拓扑：

```python
self.stage_indices = (
    (0, 1, 2, 5, 8, 11),                          # torso+head: nose, neck, shoulders, hips
    (3, 4, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17),  # limbs+face: elbows, wrists, knees, ankles, eyes, ears
)
```

### 3.5 skeleton.py：OpenPose18 骨架

```python
NUM_OPENPOSE_KEYPOINTS = 18

OPENPOSE_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),    # nose - neck
    (1, 2),    # neck - r_shoulder
    (2, 3),    # r_shoulder - r_elbow
    (3, 4),    # r_elbow - r_wrist
    (1, 5),    # neck - l_shoulder
    (5, 6),    # l_shoulder - l_elbow
    (6, 7),    # l_elbow - l_wrist
    (1, 8),    # neck - r_hip
    (8, 9),    # r_hip - r_knee
    (9, 10),   # r_knee - r_ankle
    (1, 11),   # neck - l_hip
    (11, 12),  # l_hip - l_knee
    (12, 13),  # l_knee - l_ankle
    (0, 14),   # nose - r_eye
    (14, 16),  # r_eye - r_ear
    (0, 15),   # nose - l_eye
    (15, 17),  # l_eye - l_ear
    (2, 8),    # r_shoulder - r_hip
    (5, 11),   # l_shoulder - l_hip
)
```

### 3.6 WiFlowModel：简化

| 变更 | 说明 |
|------|------|
| `input_channels` 默认值 `6 → 3` | 仅 amplitude |
| 移除 `sequence_length` 参数 | 时间维度已从 10 上采样至 64，天然在单样本内，无需序列模式 |
| 移除 `WiFlowSpatialTemporalFuser` | 不再需要 |
| 移除 5D 输入分支 (`x.ndim == 5`) | 统一 4D 输入 |
| 错误信息更新 | `"single-frame WiFlowModel expects input shaped [B, C, 114, T]"` |

---

## 4. 数据加载流程设计

### 4.1 整体流程

```
MemmapDataset.__getitem__()
  │ {csi: (64,3,114), kpts18: (18,2), meta: {env, subject, action, frame_idx}}
  ▼
memmap_collate_fn()
  │ csi: stack → permute(0,2,3,1) → (B,3,114,64)
  │ kpts18: stack → (B,18,2)
  │ meta: 展开为独立 list
  ▼
prepare_model_input()  ← 简化：仅取 csi_amplitude
  │ model_input = batch["csi_amplitude"]  (已预归一化)
  │ keypoints = batch["keypoints"]  (OpenPose18)
  ▼
WiFlowModel(input_channels=3)
  │ SpatialEncoder: [B,3,114,64] → [B,128,29,16]
  │ AxialEncoder:   [B,128,29,16] → [B,256,29,16]
  │ JointDecoder:   [B,256,29,16] → [B,18,2]
```

### 4.2 collate_fn 设计

```python
def memmap_collate_fn(batch: list[dict]) -> dict:
    csi = torch.stack([item["csi"] for item in batch])          # (B, 64, 3, 114)
    csi = csi.permute(0, 2, 3, 1).contiguous()                  # (B, 3, 114, 64)
    keypoints = torch.stack([item["kpts18"] for item in batch])  # (B, 18, 2)
    return {
        "csi_amplitude": csi,
        "keypoints": keypoints,
        "action": [item["meta"]["action"] for item in batch],
        "sample": [item["meta"]["subject"] for item in batch],
        "environment": [item["meta"]["env"] for item in batch],
        "frame_idx": [item["meta"]["frame_idx"] for item in batch],
    }
```

### 4.3 dataloader.py 重写范围

**删除**：
- `MMFiPoseDataset` 类（~200 行）
- `SampleSequence` / `FrameRecord` dataclass
- `discover_sample_sequences()` / `build_sample_splits()` / `build_frame_splits()` / `expand_frame_records()`
- 所有 HDF5 读写函数（`_decode_string`, `_read_h5_dataset` 等）
- `_normalize_*` 系列函数
- `_clean_csi_phase` / `_compute_csi_phase_cos`
- `resolve_dataset_root` / `resolve_h5_dataset_path`
- `DEFAULT_SPLIT_SCHEME` / `SPLIT_SCHEMES` / `validate_split_scheme`
- `create_data_loaders()`
- `denormalize_keypoints()`
- `preview_split()` / `main()` CLI

**保留/新增**：
- `from data.memmap_dataset import MemmapDataset`
- `memmap_collate_fn()`
- `create_memmap_data_loader()`
- `create_memmap_data_loaders()`

---

## 5. 需要修改的核心模块及具体代码位置

### 5.1 模型层

#### 5.1.1 `models/skeleton.py` — 骨架拓扑

| 行号 | 修改内容 |
|------|----------|
| L5 | `NUM_COCO_KEYPOINTS = 17` → `NUM_OPENPOSE_KEYPOINTS = 18` |
| L6-L22 | `COCO_BONE_EDGES` → `OPENPOSE_BONE_EDGES`（19 条边，18 个关节） |
| L25-L40 | `build_normalized_adjacency` 默认参数更新 |

#### 5.1.2 `models/wiflow_spatial_encoder.py` — 时空同步下采样

| 行号 | 修改内容 |
|------|----------|
| L7 | 类名 `AsymmetricResidualDownsampleBlock` → `SymmetricResidualDownsampleBlock`（语义变更） |
| L8 | 文档字符串：`"downsamples only the subcarrier axis"` → `"downsamples both time and subcarrier axes"` |
| L10 | `__init__` 参数 `spatial_stride` → `stride` |
| L14-L15 | `main_path` Conv2d stride: `(1, spatial_stride)` → `(stride, stride)` |
| L33-L34 | `shortcut` Conv2d stride: `(1, spatial_stride)` → `(stride, stride)` |
| L52 | `input_channels` 默认值 `6 → 3` |
| L53-L54 | 错误信息更新：`"must be a positive multiple of 3"` |
| L56 | `num_features = input_channels // 3` → 始终为 1（仅 amplitude） |
| L57 | `stem_channels = 32`（不变，单 stem 使用全部 32 通道） |
| L59-L68 | `_split_stem_channels` 调用 → 简化为 `[32]`（单 feature） |
| L69-L78 | `antenna_mixers` → 简化为单个 mixer（`nn.ModuleList` → 直接 `nn.Sequential`） |
| L79-L91 | `feature_stems` → 简化为单个 stem |
| L93-L95 | `resblock1/2/3` 参数 `spatial_stride` → `stride` |
| L96-L107 | `_split_stem_channels` 方法可移除（num_features=1 时无需分配） |
| L113-L120 | `_apply_feature_stems` 简化：移除 for 循环，直接处理单 feature |
| L122-L128 | `forward` 注释更新形状流转 |

#### 5.1.3 `models/wiflow_joint_decoder.py` — 18 个 joint query

| 行号 | 修改内容 |
|------|----------|
| L4 | `from .skeleton import NUM_COCO_KEYPOINTS` → `from .skeleton import NUM_OPENPOSE_KEYPOINTS` |
| L47 | `self.num_queries = NUM_COCO_KEYPOINTS` → `self.num_queries = NUM_OPENPOSE_KEYPOINTS` |
| L43 | 文档字符串：`"COCO17"` → `"OpenPose18"` |
| L73 | `flatten_tokens` 错误信息：`"[B, 256, 29, 10]"` → `"[B, 256, 29, 16]"` |

#### 5.1.4 `models/wiflow_hierarchical_joint_decoder.py` — stage_indices 重映射

| 行号 | 修改内容 |
|------|----------|
| L4 | `from .skeleton import NUM_COCO_KEYPOINTS` → `from .skeleton import NUM_OPENPOSE_KEYPOINTS` |
| L76 | `self.num_queries = NUM_COCO_KEYPOINTS` → `self.num_queries = NUM_OPENPOSE_KEYPOINTS` |
| L80-L84 | `self.stage_indices` 重映射为 OpenPose18 拓扑（见 §3.4） |
| L85-L86 | `self.stage_order` / `self.coco_order` 自动从新 indices 推导，无需手动修改 |
| L108 | `flatten_tokens` 错误信息：`"[B, 256, 29, 10]"` → `"[B, 256, 29, 16]"` |

#### 5.1.5 `models/wiflow_model.py` — 模型入口简化

| 行号 | 修改内容 |
|------|----------|
| L22 | `input_channels` 默认值 `6 → 3` |
| L24 | 移除 `sequence_length` 参数 |
| L30-L31 | 移除 `sequence_length` 赋值 |
| L36 | 移除 `self.sequence_length` |
| L39-L43 | 移除 `self.temporal_fuser` 初始化 |
| L63-L80 | `forward` 方法：移除 5D 输入分支，统一 4D 单帧路径 |
| L64 | 错误信息：`"[B, C, 114, 10]"` → `"[B, C, 114, T]"` |
| L10 | 移除 `from .wiflow_spatial_temporal_fuser import ...` |

#### 5.1.6 `models/__init__.py` — 导出更新

| 行号 | 修改内容 |
|------|----------|
| L1 | `from .skeleton import COCO_BONE_EDGES, NUM_COCO_KEYPOINTS` → `from .skeleton import OPENPOSE_BONE_EDGES, NUM_OPENPOSE_KEYPOINTS` |
| L27-L28 | `__all__` 中 `"COCO_BONE_EDGES"` → `"OPENPOSE_BONE_EDGES"`, `"NUM_COCO_KEYPOINTS"` → `"NUM_OPENPOSE_KEYPOINTS"` |

### 5.2 数据层

#### 5.2.1 `dataloader.py` — 完全重写

**删除全部旧代码**（~600 行），替换为：

```python
from __future__ import annotations

"""NPY memmap-backed dataloader for MM-Fi pose data."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.memmap_dataset import MemmapDataset


def memmap_collate_fn(batch: list[dict]) -> dict:
    """Collate MemmapDataset samples: permute CSI + flatten meta."""
    csi = torch.stack([item["csi"] for item in batch])
    csi = csi.permute(0, 2, 3, 1).contiguous()
    keypoints = torch.stack([item["kpts18"] for item in batch])
    return {
        "csi_amplitude": csi,
        "keypoints": keypoints,
        "action": [item["meta"]["action"] for item in batch],
        "sample": [item["meta"]["subject"] for item in batch],
        "environment": [item["meta"]["env"] for item in batch],
        "frame_idx": [item["meta"]["frame_idx"] for item in batch],
    }


def create_memmap_data_loader(
    data_dir: str | Path,
    split: str,
    batch_size: int,
    normalize: str = "global_minmax",
    train_subjects: tuple[str, ...] | None = None,
    random_val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    shuffle: bool | None = None,
) -> DataLoader:
    dataset = MemmapDataset(
        data_dir=data_dir,
        split=split,
        normalize=normalize,
        train_subjects=train_subjects,
        random_val_ratio=random_val_ratio,
        seed=seed,
        build_targets=False,
    )
    should_shuffle = shuffle if shuffle is not None else split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=memmap_collate_fn,
    )


def create_memmap_data_loaders(
    data_dir: str | Path,
    batch_size: int,
    normalize: str = "global_minmax",
    train_subjects: tuple[str, ...] | None = None,
    random_val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    return {
        split: create_memmap_data_loader(
            data_dir=data_dir,
            split=split,
            batch_size=batch_size,
            normalize=normalize,
            train_subjects=train_subjects,
            random_val_ratio=random_val_ratio,
            seed=seed,
            num_workers=num_workers,
        )
        for split in ("train", "val", "test")
    }
```

#### 5.2.2 `data/memmap_dataset.py` — 微调

| 行号 | 修改内容 |
|------|----------|
| L42 | `build_targets: bool = True` → 默认 `False`（joint decoder 不需要 PCM/PAF） |

### 5.3 训练/评估入口

#### 5.3.1 `train.py` — 大幅简化

| 行号 | 修改内容 |
|------|----------|
| L17 | 导入：`from dataloader import DEFAULT_SPLIT_SCHEME, SPLIT_SCHEMES, create_data_loaders` → `from dataloader import create_memmap_data_loaders` |
| L18 | 导入：`from models import COCO_BONE_EDGES` → `from models import OPENPOSE_BONE_EDGES` |
| L22-L24 | **删除** `DEFAULT_CSI_FEATURES`, `SUPPORTED_CSI_FEATURES` |
| L25-L26 | `PCK_THRESHOLDS` 保留不变 |
| L27 | `RIGHT_SHOULDER_INDEX = 6` → `RIGHT_SHOULDER_INDEX = 2` |
| L28 | `LEFT_HIP_INDEX = 11` → `LEFT_HIP_INDEX = 11`（不变） |
| L31-L55 | `TrainConfig` dataclass：**删除** `split_scheme`, `csi_features`, `sequence_length`；**新增** `memmap_normalize: str = "global_minmax"` |
| L58-L75 | **删除** `parse_csi_features()` |
| L78-L81 | **删除** `csi_feature_string()` |
| L84-L96 | `prepare_model_input()` 简化：移除 `csi_features` 参数，直接取 `batch["csi_amplitude"]` |
| L99-L103 | **删除** `effective_sequence_length()` |
| L107 | `bone_length_loss` 默认 `edges=OPENPOSE_BONE_EDGES` |
| L196-L250 | `run_epoch()`：`prepare_model_input` 调用移除 `csi_features` 参数 |
| `parse_args()` | **删除** `--split-scheme`, `--csi-features`, `--sequence-length`；**新增** `--memmap-normalize` |
| `main()` | DataLoader 创建改为 `create_memmap_data_loaders()`；`WiFlowModel` 构造移除 `sequence_length` |

#### 5.3.2 `eval.py` — 大幅简化

| 行号 | 修改内容 |
|------|----------|
| L13 | 导入：`from dataloader import ... MMFiPoseDataset, denormalize_keypoints` → `from dataloader import create_memmap_data_loader` |
| L14 | 导入：`from models import COCO_BONE_EDGES` → `from models import OPENPOSE_BONE_EDGES` |
| L18-L40 | `load_checkpoint_model()`：移除 `split_scheme` 参数；移除 `csi_features` 提取；`input_channels` 固定为 3；移除 `sequence_length` |
| L43-L62 | `plot_skeleton()`：不变（edges 参数由调用方传入） |
| L195-L230 | `collect_metric_breakdowns()`：`prepare_model_input` 调用移除 `csi_features` |
| L233-L255 | `evaluate_model()`：同上 |
| L257-L298 | `save_visualizations()`：重写以适配 MemmapDataset 字段名；使用 `OPENPOSE_BONE_EDGES` |
| L301-L315 | `parse_args()`：**删除** `--split-scheme` |
| L318-L370 | `main()`：DataLoader 创建改为 `create_memmap_data_loader()` |

### 5.4 辅助模块

#### 5.4.1 `pose_targets.py` — 关键点数量更新

| 行号 | 修改内容 |
|------|----------|
| L4 | `from models.skeleton import COCO_BONE_EDGES, NUM_COCO_KEYPOINTS` → `from models.skeleton import OPENPOSE_BONE_EDGES, NUM_OPENPOSE_KEYPOINTS` |
| L8 | `NUM_COCO_KEYPOINTS` → `NUM_OPENPOSE_KEYPOINTS` |
| L47 | `build_paf_targets` 默认 `edges=OPENPOSE_BONE_EDGES` |
| L97 | `decode_pcm_argmax` 中 `NUM_COCO_KEYPOINTS` → `NUM_OPENPOSE_KEYPOINTS` |

#### 5.4.2 `AGENTS.md` — 文档更新

| 位置 | 修改内容 |
|------|----------|
| 构建命令区 | 新增 `python scripts/build_memmap.py ...` |
| 训练命令区 | 替换所有命令：移除 `--split-scheme`, `--csi-features`, `--sequence-length`；新增 `--memmap-normalize` |
| 评估命令区 | 替换所有命令：移除 `--split-scheme` |
| 项目结构 | 更新 `dataloader.py` 描述 |

### 5.5 测试文件

#### 5.5.1 `tests/conftest.py`

| 修改内容 |
|----------|
| 新增 `memmap_dataset_dir` fixture：合成 100 样本 NPY 数据集（`csi_gminmax.npy` shape `(100, 64, 3, 114)`, `ground_truth.npy` shape `(100, 18, 2)`, `meta.npz`） |
| 移除 HDF5 相关 fixture（如 `temp_h5_dataset`） |

#### 5.5.2 `tests/test_dataloader.py`

| 修改内容 |
|----------|
| 删除 `TestMMFiPoseDataset` 类 |
| 删除 `TestH5Build` 类 |
| 新增 `TestMemmapDataset`：构造、split、`__getitem__` 形状验证 |
| 新增 `TestMemmapCollateFn`：批处理形状验证 |
| 新增 `TestMemmapDataLoader`：端到端集成 |

#### 5.5.3 `tests/test_train.py`

| 修改内容 |
|----------|
| 新增 `test_memmap_training_smoketest`：合成 NPY 数据集 2 epoch 训练，验证 loss 下降 |

#### 5.5.4 `tests/test_wiflow_model.py`

| 修改内容 |
|----------|
| 更新输入形状：`(2, 3, 114, 64)` |
| 更新输出形状断言：`(2, 18, 2)` |
| 移除 5D 序列输入测试 |

#### 5.5.5 `tests/test_wiflow_decoder.py`

| 修改内容 |
|----------|
| 更新 decoder 输入形状：`(2, 256, 29, 16)` |
| 更新输出形状断言：`(2, 18, 2)` |

#### 5.5.6 `tests/test_pose_targets.py`

| 修改内容 |
|----------|
| 关键点形状 `(B, 17, 2)` → `(B, 18, 2)` |
| PCM 通道数 `17` → `18` |

### 5.6 不变文件

```
data/heatmap_gt.py                   # 已使用 OpenPose18，无需修改
data/memmap_dataset.py               # 仅 build_targets 默认值微调
scripts/build_memmap.py              # 参考实现，不动
scripts/build_h5_dataset.py          # 保留作为历史参考
models/wiflow_axial_encoder.py       # 对时间维度透明
models/wiflow_heatmap_decoder.py     # ablation 保留，后续按需更新
models/wiflow_spatial_temporal_fuser.py  # 保留但不再使用
models/wiflow_attention_pooler.py    # legacy
models/wiflow_temporal_encoder.py    # legacy
models/wiflow_skeleton_decoder.py    # legacy
```

---

## 6. 分阶段实施步骤

### 阶段 1：骨架拓扑 + 关键点常量

| 步骤 | 文件 | 操作 | 验证 |
|------|------|------|------|
| 1.1 | `models/skeleton.py` | `NUM_COCO_KEYPOINTS=17` → `NUM_OPENPOSE_KEYPOINTS=18`；`COCO_BONE_EDGES` → `OPENPOSE_BONE_EDGES` | `from models.skeleton import NUM_OPENPOSE_KEYPOINTS; assert NUM_OPENPOSE_KEYPOINTS == 18` |
| 1.2 | `models/__init__.py` | 更新导出符号名 | `from models import OPENPOSE_BONE_EDGES, NUM_OPENPOSE_KEYPOINTS` 不报错 |
| 1.3 | `pose_targets.py` | 更新 import 和常量引用 | `pytest tests/test_pose_targets.py -v` 通过 |

### 阶段 2：SpatialEncoder 时间下采样

| 步骤 | 文件 | 操作 | 验证 |
|------|------|------|------|
| 2.1 | `models/wiflow_spatial_encoder.py` | stride `(1, N)` → `(N, N)`；`input_channels=3`；简化 feature stems | 构造 `WiFlowSpatialEncoder(input_channels=3)`，输入 `(2,3,114,64)` → 输出 `(2,128,29,16)` |
| 2.2 | 同上 | 类名 `AsymmetricResidualDownsampleBlock` → `SymmetricResidualDownsampleBlock` | 全局搜索替换引用 |

### 阶段 3：Decoder 18 queries

| 步骤 | 文件 | 操作 | 验证 |
|------|------|------|------|
| 3.1 | `models/wiflow_joint_decoder.py` | `NUM_COCO_KEYPOINTS` → `NUM_OPENPOSE_KEYPOINTS`；错误信息更新 | 构造 decoder，输入 `(2,256,29,16)` → 输出 `(2,18,2)` |
| 3.2 | `models/wiflow_hierarchical_joint_decoder.py` | 同上 + `stage_indices` 重映射 | 同上形状验证 |

### 阶段 4：WiFlowModel 简化

| 步骤 | 文件 | 操作 | 验证 |
|------|------|------|------|
| 4.1 | `models/wiflow_model.py` | `input_channels=3`；移除 `sequence_length`；移除 5D 分支 | 端到端：`(2,3,114,64)` → `(2,18,2)` |

### 阶段 5：数据层重写

| 步骤 | 文件 | 操作 | 验证 |
|------|------|------|------|
| 5.1 | `dataloader.py` | 完全重写（见 §5.2.1） | `from dataloader import create_memmap_data_loaders` 不报错 |
| 5.2 | `data/memmap_dataset.py` | `build_targets` 默认 `False` | 构造 Dataset 不报错 |

### 阶段 6：训练/评估入口适配

| 步骤 | 文件 | 操作 | 验证 |
|------|------|------|------|
| 6.1 | `train.py` | 删除 `split_scheme`/`csi_features`/`sequence_length`；简化 `prepare_model_input`；更新 torso 索引 | `python train.py --help` 显示新参数 |
| 6.2 | `eval.py` | 同上；重写 `save_visualizations` | `python eval.py --help` 显示新参数 |

### 阶段 7：测试更新

| 步骤 | 操作 | 验证 |
|------|------|------|
| 7.1 | 更新 `conftest.py`：新增 memmap fixture，移除 HDF5 fixture | fixture 可用 |
| 7.2 | 更新 `test_dataloader.py`：MemmapDataset 测试 | `pytest tests/test_dataloader.py -v` 通过 |
| 7.3 | 更新 `test_wiflow_model.py`：新形状 | `pytest tests/test_wiflow_model.py -v` 通过 |
| 7.4 | 更新 `test_wiflow_decoder.py`：18 输出 | `pytest tests/test_wiflow_decoder.py -v` 通过 |
| 7.5 | 更新 `test_pose_targets.py`：18 关键点 | `pytest tests/test_pose_targets.py -v` 通过 |
| 7.6 | 新增 `test_memmap_training_smoketest` | 2 epoch 训练完成，loss 下降 |
| 7.7 | 全量回归 | `pytest -v` 全部通过 |

### 阶段 8：文档 + 清理

| 步骤 | 操作 |
|------|------|
| 8.1 | 更新 `AGENTS.md`：新命令、新架构描述 |
| 8.2 | 全局搜索 `COCO` 关键字，确认无遗漏引用 |
| 8.3 | 全局搜索 `action_env` / `split_scheme` / `csi_features` / `csi_phase`，确认已清理 |
| 8.4 | 真实数据 smoketest：`python train.py --dataset-root <real_npy_dir> --epochs 5 --subset-size 128` |

---

## 7. 兼容性测试标准

### 7.1 单元测试

| 测试项 | 验收标准 |
|--------|----------|
| `MemmapDataset.__init__` | 非法 `split` 抛出 `ValueError`；非法 `normalize` 抛出 `ValueError` |
| `MemmapDataset._build_split` | train/val 无重叠；比例符合 `random_val_ratio`；相同 seed 可复现 |
| `MemmapDataset.__getitem__` | `csi: (64, 3, 114)`, `kpts18: (18, 2)`, `meta` 四字段 |
| `MemmapDataset.__len__` | 返回 `len(self.indices)` |
| `memmap_collate_fn` | `csi_amplitude: (B, 3, 114, 64)`, `keypoints: (B, 18, 2)` |
| `SpatialEncoder` 形状 | `(2, 3, 114, 64)` → `(2, 128, 29, 16)` |
| `AxialEncoder` 形状 | `(2, 128, 29, 16)` → `(2, 256, 29, 16)` |
| `JointDecoder` 形状 | `(2, 256, 29, 16)` → `(2, 18, 2)` |
| `HierarchicalJointDecoder` 形状 | 同上 |
| `WiFlowModel` 端到端 | `(2, 3, 114, 64)` → `(2, 18, 2)`，无 NaN/Inf |
| `build_normalized_adjacency(18, OPENPOSE_BONE_EDGES)` | `(18, 18)` 对称正定 |

### 7.2 集成测试

| 测试项 | 验收标准 |
|--------|----------|
| 单 epoch 训练 smoketest | loss 单调下降，无 CUDA OOM |
| checkpoint 保存/加载 | `train_config` 含 `memmap_normalize`；加载后可继续训练 |
| 评估指标计算 | MPJPE/PCK 数值在合理范围（非零、非 NaN） |
| 可视化输出 | 骨架图使用 OpenPose18 边，无索引越界 |

---

## 8. 关键代码片段预览

### 8.1 `models/skeleton.py` — 新骨架

```python
NUM_OPENPOSE_KEYPOINTS = 18

OPENPOSE_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),    # nose - neck
    (1, 2),    # neck - r_shoulder
    (2, 3),    # r_shoulder - r_elbow
    (3, 4),    # r_elbow - r_wrist
    (1, 5),    # neck - l_shoulder
    (5, 6),    # l_shoulder - l_elbow
    (6, 7),    # l_elbow - l_wrist
    (1, 8),    # neck - r_hip
    (8, 9),    # r_hip - r_knee
    (9, 10),   # r_knee - r_ankle
    (1, 11),   # neck - l_hip
    (11, 12),  # l_hip - l_knee
    (12, 13),  # l_knee - l_ankle
    (0, 14),   # nose - r_eye
    (14, 16),  # r_eye - r_ear
    (0, 15),   # nose - l_eye
    (15, 17),  # l_eye - l_ear
    (2, 8),    # r_shoulder - r_hip
    (5, 11),   # l_shoulder - l_hip
)
```

### 8.2 `models/wiflow_spatial_encoder.py` — 关键 diff

```python
# 修改前:
class AsymmetricResidualDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_stride):
        # stride=(1, spatial_stride) — 时间轴不下采样

# 修改后:
class SymmetricResidualDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        # stride=(stride, stride) — 时空同步下采样

# 修改前:
self.resblock1 = AsymmetricResidualDownsampleBlock(32, 64, spatial_stride=2)
self.resblock2 = AsymmetricResidualDownsampleBlock(64, 128, spatial_stride=2)
self.resblock3 = AsymmetricResidualDownsampleBlock(128, 128, spatial_stride=1)

# 修改后:
self.resblock1 = SymmetricResidualDownsampleBlock(32, 64, stride=2)
self.resblock2 = SymmetricResidualDownsampleBlock(64, 128, stride=2)
self.resblock3 = SymmetricResidualDownsampleBlock(128, 128, stride=1)
```

### 8.3 `train.py` — `prepare_model_input` 简化

```python
# 修改前:
def prepare_model_input(
    batch, device, csi_features=DEFAULT_CSI_FEATURES,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_tensors = [
        torch.as_tensor(batch[name], dtype=torch.float32, device=device)
        for name in csi_features
    ]
    model_input = torch.cat(feature_tensors, dim=...)
    ...

# 修改后:
def prepare_model_input(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_input = torch.as_tensor(
        batch["csi_amplitude"], dtype=torch.float32, device=device
    )
    keypoints = torch.as_tensor(
        batch["keypoints"], dtype=torch.float32, device=device
    )
    return model_input, keypoints
```

### 8.4 `train.py` — `TrainConfig` 变更

```python
# 修改前:
@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    split_scheme: str = DEFAULT_SPLIT_SCHEME      # 删除
    csi_features: tuple[str, ...] = ...            # 删除
    sequence_length: int = 1                       # 删除
    ...

# 修改后:
@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    output_dir: str = "outputs/train"
    memmap_normalize: str = "global_minmax"        # 新增
    axial_mode: str = "spatial_then_temporal"
    decoder_type: str = "joint"
    epochs: int = 50
    batch_size: int = 64
    ...
```

---

## 9. 文件变更清单

### 9.1 新增文件

```
data/__init__.py                   ✅ 已创建
data/heatmap_gt.py                 ✅ 已复制
data/memmap_dataset.py             ✅ 已复制
scripts/build_memmap.py            ✅ 已复制
docs/memmap_migration_plan.md      ✅ 本文档
```

### 9.2 修改文件（按优先级排序）

| 优先级 | 文件 | 变更类型 | 说明 |
|--------|------|----------|------|
| P0 | `models/skeleton.py` | 重写 | COCO17→OpenPose18 骨架 |
| P0 | `models/wiflow_spatial_encoder.py` | 重写 | stride 2×2 + input_channels=3 |
| P0 | `models/wiflow_joint_decoder.py` | 微调 | 18 queries |
| P0 | `models/wiflow_hierarchical_joint_decoder.py` | 微调 | 18 queries + stage_indices |
| P0 | `models/wiflow_model.py` | 简化 | 移除 sequence_length + 5D 分支 |
| P0 | `models/__init__.py` | 微调 | 导出符号更新 |
| P0 | `dataloader.py` | 完全重写 | 移除 HDF5，仅 NPY |
| P0 | `train.py` | 大幅简化 | 移除 split_scheme/csi_features/sequence_length |
| P0 | `eval.py` | 大幅简化 | 同上 |
| P0 | `pose_targets.py` | 微调 | 常量引用更新 |
| P1 | `data/memmap_dataset.py` | 微调 | build_targets 默认 False |
| P1 | `AGENTS.md` | 更新 | 新命令、新架构 |
| P1 | `tests/conftest.py` | 重写 | memmap fixture |
| P1 | `tests/test_dataloader.py` | 重写 | MemmapDataset 测试 |
| P1 | `tests/test_wiflow_model.py` | 更新 | 新形状 |
| P1 | `tests/test_wiflow_decoder.py` | 更新 | 18 输出 |
| P1 | `tests/test_pose_targets.py` | 更新 | 18 关键点 |
| P1 | `tests/test_train.py` | 新增 | memmap smoketest |

### 9.3 不变文件

```
data/heatmap_gt.py
scripts/build_memmap.py
scripts/build_h5_dataset.py
models/wiflow_axial_encoder.py
models/wiflow_heatmap_decoder.py
models/wiflow_spatial_temporal_fuser.py
models/wiflow_attention_pooler.py
models/wiflow_temporal_encoder.py
models/wiflow_skeleton_decoder.py
tests/test_eval.py
tests/test_skeleton.py
```

---

## 10. 风险识别与缓解

| ID | 风险描述 | 概率 | 影响 | 缓解措施 |
|----|----------|------|------|----------|
| R1 | SpatialEncoder 时间下采样（64→16）丢失过多时间信息 | 中 | 高 | 对比实验：stride (2,2) vs (1,2)；如精度显著下降，考虑仅 1 次时间下采样（64→32） |
| R2 | 时间维度从 10 上采样至 64 导致 GPU 显存增加（中间激活增大约 6.4×） | 中 | 中 | 监控显存；必要时减小 `batch_size` |
| R3 | OpenPose18 骨架 GNN 邻接矩阵语义变化影响 decoder 精度 | 低 | 中 | 对比 COCO17 vs OpenPose18 骨架的 MPJPE |
| R4 | Windows mmap 多进程共享行为异常 | 中 | 低 | 初始 `num_workers=2`；必要时回退 `num_workers=0` |
| R5 | 全局搜索遗漏 COCO17 引用 | 中 | 中 | 阶段 8 执行 `grep -r "COCO" --include="*.py"` 确认 |
| R6 | 旧 checkpoint 无法加载（字段不兼容） | 高 | 低 | 旧 checkpoint 本身就需要重新训练；文档说明不兼容 |

---

> **下一步**：请审核本计划 v2，确认后进行系统性完善与纠正，然后进入代码实施阶段。