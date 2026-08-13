"""
彗星指标计算模块
================
从分割结果中计算彗星实验的关键定量指标：

指标说明：
┌─────────────────────┬──────────────────────────────────────────────┐
│ Tail Length         │ 彗星尾部 DNA 迁移的最大距离 (μm 或像素)     │
│ Tail DNA%           │ 尾部荧光强度占整颗彗星的百分比               │
│ Tail Moment (TM)    │ TM = Tail DNA% × Tail Length                 │
│ Olive Tail Moment   │ OTM = (Tail_COM_x - Head_COM_x) × Tail DNA%  │
└─────────────────────┴──────────────────────────────────────────────┘
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .segmentation import CometRegion


@dataclass
class CometMetrics:
    """彗星分析指标结果"""
    # ---- 基本指标（像素单位）----
    tail_length_px: float          # 尾长（像素）
    head_area_px: int              # 头部面积（像素）
    tail_area_px: int              # 尾部面积（像素）
    comet_area_px: int             # 彗星总面积（像素）

    # ---- 荧光强度 ----
    head_intensity: float          # 头部总荧光强度
    tail_intensity: float          # 尾部总荧光强度
    comet_intensity: float         # 彗星总荧光强度

    # ---- 核心指标 ----
    tail_dna_percent: float        # Tail DNA% (0-100)
    tail_moment: float             # Tail Moment (TM)
    olive_tail_moment: float       # Olive Tail Moment (OTM)

    # ---- 质心 ----
    head_com: Tuple[float, float]  # 头部质心 (x, y)
    tail_com: Tuple[float, float]  # 尾部质心 (x, y)

    # ---- 像素比例 ----
    pixel_size_um: float = 1.0     # 像素对应的微米数（默认1:1）

    @property
    def tail_length_um(self) -> float:
        """尾长（微米）"""
        return self.tail_length_px * self.pixel_size_um

    def to_dict(self) -> Dict[str, float]:
        """转为字典，方便导出"""
        return {
            "Tail Length (px)": round(self.tail_length_px, 2),
            "Tail Length (µm)": round(self.tail_length_um, 2),
            "Head Area (px)": self.head_area_px,
            "Tail Area (px)": self.tail_area_px,
            "Comet Area (px)": self.comet_area_px,
            "Head Intensity": round(self.head_intensity, 2),
            "Tail Intensity": round(self.tail_intensity, 2),
            "Comet Intensity": round(self.comet_intensity, 2),
            "Tail DNA%": round(self.tail_dna_percent, 2),
            "Tail Moment": round(self.tail_moment, 2),
            "Olive Tail Moment": round(self.olive_tail_moment, 2),
        }


def compute_tail_length(
    tail_mask: np.ndarray,
    head_center: Tuple[int, int]
) -> float:
    """
    计算尾长：从头部中心到尾部最远像素的距离。

    Args:
        tail_mask: 尾部二值掩码
        head_center: 头部中心 (x, y)

    Returns:
        尾长（像素）
    """
    # 找尾部所有非零像素坐标
    ys, xs = np.where(tail_mask > 0)
    if len(xs) == 0:
        return 0.0

    hx, hy = head_center
    distances = np.sqrt((xs - hx) ** 2 + (ys - hy) ** 2)
    return float(np.max(distances))


def compute_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """
    计算二值掩码的质心（几何中心）。

    Args:
        mask: 二值掩码

    Returns:
        质心坐标 (x, y)
    """
    moments = cv2_moments(mask)
    if moments["m00"] == 0:
        return (0.0, 0.0)
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return (cx, cy)


def cv2_moments(mask: np.ndarray) -> Dict[str, float]:
    """轻量级矩计算（避免导入 cv2 用于此小功能）"""
    ys, xs = np.where(mask > 0)
    m00 = len(xs)
    if m00 == 0:
        return {"m00": 0, "m10": 0, "m01": 0}
    return {
        "m00": float(m00),
        "m10": float(np.sum(xs)),
        "m01": float(np.sum(ys))
    }


def compute_region_intensity(
    gray: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    计算指定区域内的总荧光强度。

    Args:
        gray: 灰度图像
        mask: 区域掩码

    Returns:
        总荧光强度
    """
    return float(np.sum(gray[mask > 0]))


def compute_all_metrics(
    gray: np.ndarray,              # 背景扣除后的灰度图（用于强度计算）
    region: CometRegion,
    pixel_size_um: float = 1.0
) -> CometMetrics:
    """
    根据分割结果计算所有彗星指标。

    Args:
        gray: 背景扣除后的灰度图像（用于计算真实荧光强度）
        region: 分割结果
        pixel_size_um: 像素对应的微米数

    Returns:
        CometMetrics 对象
    """
    # 面积
    head_area = int(np.sum(region.head_mask > 0))
    tail_area = int(np.sum(region.tail_mask > 0))
    comet_area = int(np.sum(region.comet_mask > 0))

    # 荧光强度（使用原始灰度图）
    head_intensity = compute_region_intensity(gray, region.head_mask)
    tail_intensity = compute_region_intensity(gray, region.tail_mask)
    comet_intensity = head_intensity + tail_intensity

    # 质心
    head_com = compute_centroid(region.head_mask)
    tail_com = compute_centroid(region.tail_mask)

    # 尾长
    tail_len = compute_tail_length(region.tail_mask, region.head_center)

    # Tail DNA%
    if comet_intensity > 0:
        tail_dna_pct = (tail_intensity / comet_intensity) * 100.0
    else:
        tail_dna_pct = 0.0

    # Tail Moment = Tail DNA% × Tail Length
    tail_moment = (tail_dna_pct / 100.0) * tail_len

    # Olive Tail Moment = (Tail_COM_x - Head_COM_x) × Tail DNA%
    olive_tail_moment = abs(tail_com[0] - head_com[0]) * (tail_dna_pct / 100.0)

    return CometMetrics(
        tail_length_px=tail_len,
        head_area_px=head_area,
        tail_area_px=tail_area,
        comet_area_px=comet_area,
        head_intensity=head_intensity,
        tail_intensity=tail_intensity,
        comet_intensity=comet_intensity,
        tail_dna_percent=tail_dna_pct,
        tail_moment=tail_moment,
        olive_tail_moment=olive_tail_moment,
        head_com=head_com,
        tail_com=tail_com,
        pixel_size_um=pixel_size_um
    )