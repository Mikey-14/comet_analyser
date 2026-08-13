"""
彗星分割模块
============
从预处理后的图像中分割出彗星区域（头部 + 尾部）。

方法：
1. Otsu 自动阈值分割
2. 轮廓检测与筛选
3. 彗星头部定位（亮度峰值）
4. 彗星尾部分离
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CometRegion:
    """彗星分割结果"""
    head_mask: np.ndarray       # 头部二值掩码
    tail_mask: np.ndarray       # 尾部二值掩码
    comet_mask: np.ndarray      # 整体彗星掩码（头+尾）
    head_center: Tuple[int, int]  # 头部中心坐标 (x, y)
    head_bbox: Tuple[int, int, int, int]  # 头部包围盒 (x, y, w, h)
    tail_bbox: Tuple[int, int, int, int]  # 尾部包围盒
    orientation: float          # 彗星朝向角度（弧度）


def threshold_otsu(img: np.ndarray) -> np.ndarray:
    """
    Otsu 自动阈值二值化。

    Args:
        img: 灰度图像 (uint8)

    Returns:
        二值掩码 (背景0, 前景255)
    """
    _, binary = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def threshold_adaptive(
    img: np.ndarray,
    block_size: int = 31,
    c: int = 2
) -> np.ndarray:
    """
    自适应阈值二值化（处理光照不均）。

    Args:
        img: 灰度图像
        block_size: 邻域大小（奇数）
        c: 减去的常数

    Returns:
        二值掩码
    """
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )


def find_largest_object(binary: np.ndarray) -> np.ndarray:
    """
    保留二值图中最大的连通域（主体彗星），去除小噪点。

    Args:
        binary: 二值图像

    Returns:
        仅包含最大物体的二值掩码
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return binary

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


def find_head_center(gray: np.ndarray, mask: np.ndarray) -> Tuple[int, int]:
    """
    通过寻找彗星区域内亮度最高点来定位头部中心。
    彗星头部通常是整颗彗星中最亮的区域。

    Args:
        gray: 原始灰度图像
        mask: 彗星整体掩码

    Returns:
        头部中心坐标 (x, y)
    """
    # 只在彗星区域内搜索
    masked = gray.copy()
    masked[mask == 0] = 0

    _, _, _, max_loc = cv2.minMaxLoc(masked)
    return max_loc  # (x, y)


def separate_head_tail(
    gray: np.ndarray,
    comet_mask: np.ndarray,
    head_center: Tuple[int, int],
    head_radius_ratio: float = 0.25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将彗星分为头部和尾部。

    策略：以头部中心为圆心，用彗星总长度的 head_radius_ratio 比例
    作为头部半径，圆内为头部，其余为尾部。

    Args:
        gray: 灰度图像
        comet_mask: 整体彗星掩码
        head_center: 头部中心 (x, y)
        head_radius_ratio: 头部半径占彗星总长度的比例

    Returns:
        (head_mask, tail_mask)
    """
    hx, hy = head_center

    # 计算彗星轮廓
    contours, _ = cv2.findContours(
        comet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros_like(comet_mask), comet_mask

    cnt = max(contours, key=cv2.contourArea)

    # 估算彗星总长度：轮廓点到头部中心的最大距离
    distances = []
    for pt in cnt[:, 0, :]:
        d = np.sqrt((pt[0] - hx) ** 2 + (pt[1] - hy) ** 2)
        distances.append(d)
    comet_length = max(distances) if distances else 1.0

    # 头部半径
    head_r = int(comet_length * head_radius_ratio)
    head_r = max(head_r, 5)  # 最小保护

    # 创建头部圆形掩码
    head_mask = np.zeros_like(comet_mask)
    cv2.circle(head_mask, (hx, hy), head_r, 255, -1)

    # 与彗星掩码取交集
    head_mask = cv2.bitwise_and(head_mask, comet_mask)

    # 尾部 = 彗星整体 - 头部
    tail_mask = cv2.subtract(comet_mask, head_mask)

    return head_mask, tail_mask


def segment_comet(
    gray: np.ndarray,
    threshold_method: str = "otsu",
    head_radius_ratio: float = 0.25,
    adaptive_block: int = 31,
    adaptive_c: int = 2
) -> Optional[CometRegion]:
    """
    完整的彗星分割管线。

    Args:
        gray: 预处理后的灰度图像
        threshold_method: 'otsu' 或 'adaptive'
        head_radius_ratio: 头部半径比例
        adaptive_block: 自适应阈值块大小
        adaptive_c: 自适应阈值常数

    Returns:
        CometRegion 对象，分割失败则返回 None
    """
    # 1. 阈值分割
    if threshold_method == "otsu":
        binary = threshold_otsu(gray)
    elif threshold_method == "adaptive":
        binary = threshold_adaptive(gray, adaptive_block, adaptive_c)
    else:
        raise ValueError(f"未知阈值方法: {threshold_method}")

    # 2. 保留最大连通域
    comet_mask = find_largest_object(binary)

    if cv2.countNonZero(comet_mask) == 0:
        return None

    # 3. 定位头部中心（亮度最高点）
    head_center = find_head_center(gray, comet_mask)

    # 4. 分离头尾
    head_mask, tail_mask = separate_head_tail(
        gray, comet_mask, head_center, head_radius_ratio
    )

    # 5. 计算包围盒和朝向
    hx, hy = head_center

    # 头部包围盒
    head_cnt, _ = cv2.findContours(
        head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if head_cnt:
        hx_bb, hy_bb, hw, hh = cv2.boundingRect(max(head_cnt, key=cv2.contourArea))
        head_bbox = (hx_bb, hy_bb, hw, hh)
    else:
        head_bbox = (hx - 5, hy - 5, 10, 10)

    # 尾部包围盒
    tail_cnt, _ = cv2.findContours(
        tail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if tail_cnt:
        tx, ty, tw, th = cv2.boundingRect(max(tail_cnt, key=cv2.contourArea))
        tail_bbox = (tx, ty, tw, th)
    else:
        tail_bbox = (0, 0, 0, 0)

    # 朝向角度（通过彗星轮廓的拟合椭圆计算）
    contour, _ = cv2.findContours(
        comet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    orientation = 0.0
    if contour and len(contour[0]) >= 5:
        ellipse = cv2.fitEllipse(contour[0])
        orientation = np.deg2rad(ellipse[2])  # 转弧度

    return CometRegion(
        head_mask=head_mask,
        tail_mask=tail_mask,
        comet_mask=comet_mask,
        head_center=head_center,
        head_bbox=head_bbox,
        tail_bbox=tail_bbox,
        orientation=orientation
    )