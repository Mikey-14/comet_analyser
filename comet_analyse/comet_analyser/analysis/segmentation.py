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
    head_centroid: Tuple[float, float] = (0.0, 0.0)  # 头部质心（用于标注圆）
    head_radius: float = 0.0                         # 头部拟合圆半径（像素）


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


def find_component_at(
    binary: np.ndarray,
    point: Tuple[int, int]
) -> np.ndarray:
    """
    保留包含指定点的连通域，其余清零。

    相比 find_largest_object（保留最大连通域），此函数以"种子点"锚定目标，
    在低阈值下更稳健——避免选中与彗星无关的大片背景。

    Args:
        binary: 二值图像
        point: 种子点坐标 (x, y)

    Returns:
        仅包含种子点所在连通域的二值掩码
    """
    x, y = int(point[0]), int(point[1])
    h, w = binary.shape[:2]
    if not (0 <= x < w and 0 <= y < h) or binary[y, x] == 0:
        return np.zeros_like(binary)

    num, labels = cv2.connectedComponents(binary)
    label = labels[y, x]
    return ((labels == label) * 255).astype(np.uint8)


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


def compute_head_circle(
    head_mask: np.ndarray
) -> Tuple[Tuple[float, float], float]:
    """
    根据头部掩码形状计算头部拟合圆：
    圆心取头部掩码的几何质心，半径取质心到最远头部像素的距离。

    Args:
        head_mask: 头部二值掩码

    Returns:
        ((cx, cy), radius) — 圆心坐标与半径（像素）
    """
    ys, xs = np.where(head_mask > 0)
    if len(xs) == 0:
        return (0.0, 0.0), 0.0
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    radius = float(np.max(distances))
    return (cx, cy), max(radius, 1.0)


def _apply_tail_direction(
    mask: np.ndarray,
    cx: int,
    cy: int,
    direction: str = "right"
) -> np.ndarray:
    """
    将尾部掩码限制在头部指定的拖尾方向上。

    统一标准：彗星从右往左飞行，头部在左、尾部拖向右（默认 "right"）。

    Args:
        mask: 尾部二值掩码
        cx: 头部中心 x
        cy: 头部中心 y
        direction: 拖尾方向 'right' / 'left' / 'up' / 'down'

    Returns:
        限制方向后的尾部掩码
    """
    h, w = mask.shape[:2]
    keep = np.zeros_like(mask)

    if direction == "right":
        keep[:, cx + 1:] = 255
    elif direction == "left":
        keep[:, :cx] = 255
    elif direction == "up":
        keep[:cy, :] = 255
    elif direction == "down":
        keep[cy + 1:, :] = 255
    else:
        raise ValueError(f"未知拖尾方向: {direction}")

    return cv2.bitwise_and(mask, keep)


def _apply_tail_width_limit(
    tail_mask: np.ndarray,
    head_center: Tuple[int, int],
    head_radius: float
) -> np.ndarray:
    """
    限制尾部宽度：任意位置的尾部上下宽度不得超过头部直径。

    以头部中心 y 为基准，截取高度 = 头部直径的水平带，
    超出部分全部裁掉。

    Args:
        tail_mask: 已限制方向的尾部掩码
        head_center: 头部中心 (x, y)
        head_radius: 头部拟合圆半径

    Returns:
        宽度限制后的尾部掩码
    """
    if head_radius <= 0:
        return tail_mask

    _, hy = head_center
    head_diameter = 2.0 * head_radius
    max_half_width = head_diameter / 2.0

    h = tail_mask.shape[0]
    y_min = max(0, int(round(hy - max_half_width)))
    y_max = min(h, int(round(hy + max_half_width)) + 1)

    limit = np.zeros_like(tail_mask)
    limit[y_min:y_max, :] = 255

    return cv2.bitwise_and(tail_mask, limit)


def _finalize_region(
    comet_mask: np.ndarray,
    head_mask: np.ndarray,
    tail_mask: np.ndarray,
    head_center: Tuple[int, int],
    tail_direction: str = "right",
    tail_mode: int = 1
) -> CometRegion:
    """根据已分离的头/尾掩码补齐包围盒、朝向与头部拟合圆信息。"""
    head_centroid, head_radius = compute_head_circle(head_mask)

    # 限制尾部只在头部拖尾方向（保持头部干净，去除环绕头部的光晕）
    if tail_direction is not None:
        tail_mask = _apply_tail_direction(
            tail_mask, head_center[0], head_center[1], tail_direction
        )

    # 模式2：宽度限制（尾部任意位置宽度不得超过头部直径）
    if tail_mode == 2:
        tail_mask = _apply_tail_width_limit(tail_mask, head_center, head_radius)

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
        orientation=orientation,
        head_centroid=head_centroid,
        head_radius=head_radius
    )


def _segment_by_threshold(
    gray: np.ndarray,
    head_thresh: float,
    tail_thresh: float,
    tail_direction: str = "right",
    tail_mode: int = 1
) -> Optional[CometRegion]:
    """
    基于自定义强度阈值分割彗星。

    头部亮度高于尾部，因此：
    - 彗星整体 = 强度 >= tail_thresh 的像素（取最大连通域）
    - 头部     = 以亮核（>= head_thresh）拟合的圆盘 ∩ 彗星（保持头部干净）
    - 尾部     = 彗星整体 - 头部，且局限在头部拖尾方向（默认右侧）
      模式1：方向限制；模式2：额外锥形限制（尾部宽度随距离线性收缩）

    Args:
        gray: 灰度图像 (uint8)
        head_thresh: 头部阈值 (0-1)
        tail_thresh: 尾部阈值 (0-1)
        tail_direction: 拖尾方向，默认 'right'
        tail_mode: 分割模式 1=方向限制 2=锥形限制

    Returns:
        CometRegion 对象，分割失败则返回 None
    """
    gray_norm = gray.astype(np.float32) / 255.0

    # 1. 头部亮核（用于定位头部中心与确定头部形状）
    head_core_raw = ((gray_norm >= head_thresh) * 255).astype(np.uint8)

    if cv2.countNonZero(head_core_raw) > 0:
        head_centroid, head_radius = compute_head_circle(head_core_raw)
        seed = (int(round(head_centroid[0])), int(round(head_centroid[1])))
    else:
        # 亮核为空（头部阈值过高）：用全图最亮点定位头部
        _, _, _, max_loc = cv2.minMaxLoc(gray)
        seed = max_loc

    # 2. 彗星 = 包含头部种子的连通域（>= tail_thresh）
    #    以头部锚定，避免低阈值时误选大片背景作为彗星
    comet_binary = ((gray_norm >= tail_thresh) * 255).astype(np.uint8)
    comet_mask = find_component_at(comet_binary, seed)
    if cv2.countNonZero(comet_mask) == 0:
        return None

    # 3. 头部亮核 ∩ 彗星
    head_core = cv2.bitwise_and(head_core_raw, comet_mask)
    head_centroid, head_radius = compute_head_circle(head_core)

    if cv2.countNonZero(head_core) == 0:
        # 彗星内无亮核：退化为最亮点 + 圆形分离
        head_center = find_head_center(gray, comet_mask)
        head_mask, tail_mask = separate_head_tail(gray, comet_mask, head_center)
        return _finalize_region(
            comet_mask, head_mask, tail_mask, head_center, tail_direction, tail_mode
        )

    cx = int(round(head_centroid[0]))
    cy = int(round(head_centroid[1]))

    # 4. 头部圆盘（干净的圆头部）
    head_disk = np.zeros_like(comet_mask)
    cv2.circle(head_disk, (cx, cy), int(round(head_radius)), 255, -1)
    head_mask = cv2.bitwise_and(head_disk, comet_mask)

    # 5. 尾部 = 彗星 - 头部圆盘（方向限制在 _finalize_region 中统一处理）
    tail_mask = cv2.subtract(comet_mask, head_mask)

    head_center = (cx, cy)
    return _finalize_region(
        comet_mask, head_mask, tail_mask, head_center, tail_direction, tail_mode
    )


def segment_comet(
    gray: np.ndarray,
    threshold_method: str = "otsu",
    head_radius_ratio: float = 0.25,
    adaptive_block: int = 31,
    adaptive_c: int = 2,
    head_thresh: Optional[float] = None,
    tail_thresh: Optional[float] = None,
    tail_direction: str = "right",
    tail_mode: int = 1
) -> Optional[CometRegion]:
    """
    完整的彗星分割管线。

    Args:
        gray: 预处理后的灰度图像
        threshold_method: 'otsu' 或 'adaptive'
        head_radius_ratio: 头部半径比例（仅自动模式使用）
        adaptive_block: 自适应阈值块大小
        adaptive_c: 自适应阈值常数
        head_thresh: 自定义头部阈值 (0-1)，与 tail_thresh 同时提供时启用
        tail_thresh: 自定义尾部阈值 (0-1)，与 head_thresh 同时提供时启用
        tail_direction: 拖尾方向，默认 'right'
        tail_mode: 分割模式 1=方向限制 2=锥形限制

    Returns:
        CometRegion 对象，分割失败则返回 None
    """
    # 自定义阈值模式（头/尾阈值分开设置）
    if head_thresh is not None and tail_thresh is not None:
        return _segment_by_threshold(
            gray, head_thresh, tail_thresh, tail_direction, tail_mode
        )

    # 自动阈值模式
    if threshold_method == "otsu":
        binary = threshold_otsu(gray)
    elif threshold_method == "adaptive":
        binary = threshold_adaptive(gray, adaptive_block, adaptive_c)
    else:
        raise ValueError(f"未知阈值方法: {threshold_method}")

    # 保留最大连通域
    comet_mask = find_largest_object(binary)
    if cv2.countNonZero(comet_mask) == 0:
        return None

    # 定位头部中心（亮度最高点）
    head_center = find_head_center(gray, comet_mask)

    # 分离头尾
    head_mask, tail_mask = separate_head_tail(
        gray, comet_mask, head_center, head_radius_ratio
    )

    return _finalize_region(
        comet_mask, head_mask, tail_mask, head_center, tail_direction, tail_mode
    )