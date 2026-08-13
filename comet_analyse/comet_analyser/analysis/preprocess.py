"""
图像预处理模块
==============
包含彗星实验图像预处理管线：
- 灰度化
- 去噪（高斯滤波/中值滤波）
- 对比度增强（CLAHE 自适应直方图均衡化）
- 背景校正
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    """
    安全加载图像，支持中文路径。

    Args:
        path: 图片路径
        grayscale: 是否直接以灰度模式读取

    Returns:
        BGR 或灰度图像数组
    """
    # 使用 imdecode 方式绕过 cv2.imread 的中文路径问题
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    将 BGR/彩色图转换为灰度图。
    若输入已是单通道则直接返回。
    """
    if img.ndim == 2:
        return img.copy()
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def denoise(
    img: np.ndarray,
    method: str = "gaussian",
    kernel_size: int = 5
) -> np.ndarray:
    """
    图像降噪。

    Args:
        img: 灰度图
        method: 'gaussian', 'median', 'bilateral'
        kernel_size: 核大小（自动调整为奇数）

    Returns:
        降噪后的图像
    """
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif method == "median":
        return cv2.medianBlur(img, ksize)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, d=ksize, sigmaColor=75, sigmaSpace=75)
    else:
        raise ValueError(f"未知降噪方法: {method}")


def enhance_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    使用 CLAHE（自适应直方图均衡化）增强图像对比度。
    对荧光彗星图像尤其有效。

    Args:
        img: 灰度图 (uint8)
        clip_limit: 裁剪阈值
        tile_grid_size: 瓦片大小

    Returns:
        增强后的图像
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def subtract_background(
    img: np.ndarray,
    filter_radius: int = 50
) -> np.ndarray:
    """
    Rolling ball / 大核均值滤波 背景减除。

    用大核模糊估计背景，然后减去背景以校正不均匀光照。

    Args:
        img: 灰度图
        filter_radius: 背景滤波器半径

    Returns:
        背景校正后的图像（归一化到 0-255）
    """
    # 使用大核均值滤波近似背景
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (filter_radius * 2 + 1, filter_radius * 2 + 1)
    )
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # 减去背景
    corrected = cv2.subtract(img, background)

    # 归一化到 [0, 255]
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


def preprocess_pipeline(
    path_or_img,
    do_denoise: bool = True,
    do_clahe: bool = True,
    do_bg_subtract: bool = True,
    denoise_method: str = "gaussian",
    denoise_kernel: int = 5,
    clahe_clip: float = 2.0,
    bg_radius: int = 50
) -> np.ndarray:
    """
    完整预处理管线：灰度化 → 去噪 → CLAHE → 背景校正

    Args:
        path_or_img: 图片路径(str) 或 numpy 数组
        do_denoise: 是否去噪
        do_clahe: 是否 CLAHE 增强
        do_bg_subtract: 是否背景减除
        denoise_method: 去噪方法
        denoise_kernel: 去噪核大小
        clahe_clip: CLAHE 裁剪阈值
        bg_radius: 背景减除半径

    Returns:
        预处理后的灰度图 (uint8)
    """
    # 加载
    if isinstance(path_or_img, str):
        img = load_image(path_or_img)
    else:
        img = path_or_img

    # 灰度化
    gray = to_grayscale(img)

    # 去噪
    if do_denoise:
        gray = denoise(gray, method=denoise_method, kernel_size=denoise_kernel)

    # CLAHE 增强
    if do_clahe:
        gray = enhance_clahe(gray, clip_limit=clahe_clip)

    # 背景校正
    if do_bg_subtract:
        gray = subtract_background(gray, filter_radius=bg_radius)

    return gray