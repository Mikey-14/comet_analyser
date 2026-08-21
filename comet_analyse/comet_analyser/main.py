"""
彗星实验分析工具 (Comet Assay Analyser)
========================================
用于分析彗星实验图像中 DNA 损伤程度的桌面应用。
基于 PyQt5 + OpenCV 构建。
"""

import sys
import ctypes

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow


def _set_windows_dpi_awareness():
    """显式设置进程 DPI 感知级别（Per-Monitor V2）。

    打包成 exe 后，PyInstaller 默认 manifest 可能已声明系统 DPI 感知，
    导致 Qt 的 AA_EnableHighDpiScaling 无法生效（devicePixelRatio 恒为 1），
    最终界面在 2K/4K 屏上显得过小。这里在 QApplication 创建前显式设置，
    可绕过 manifest 冲突，让 Qt 正确按显示器缩放。
    """
    if sys.platform != "win32":
        return
    try:
        # Windows 10 1703+：Per-Monitor V2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            # 回退：Per-Monitor V1
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                # 回退：系统 DPI 感知
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def _configure_high_dpi():
    """配置 Qt 高 DPI 缩放（必须在 QApplication 创建前调用）。"""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )


def _configure_global_font(app):
    """设置全局字体与控件样式，保证 1080p/2K/4K 下字体、按钮大小合适。"""
    # 舒适基准字号（点）：Qt 会按屏幕 DPI 自动缩放为正确的物理尺寸
    font = QFont()
    font.setFamily("Segoe UI")
    font.setPointSizeF(11.0)
    app.setFont(font)

    # 全局样式：统一按钮内边距，避免高 DPI 下按钮过于拥挤
    app.setStyleSheet("QPushButton { padding: 6px 14px; }")


def main():
    _set_windows_dpi_awareness()
    _configure_high_dpi()

    app = QApplication(sys.argv)
    app.setApplicationName("Comet Analyser")
    app.setStyle("Fusion")

    _configure_global_font(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
