"""
彗星实验分析工具 (Comet Assay Analyser)
========================================
用于分析彗星实验图像中 DNA 损伤程度的桌面应用。
基于 PyQt5 + OpenCV + scikit-image 构建。
"""

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    # 高 DPI 缩放：必须在 QApplication 创建之前设置，
    # 使界面按系统显示缩放比例自动适配（支持 125%/150% 等分数缩放）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Comet Analyser")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()