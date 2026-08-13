"""
彗星实验分析工具 (Comet Assay Analyser)
========================================
用于分析彗星实验图像中 DNA 损伤程度的桌面应用。
基于 PyQt5 + OpenCV + scikit-image 构建。
"""

import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Comet Analyser")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()