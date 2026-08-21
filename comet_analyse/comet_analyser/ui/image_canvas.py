"""
交互式图像画布
==============
支持鼠标拖拽绘制矩形框选区域的 QLabel 子类。

鼠标操作：
- 按下左键并拖拽：绘制矩形框选
- 松开左键：确认框选，回调通知外部
- 右键单击：取消当前框选
"""

from PyQt5.QtWidgets import QLabel, QMenu
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QPoint, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QPolygonF
import numpy as np

# ==================== 标注图层颜色常量 ====================
HEAD_CIRCLE_COLOR = QColor(0, 150, 255)      # 头部：亮蓝描边、无填充
TAIL_COLOR = QColor(255, 255, 0)             # 尾部：明黄描边
TAIL_FILL_COLOR = QColor(255, 255, 0, 150)   # 尾部：半透明明黄填充


class ImageCanvas(QLabel):
    """
    可交互的图像显示控件。

    信号：
        rect_selected(QRect): 用户完成框选后触发
        rect_changed(QRect): 拖拽过程中实时触发（用于预览）
    """

    rect_selected = pyqtSignal(QRect)   # 框选完成
    rect_changed = pyqtSignal(QRect)    # 拖拽中
    selection_cleared = pyqtSignal()     # 取消框选
    cell_selected = pyqtSignal(int)     # 选中细胞（索引）
    cell_delete_requested = pyqtSignal(int)  # 请求删除细胞（索引）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

        # 缩放相关
        self._original_pixmap: QPixmap = None       # 原始图像
        self._scaled_pixmap: QPixmap = None         # 缩放后
        self._scale_factor: float = 1.0              # 缩放比例
        self._offset_x: int = 0                      # 图像在控件中的偏移 X
        self._offset_y: int = 0                      # 图像在控件中的偏移 Y

        # 框选状态
        self._drawing: bool = False
        self._start_point: QPoint = None
        self._current_rect: QRect = None             # 控件坐标系的矩形

        # 框选样式
        self._pen = QPen(QColor(255, 50, 50), 2, Qt.DashLine)
        self._fill_color = QColor(255, 100, 100, 60)  # 半透明填充

        # 对外显示原始 pixmap
        self._display_pixmap: QPixmap = None

        # 标注图层（叠加在图像上层的持久标注，直到切换图片才清空）
        # 每个标注: {"head_center": (x, y), "head_radius": float,
        #            "tail_contours": [[(x, y), ...], ...]}
        self._annotations: list = []
        self._cell_bboxes: list = []       # 每个细胞的外接矩形（原图坐标）
        self._selected_cell_idx: int = -1  # 当前选中的细胞索引

    # ==================== 公共接口 ====================

    def set_image(self, pixmap: QPixmap):
        """设置原始图像并自适应缩放显示"""
        self._original_pixmap = pixmap
        self._current_rect = None
        self._annotations = []
        self._update_scaled()

    def set_annotations(self, annotations: list):
        """设置标注图层内容（替换现有标注），并计算碰撞检测用的外接矩形"""
        self._annotations = list(annotations)
        self._cell_bboxes = [self._compute_cell_bbox(ann) for ann in annotations]
        self._selected_cell_idx = -1
        self._update_display()

    def clear_annotations(self):
        """清空标注图层"""
        self._annotations = []
        self._update_display()

    def get_selected_image_rect(self) -> QRect:
        """获取当前框选区域（映射回原始图像坐标）"""
        if self._current_rect and self._original_pixmap:
            return self._map_to_original(self._current_rect)
        return QRect()

    def get_selected_region_numpy(self, cv_image: np.ndarray) -> np.ndarray:
        """
        从原始 cv2 图像中裁剪出框选区域。

        Args:
            cv_image: 原始图像 (BGR numpy array)

        Returns:
            裁剪后的区域
        """
        rect = self.get_selected_image_rect()
        if rect.isEmpty():
            return cv_image
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        # 边界保护
        x, y = max(0, x), max(0, y)
        h_img, w_img = cv_image.shape[:2]
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        return cv_image[y:y + h, x:x + w]

    def clear_selection(self):
        """清除当前框选"""
        self._current_rect = None
        self._update_display()
        self.selection_cleared.emit()

    def deselect_cell(self):
        """取消细胞选中状态"""
        self._selected_cell_idx = -1
        self._update_display()

    def _compute_cell_bbox(self, ann: dict) -> QRect:
        """根据标注字典计算外接矩形（原图坐标）"""
        hx, hy = ann.get("head_center", (0.0, 0.0))
        r = float(ann.get("head_radius", 0.0))
        min_x, min_y = hx - r, hy - r
        max_x, max_y = hx + r, hy + r
        for poly in ann.get("tail_contours", []):
            for px, py in poly:
                min_x = min(min_x, px)
                min_y = min(min_y, py)
                max_x = max(max_x, px)
                max_y = max(max_y, py)
        return QRect(int(min_x), int(min_y),
                      max(1, int(max_x - min_x)), max(1, int(max_y - min_y)))

    def _hit_test_cell(self, img_point: QPoint) -> int:
        """检测原图坐标点是否命中某个细胞，返回索引（-1 表示未命中）"""
        for i, bbox in enumerate(self._cell_bboxes):
            if bbox.contains(img_point):
                return i
        return -1

    def has_selection(self) -> bool:
        """是否有有效框选"""
        return self._current_rect is not None and not self._current_rect.isEmpty()

    # ==================== 内部缩放逻辑 ====================

    def resizeEvent(self, event):
        """控件大小改变时重新缩放图像"""
        super().resizeEvent(event)
        if self._original_pixmap:
            self._update_scaled()

    def _update_scaled(self):
        """重新计算缩放参数"""
        if not self._original_pixmap:
            return

        label_w = self.width()
        label_h = self.height()
        img_w = self._original_pixmap.width()
        img_h = self._original_pixmap.height()

        if label_w <= 0 or label_h <= 0:
            return

        # 等比缩放
        scale_w = label_w / img_w
        scale_h = label_h / img_h
        self._scale_factor = min(scale_w, scale_h)

        new_w = int(img_w * self._scale_factor)
        new_h = int(img_h * self._scale_factor)

        self._scaled_pixmap = self._original_pixmap.scaled(
            new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # 计算居中偏移
        self._offset_x = (label_w - new_w) // 2
        self._offset_y = (label_h - new_h) // 2

        self._update_display()

    def _update_display(self):
        """刷新显示（图像 + 框选矩形）"""
        if not self._scaled_pixmap:
            return

        # 创建绘制画布
        canvas = QPixmap(self.size())
        canvas.fill(QColor(240, 240, 240))  # 浅灰背景

        painter = QPainter(canvas)

        # 绘制缩放后的图像
        painter.drawPixmap(self._offset_x, self._offset_y, self._scaled_pixmap)

        # 绘制框选矩形
        if self._current_rect and not self._current_rect.isEmpty():
            painter.setPen(self._pen)
            painter.setBrush(self._fill_color)
            painter.drawRect(self._current_rect)

        # 绘制标注图层（彗星轮廓：头部圆 + 尾部明黄）
        self._draw_annotations(painter)

        painter.end()
        self._display_pixmap = canvas
        self.setPixmap(canvas)

    def _draw_annotations(self, painter: QPainter):
        """在图像上层绘制持久标注（头部圆无填充、尾部明黄填充）"""
        if not self._annotations or not self._scaled_pixmap:
            return

        s = self._scale_factor
        ox = self._offset_x
        oy = self._offset_y

        painter.save()

        # 1. 尾部：明黄填充 + 描边
        for ann in self._annotations:
            for poly in ann.get("tail_contours", []):
                if len(poly) < 3:
                    continue
                qpoly = QPolygonF([QPointF(ox + px * s, oy + py * s) for (px, py) in poly])
                painter.setPen(QPen(TAIL_COLOR, 1))
                painter.setBrush(QBrush(TAIL_FILL_COLOR))
                painter.drawPolygon(qpoly)

        # 2. 头部：正圆、无填充描边
        pen_width = max(1, round(2 * s))
        for ann in self._annotations:
            hx, hy = ann.get("head_center", (0.0, 0.0))
            radius = float(ann.get("head_radius", 0.0))
            if radius <= 0:
                continue
            cx = ox + hx * s
            cy = oy + hy * s
            r = radius * s
            painter.setPen(QPen(HEAD_CIRCLE_COLOR, pen_width))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), r, r)

        # 3. 选中细胞高亮（红色虚线矩形）
        if (self._selected_cell_idx >= 0 and
                self._selected_cell_idx < len(self._cell_bboxes)):
            bbox = self._cell_bboxes[self._selected_cell_idx]
            wx = ox + bbox.x() * s
            wy = oy + bbox.y() * s
            ww = bbox.width() * s
            wh = bbox.height() * s
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
            painter.setBrush(QBrush(QColor(255, 0, 0, 30)))
            painter.drawRect(QRect(int(wx), int(wy), int(ww), int(wh)))

        painter.restore()

    # ==================== 坐标映射 ====================

    def _widget_to_image(self, point: QPoint) -> QPoint:
        """控件坐标 → 缩放图像坐标"""
        x = (point.x() - self._offset_x) / self._scale_factor
        y = (point.y() - self._offset_y) / self._scale_factor
        return QPoint(int(x), int(y))

    def _map_to_original(self, rect: QRect) -> QRect:
        """控件坐标矩形 → 原始图像坐标矩形"""
        top_left = self._widget_to_image(rect.topLeft())
        bottom_right = self._widget_to_image(rect.bottomRight())

        x = min(top_left.x(), bottom_right.x())
        y = min(top_left.y(), bottom_right.y())
        w = abs(bottom_right.x() - top_left.x())
        h = abs(bottom_right.y() - top_left.y())

        # 边界保护
        if self._original_pixmap:
            x = max(0, min(x, self._original_pixmap.width()))
            y = max(0, min(y, self._original_pixmap.height()))
            w = min(w, self._original_pixmap.width() - x)
            h = min(h, self._original_pixmap.height() - y)

        return QRect(x, y, w, h)

    def _is_inside_image(self, point: QPoint) -> bool:
        """判断控件坐标点是否在图像区域内"""
        if not self._scaled_pixmap:
            return False
        x, y = point.x(), point.y()
        return (self._offset_x <= x <= self._offset_x + self._scaled_pixmap.width() and
                self._offset_y <= y <= self._offset_y + self._scaled_pixmap.height())

    # ==================== 鼠标事件 ====================

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._is_inside_image(event.pos()):
                # 先检测是否点击在已有细胞上
                img_pos = self._widget_to_image(event.pos())
                clicked_idx = self._hit_test_cell(img_pos)
                if clicked_idx >= 0:
                    self._selected_cell_idx = clicked_idx
                    self.setFocus()
                    self._update_display()
                    self.cell_selected.emit(clicked_idx)
                    return
                # 未命中细胞，开始框选
                self._drawing = True
                self._start_point = event.pos()
                self._current_rect = QRect(self._start_point, self._start_point)
                self.setCursor(Qt.CrossCursor)
        elif event.button() == Qt.RightButton:
            if self._is_inside_image(event.pos()):
                img_pos = self._widget_to_image(event.pos())
                clicked_idx = self._hit_test_cell(img_pos)
                if clicked_idx >= 0:
                    self._selected_cell_idx = clicked_idx
                    self._update_display()
                    self._show_cell_context_menu(event.pos(), clicked_idx)
                    return
            self.clear_selection()

    def mouseMoveEvent(self, event):
        if self._drawing:
            end_point = event.pos()
            # 限制在图像区域
            if not self._is_inside_image(end_point):
                # 钳制到图像边界
                x = max(self._offset_x,
                        min(end_point.x(), self._offset_x + self._scaled_pixmap.width()))
                y = max(self._offset_y,
                        min(end_point.y(), self._offset_y + self._scaled_pixmap.height()))
                end_point = QPoint(x, y)

            self._current_rect = QRect(self._start_point, end_point).normalized()
            self._update_display()
            self.rect_changed.emit(self._current_rect)

        # 更新光标样式
        if not self._drawing:
            if self._is_inside_image(event.pos()):
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False

            # 最小框选尺寸过滤（避免误触）
            if self._current_rect and min(self._current_rect.width(),
                                          self._current_rect.height()) >= 5:
                self.rect_selected.emit(self._current_rect)
            else:
                self._current_rect = None
                self._update_display()

    def _show_cell_context_menu(self, pos: QPoint, idx: int):
        """在鼠标位置弹出细胞右键菜单"""
        menu = QMenu(self)
        delete_action = menu.addAction("删除此细胞")
        action = menu.exec_(self.mapToGlobal(pos))
        if action == delete_action:
            self.cell_delete_requested.emit(idx)

    def keyPressEvent(self, event):
        """Delete / Backspace 删除选中细胞"""
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self._selected_cell_idx >= 0:
                self.cell_delete_requested.emit(self._selected_cell_idx)
                return
        super().keyPressEvent(event)

    # ==================== 框选样式设置 ====================

    def set_selection_color(self, r, g, b, alpha=60):
        """设置框选颜色"""
        self._pen = QPen(QColor(r, g, b), 2, Qt.DashLine)
        self._fill_color = QColor(r, g, b, alpha)
        if self._current_rect:
            self._update_display()