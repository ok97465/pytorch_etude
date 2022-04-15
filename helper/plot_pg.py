# -*- coding: utf-8 -*-
r"""PyQtGraph를 이용하여 plot하는 module.

[Description]

Example
-------
[example]

Notes
-----
[Notes]

References
----------
.. [] 책: 저자명. (년). 챕터명. In 편집자명 (역할), 책명 (쪽). 발행지 : 발행사
.. [] 학위 논문: 학위자명, "논문제목", 대학원 이름 석사 학위논문, 1990
.. [] 저널 논문: 저자. "논문제목". 저널명, . pp.

:File name: plot_pg.py
:author: ok97465
:Date created: 2018-11-16 오후 5:28
"""
# Standard library imports
import sys
import os.path as osp
from typing import Iterable, Optional

# Third party imports
import numpy as np
import pyqtgraph as pg
from numpy import arange, argmin, array, ndarray
from qtpy.QtCore import QPoint, Qt, Signal, Slot
from qtpy.QtGui import QFont, QKeySequence
from qtpy.QtWidgets import QAction, QApplication, QMenu, QFileDialog


class PgTextItem(pg.TextItem):
    """PyQtGraph의 TextItem에서 Mouse event를 override한다."""

    sig_change_font_size = Signal(int)

    def __init__(self, *args, **kwargs):
        """."""
        super().__init__(*args, **kwargs)

    def updateTextPos(self):
        """Text Pos 계산 시 Parent 크기를 고려한다."""
        # update text position to obey anchor
        txt_rect = self.textItem.boundingRect()
        txt_tl = self.textItem.mapToParent(txt_rect.topLeft())
        txt_br = self.textItem.mapToParent(txt_rect.bottomRight())
        offset = (txt_br - txt_tl) * self.anchor

        # ROI의 Rect 크기를 반영한다.
        parent = self.parentItem()
        if parent:
            pixel_width = parent.pixelWidth()
            pixel_height = parent.pixelHeight()
            parent_width, parent_height = parent.size()
            if self.anchor[1] == 1:
                offset.setY(offset.y() - parent_height / pixel_height)
            if self.anchor[0] == 1:
                offset.setX(offset.x() - parent_width / pixel_width)
        # ROI의 Rect 크기를 반영한다 끝.

        self.textItem.setPos(-offset)

    def mouseClickEvent(self, event):
        """Textitem Click시 Text Item 위치를 변경한다."""
        cur_anchor = (int(self.anchor[0]), int(self.anchor[1]))
        next_anchor = {
            (1, 1): (0, 1),
            (0, 1): (0, 0),
            (0, 0): (1, 0),
            (1, 0): (1, 1),
        }
        self.setAnchor(next_anchor[cur_anchor])
        event.accept()

    def mouseDoubleClickEvent(self, event):
        """Textitem double click시 View가 reset되는 것을 방지한다."""
        event.accept()

    def wheelEvent(self, event):
        """Adjust font size."""
        delta = event.delta()
        font = self.textItem.font()
        font_size = font.pointSize()

        if delta > 0:
            font_size += 1
        else:
            font_size -= 1
        font_size = max([1, font_size])

        self.sig_change_font_size.emit(font_size)
        event.accept()


class PgImageViewROI(pg.ImageView):
    """ImageView 에 Mouse 명령을 추가 한다."""

    def __init__(self, *args, **kwargs):
        """PgImageViewOk를 초기화한다."""
        super().__init__(*args, **kwargs)
        self.font_family: str = "Courier New"
        self.font_size: str = "4"
        self.value_color: str = "#0581FF"
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        self.ratio_of_rect_size_to_pixel = 0.93
        self.axis_x: ndarray = array([])
        self.axis_y: ndarray = array([])
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()

        # menu creation is deferred because it is expensive and often
        # the user will never see the menu anyway.
        self.menu = None
        self.view.mouseClickEvent = self.rightClickEvent

    def set_scale(self, scale_x: float, scale_y: float):
        """Set scale."""
        self.scale_x = scale_x
        self.scale_y = scale_y

    def set_axis(self, axis_x: ndarray, axis_y: ndarray):
        """Set axis."""
        self.axis_x = axis_x
        self.axis_y = axis_y

    def size_of_roi(self):
        r"""그림의 Pixel을 1로 계산 했을 때 화면에 그릴 Data Marker(ROI)의 크기를 계산한다."""
        size_x = (1 - 2 * self.ratio_of_rect_size_to_pixel) * self.scale_x
        size_y = (1 - 2 * self.ratio_of_rect_size_to_pixel) * self.scale_y
        return pg.Point(size_x, size_y)

    def pos_of_roi(self, data_point_x: int, data_point_y: int):
        r"""그림의 Pixel을 1로 계산 했을 때 화면에 그릴 Data Marker(ROI)의 크기를 계산한다."""
        x = (data_point_x + self.ratio_of_rect_size_to_pixel) * self.scale_x
        y = (data_point_y + self.ratio_of_rect_size_to_pixel) * self.scale_y
        return x, y

    def html_of_data_marker_name(self, string):
        r"""Data Marker의 Text에 항목을 HTML로 변환하여 반환한다."""
        return f'<font size="{self.font_size}"; color="black">{string} </font>'

    def html_of_data_marker_value(self, string):
        r"""Data Marker의 Text에 값을 HTML로 변환하여 반환한다."""
        return (
            f'<strong><font size="{self.font_size}"; color=#0581FF>'
            f"{string}</font></strong>"
        )

    @staticmethod
    def contrast_color(rgb_hex: str):
        r"""배경색에서 잘보이는 색을 선택."""
        rgb_in = array(
            (int(rgb_hex[4:6], 16), int(rgb_hex[6:8], 16), int(rgb_hex[8:], 16))
        )
        rgb_out = array([0, 0, 0])
        rgb_std = rgb_in.std()
        rgb_min = rgb_in.min()
        rgb_min_idx = argmin(rgb_in)

        if rgb_min < 180:
            rgb_out[rgb_min_idx] = 230
        elif rgb_std < 20:
            rgb_out[0] = 230

        return rgb_out[0], rgb_out[1], rgb_out[2]

    def keyPressEvent(self, ev):
        """Shift+S를 누르면 mouse Mode를 변경한다."""
        key = ev.key()
        shift_pressed = ev.modifiers() & Qt.ShiftModifier
        if key == Qt.Key_S and shift_pressed:
            if self.view.vb.state["mouseMode"] == self.view.vb.RectMode:
                self.view.vb.setMouseMode(self.view.vb.PanMode)
            else:
                self.view.vb.setMouseMode(self.view.vb.RectMode)
        else:
            super().keyPressEvent(ev)

    def rightClickEvent(self, event):
        """Handle rightClick."""
        if event.button() == Qt.RightButton:
            if self.raiseContextMenu(event):
                event.accept()

    def raiseContextMenu(self, ev):
        """Raise context menu."""
        menu = self.getContextMenus()

        pos = ev.screenPos()
        menu.popup(QPoint(int(pos.x()), int(pos.y())))
        return True

    def export_view_clicked(self):
        """Export view."""
        self.scene.contextMenuItem = self.view
        self.scene.showExportDialog()

    def export_data_as_img_clicked(self):
        """Export data as image."""
        filename = QFileDialog.getSaveFileName(self, "저장할 이미지 이름 선택", "", "PNG (*.png)")
        if isinstance(filename, tuple):
            filename = filename[0]  # Qt4/5 API difference
        if filename == "":
            return
        if osp.splitext(filename)[1] != ".png":
            filename += ".png"
        self.export(filename)

    # This method will be called when this item's _children_ want to raise
    # a context menu that includes their parents' menus.
    def getContextMenus(self, event=None):
        """Get context menus.

        Args:
          event: DESCRIPTION. Defaults to None.

        """
        if self.menu is None:
            self.menu = QMenu()
            self.menu.setTitle(self.name + " options..")

            view_all = QAction("View all", self.menu)
            view_all.triggered.connect(self.view_all)
            self.menu.addAction(view_all)
            self.menu.view_all = view_all

            toggle_aspect_mode = QAction("Locked aspect", self.menu, checkable=True)
            toggle_aspect_mode.triggered.connect(self.toggle_aspect_mode)
            toggle_aspect_mode.setChecked(True)
            self.menu.addAction(toggle_aspect_mode)
            self.menu.toggle_aspect_mode = toggle_aspect_mode

            toggle_click_mode = QAction(
                "Mouse panmode",
                self.menu,
                shortcut=QKeySequence("Shift+S"),
                checkable=True,
            )
            toggle_click_mode.triggered.connect(self.toggle_mouse_mode)
            self.menu.addAction(toggle_click_mode)
            self.menu.toggle_mode = toggle_click_mode

            export_view = QAction("Export View", self.menu)
            export_view.setToolTip("Axis와 Marker를 포함한 화면을 캡쳐한다.")
            export_view.triggered.connect(self.export_view_clicked)
            self.menu.addAction(export_view)

            export_img = QAction("Export data as png", self.menu)
            export_img.setToolTip("Imagesc의 Data 원본을 Image 파일로 저장한다.")
            export_img.triggered.connect(self.export_data_as_img_clicked)
            self.menu.addAction(export_img)

        if self.view.vb.state["mouseMode"] == self.view.vb.PanMode:
            self.menu.toggle_mode.setChecked(True)
        else:
            self.menu.toggle_mode.setChecked(False)

        return self.menu

    def func_shift_left_click(self, event):
        """Shift를 누르고 마우스 왼쪽을 클릭하면 Data Marker를 표시한다.

        Args:
          event: Event.

        """
        modifiers = QApplication.keyboardModifiers()
        if event.button() == Qt.LeftButton and modifiers == Qt.ShiftModifier:
            pos = event.scenePos()
            data_point = self.getImageItem().mapFromScene(pos)

            data_point_x = int(data_point.x())
            data_point_y = int(data_point.y())

            # view_rect = self.getImageItem().viewRect()
            # text_size = self.size_text(view_rect.width(), view_rect.height())
            if self.image.ndim > 2:
                val_point = f"{self.image[data_point_y, data_point_x, :]}"
            else:
                val_point = f"{self.image[data_point_y, data_point_x]}"

            rgb_hex = hex(
                self.getImageItem()
                .getPixmap()
                .toImage()
                .pixel(data_point_x, data_point_y)
            )
            rgb = (
                f"[{int(rgb_hex[4:6], 16)} {int(rgb_hex[6:8], 16)}"
                f" {int(rgb_hex[8:], 16)}]"
            )

            roi = pg.ROI(
                pos=self.pos_of_roi(data_point_x, data_point_y),
                pen=self.contrast_color(rgb_hex),
                size=self.size_of_roi(),
                movable=False,
                removable=True,
            )
            roi.setPen(color=self.contrast_color(rgb_hex), width=4.5)

            roi.setAcceptedMouseButtons(Qt.LeftButton)
            text = PgTextItem(
                html=(
                    f'<span style="font-family: {self.font_family};">'
                    + self.html_of_data_marker_name("PIXEL[X,Y]")
                    + self.html_of_data_marker_value(f"[{data_point_x} {data_point_y}]")
                    + "<br>"
                    + self.html_of_data_marker_name("AXIS [X,Y]")
                    + self.html_of_data_marker_value(
                        f"[{self.axis_x[data_point_x]:6g}"
                        f" {self.axis_y[data_point_y]:6g}]"
                    )
                    + "<br>"
                    + self.html_of_data_marker_name("Value")
                    + self.html_of_data_marker_value(val_point)
                    + "<br>"
                    + self.html_of_data_marker_name("[R,G,B]")
                    + self.html_of_data_marker_value(rgb)
                    + "</span>"
                ),
                border={"color": "000000", "width": 1},
                anchor=(0, 1),
                fill=(250, 250, 255, 255),
            )
            text.sig_change_font_size.connect(self.change_roi_font_size)
            text.setParentItem(roi)
            roi.sigClicked.connect(self.roi_click)
            roi.sigRemoveRequested.connect(self.roi_remove)

            self.addItem(roi)
            event.accept()
        else:
            event.ignore()

    def roi_remove(self, roi):
        """ROI를 제거한다."""
        self.removeItem(roi)

    def roi_click(self, roi, event):
        """클릭된 ROI를 앞에 표시한다."""
        for item in self.view.items:
            if isinstance(item, pg.ROI):
                item.setZValue(0)
        roi.setZValue(1)

    @Slot(int)
    def change_roi_font_size(self, font_size: int):
        """모든 ROI의 PgTextItem 글씨 크기를 변경한다."""
        for roi in self.view.items:
            if isinstance(roi, pg.ROI) is False:
                continue
            for text_item in roi.allChildItems():
                if isinstance(text_item, PgTextItem):
                    font = text_item.textItem.font()
                    font.setPointSize(font_size)
                    text_item.setFont(font)

    def view_all(self):
        """이미지를 전체 뷰로 본다."""
        self.view.autoRange()

    def mouseDoubleClickEvent(self, ev):
        """Double 클릭 시 이미지를 전체 뷰로 본다."""
        self.view.autoRange()

    def toggle_mouse_mode(self):
        """Mouse 왼쪽 클릭 모드를 전환한다."""
        if self.view.vb.state["mouseMode"] == self.view.vb.RectMode:
            self.view.vb.setMouseMode(self.view.vb.PanMode)
        else:
            self.view.vb.setMouseMode(self.view.vb.RectMode)

    def toggle_aspect_mode(self):
        """Toggle aspect mode."""
        if self.view.vb.state["aspectLocked"] is False:
            self.view.vb.setAspectLocked(True)
        else:
            self.view.vb.setAspectLocked(False)

    def set_limit_view(self, x_axis, y_axis):
        """Panning과 Scale의 한계를 지정한다."""
        x_range = max(x_axis) - min(x_axis)
        y_range = max(y_axis) - min(y_axis)
        max_range = max([x_range, y_range])

        self.view.vb.setLimits(
            xMin=min(x_axis) - max_range,
            xMax=max(x_axis) + max_range,
            yMin=min(y_axis) - max_range,
            yMax=max(y_axis) + max_range,
        )


class PgUserAxisItem(pg.AxisItem):
    """입력받은 Axis가 표시되도록 linkedViewChanged을 Overide한다."""

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.offset: float = 0.0  # pixel의 중앙에 axis 값이 표시되도록 보정
        self.delta: float = 1.0
        self.value_of_view_axis_0: float = 0.0

    def set_axis(self, value_of_view_axis_0, delta, scale):
        """Set axis."""
        self.value_of_view_axis_0 = value_of_view_axis_0
        self.delta = delta * scale
        self.offset = -delta / 2

    def custom_range(self, newRange, inverted: bool):
        """Custom Range."""
        value_first = newRange[0] * self.delta + self.value_of_view_axis_0 + self.offset
        value_end = newRange[1] * self.delta + self.value_of_view_axis_0 + self.offset

        if inverted:
            return value_end, value_first
        else:
            return value_first, value_end

    def linkedViewChanged(self, view, newRange=None):
        """Link viewChanged."""
        if self.orientation in ["right", "left"]:
            newRange = view.viewRange()[1]
            self.setRange(*self.custom_range(newRange, view.yInverted()))
        else:
            newRange = view.viewRange()[0]
            self.setRange(*self.custom_range(newRange, view.xInverted()))

    def set_style(self, style=None):
        """PgUserAxisItem의 모양을 설정한다.

        Args:
          style : Dict형태로 모양을 입력한다.
            {'font_family': 'Courier New',
             'label_font_size': 15,
             'tick_font_size': 15,
             'tick_thickness': 2,
             'tickTextOffset': 10}

        """
        if style is None:
            return

        if "tick_thickness" in style:
            self.setPen(pg.getConfigOption("foreground"), width=style["tick_thickness"])
        if "tickTextOffset" in style:
            self.setStyle(tickTextOffset=style["tickTextOffset"])
        if "font_family" in style and "tick_font_size" in style:
            self.setTickFont(QFont(style["font_family"], style["tick_font_size"]))
        elif "font_family" in style:
            self.setTickFont(QFont(style["font_family"]))
        elif "tick_font_size" in style:
            self.setTickFont(QFont("Courier New", style["tick_font_size"]))

        if "font_family" in style and "label_font_size" in style:
            self.label.setFont(QFont(style["font_family"], style["label_font_size"]))
        elif "font_family" in style:
            self.label.setFont(QFont(style["font_family"]))
        elif "label_font_size" in style:
            self.label.setFont(QFont("Courier New", style["label_font_size"]))


def imagescpg(
    *arg,
    cmap: str = "viridis",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    colorbar: bool = False,
    style: Optional[dict] = None,
):
    r"""Implement Imagesc using pyqtgraph.

    return을 받지 않으면 figure 창이 사라진다.

    Args:
      cmap : ['Grey', 'Grey_r', ,'jet', 'parula', 'viridis', ..., None]
        (the default is 'viridis')
      title :  title
      xlabel : xlabel
      ylabel : ylabel
      colorbar : colorbar
      style : dict, optional
        {'font_family': 'Courier New',
         'title_font_size': '15pt',
         'title_bold': True,
         'label_font_size': 15,
         'tick_font_size': 15,
         'tick_thickness': 2,
         'tickTextOffset': 10}

    Returns:
      (pg.ImageView, pg.PlotItem)
          return을 받지 않으면 Figure가 사라진다.

    """
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    pg.setConfigOptions(imageAxisOrder="row-major")

    if len(arg) == 1:
        data = arg[0]
        nr = data.shape[0]
        nc = data.shape[1]
        col_axis = arange(nc)
        row_axis = arange(nr)
    elif len(arg) == 3:
        data = arg[2]
        nr = data.shape[0]
        nc = data.shape[1]
        col_axis = arg[0]
        row_axis = arg[1]
    else:
        print("Plz Check argument of imagescpg")
        return

    if len(col_axis) != nc or len(row_axis) != nr:
        print("check size of axis and data")
        return

    style_default = {
        "font_family": "D2Coding ligature",
        "title_font_size": "16pt",
        "title_bold": True,
        "label_font_size": 14,
        "tick_font_size": 11,
        "tick_thickness": 2,
        "tickTextOffset": 5,
    }

    if style is None:
        style = style_default
    else:
        style_default.update(style)
        style = style_default

    if cmap and data.ndim > 2:
        print("Ignored colormap (data ndim > 2)", file=sys.stderr)
        cmap = ""

    diff_x = col_axis[1] - col_axis[0]
    diff_y = row_axis[1] - row_axis[0]

    if diff_x > diff_y:
        scale = (1.0, abs(diff_y / diff_x))
    else:
        scale = (abs(diff_x / diff_y), 1.0)

    axis_bottom = PgUserAxisItem(orientation="bottom")
    axis_bottom.set_axis(col_axis[0], diff_x, 1.0 / scale[0])
    axis_bottom.set_style(style)

    axis_left = PgUserAxisItem(orientation="left")
    axis_left.set_axis(row_axis[0], diff_y, 1.0 / scale[1])
    axis_left.set_style(style)

    view = pg.PlotItem(
        title=title,
        labels={"left": ylabel, "bottom": xlabel},
        axisItems={"left": axis_left, "bottom": axis_bottom},
    )

    if style and ("title_font_size" in style and "title_bold" in style):
        view.setTitle(title, size=style["title_font_size"], bold=style["title_bold"])
    elif style and "title_bold" in style:
        view.setTitle(title, bold=style["title_bold"])
    elif style and "title_font_size" in style:
        view.setTitle(title, size=style["title_font_size"])

    imv = PgImageViewROI(view=view)

    imv.set_scale(*scale)
    imv.set_axis(col_axis, row_axis)
    imv.toggle_mouse_mode()
    # imv.set_limit_view(col_axis, row_axis)

    imv.show()

    imv.setImage(
        data,
        scale=scale,
        autoHistogramRange=False,
        autoLevels=False,
        levels=(data.min(), data.max()),
    )

    if cmap:
        if cmap in pg.colormap.listMaps():
            colormap = pg.colormap.get(cmap)
        elif cmap in pg.colormap.listMaps("matplotlib"):
            colormap = pg.colormap.get(cmap, source="matplotlib", skipCache=True)
        else:
            colormap = None
        imv.setColorMap(colormap)

    # imv.getImageItem().mouseDoubleClickEvent = imv.func_double_click
    imv.getImageItem().mouseClickEvent = imv.func_shift_left_click

    if colorbar:
        imv.ui.histogram.show()
    else:
        imv.ui.histogram.hide()

    return imv, view


class PlotItemWithMarker(pg.PlotItem):
    """PlotItem에 Marker 기능을 추가한다."""

    def __init__(self, **kargs):
        """Init."""
        super().__init__(**kargs)
        self.colors = [
            "#0072bd",
            "#d95319",
            "#edb120",
            "7e2f8e",
            "#77ac30",
            "#4dbeee",
            "#a2142f",
        ]
        self.color_idx = 0
        self.pen_width = 3
        self.proxy = None  # mouse move event 처리를 위한 proxy
        self.mouse_pos_x = 0
        self.mouse_pos_y = 0

    def plot(self, *args, **kargs):
        """Override plot method of PlotItem."""
        # Plot에 색깔 입력이 되지 않은 경우 색을 설정한다.
        if "pen" not in kargs or kargs["pen"] is None:
            kargs["pen"] = pg.mkPen(self.colors[self.color_idx], width=self.pen_width)
            self.color_idx += 1
            self.color_idx = self.color_idx % len(self.colors)

        return super(PlotItemWithMarker, self).plot(*args, **kargs)

    def mouseDoubleClickEvent(self, ev):
        """Double Click 시 화면 배율을 초기화한다."""
        self.autoRange()
        super().mouseDoubleClickEvent(ev)

    def keyPressEvent(self, ev):
        """Shift+S를 누르면 mouse Mode를 변경한다."""
        key = ev.key()
        shift_pressed = ev.modifiers() & Qt.ShiftModifier
        if key == Qt.Key_S and shift_pressed:
            if self.vb.state["mouseMode"] == self.vb.RectMode:
                self.vb.setMouseMode(self.vb.PanMode)
            else:
                self.vb.setMouseMode(self.vb.RectMode)
        else:
            super().keyPressEvent(ev)

    def mouseMoved(self, ev, label_item):
        """Mouse 이동에 따른 위치 표시."""
        pos = ev[0]  # using signal proxy turns original arguments into a tuple
        if self.sceneBoundingRect().contains(pos):
            mouse_point = self.vb.mapSceneToView(pos)
            self.mouse_pos_x = mouse_point.x()
            self.mouse_pos_y = mouse_point.y()
            label_item.setText(f"x:{self.mouse_pos_x}, y:{self.mouse_pos_y}")

    def set_mouseMoved_proxy(self, label):
        """Mouse 위치를 label에 표시하는 proxy 설정."""
        self.proxy = pg.SignalProxy(
            self.scene().sigMouseMoved,
            rateLimit=60,
            slot=lambda event: self.mouseMoved(event, label),
        )

    def line_clicked(self, plot_data_item):
        """line을 Shift키와 함께 click하면 marker를 표시한다."""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            x = plot_data_item.xData
            y = plot_data_item.yData
            if isinstance(x, ndarray) and isinstance(y, ndarray):
                idx_near = (np.abs(x - self.mouse_pos_x)).argmin()

                # calc radius
                idx_prev = idx_near - 1
                if idx_near == 0:
                    idx_prev = 1

                radius = (
                    min(
                        [abs(x[idx_near] - x[idx_prev]), abs(y[idx_near] - y[idx_prev])]
                    )
                    / 2
                )

                roi = pg.CircleROI(
                    pos=(x[idx_near] - radius, y[idx_near] - radius),
                    radius=radius,
                    movable=False,
                    removable=True,
                )

                roi.setAcceptedMouseButtons(Qt.LeftButton)

                arrow = pg.ArrowItem(angle=90, pos=(radius, radius))
                arrow.setParentItem(roi)

                text = pg.TextItem(
                    html=(
                        '<span style="font-family: D2Conding ligature;">'
                        + f"x {x[idx_near]:g}<br>y {y[idx_near]:g}<br>"
                        + f"idx {idx_near}"
                        + "</span>"
                    ),
                    border={"color": "222222", "width": 1},
                    anchor=(0.5, -0.5),
                    fill=(250, 250, 255, 50),
                )
                text.setParentItem(roi)

                roi.sigClicked.connect(self.roi_click)
                roi.sigRemoveRequested.connect(self.roi_remove)

                self.addItem(roi)

    def roi_remove(self, roi):
        """ROI를 제거한다."""
        self.removeItem(roi)

    def roi_click(self, roi, event):
        """클릭된 ROI를 앞에 표시한다."""
        for item in self.items:
            if isinstance(item, pg.ROI):
                item.setZValue(0)
        roi.setZValue(1)


def plotpg(
    *arg,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    name: str = "",
    pen=None,
    grid=True,
    style=None,
    fig=None,
    ax=None,
):
    """PyQtGraph를 이용하여 선을 그린다.

    Args:
      title: title.
      xlabel: xlabel.
      ylabel: ylabel.
      name: name
      pen: DESCRIPTION. Defaults to None.
      grid: DESCRIPTION. Defaults to True.
      style: dict
        {'font_family': 'D2Coding ligature',
         'title_font_size': '16pt',
         'title_bold': True,
         'label_font_size': '14pt',
         'tick_font_size': 13,
         'tick_thickness': 2,
         'tickTextOffset': 5}
      fig: pg.GraphicsWindow
      ax: PlotItemWithMarker

    Returns:
      pg.GraphicsWindow: fig, DESCRIPTION.
      PlotItemWithMarker: ax, DESCRIPTION.

    """
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    if len(arg) == 1:
        y = array(arg[0]).ravel()
        x = arange(len(y))
    elif len(arg) == 2:
        y = array(arg[1]).ravel()
        x = array(arg[0]).ravel()
    else:
        print("Plz Check argument of plotqt")
        return

    style_default = {
        "font_family": "D2Coding ligature",
        "title_font_size": "16pt",
        "title_bold": True,
        "label_font_size": "14pt",
        "tick_font_size": 13,
        "tick_thickness": 2,
        "tickTextOffset": 5,
    }

    if style is None:
        style = style_default
    else:
        style_default.update(style)
        style = style_default

    if ax is not None:
        curve = ax.plot(x=x, y=y, name=name, pen=pen, clickable=True)
        curve.curve.setClickable(True)
        curve.sigClicked.connect(ax.line_clicked)
        return fig, ax

    fig = pg.GraphicsWindow()

    ax = PlotItemWithMarker()
    fig.addItem(ax, 0, 0)

    ax.addLegend()
    curve = ax.plot(x=x, y=y, name=name, pen=pen, clickable=True)
    curve.curve.setClickable(True)
    curve.sigClicked.connect(ax.line_clicked)

    label = pg.LabelItem(justify="right")
    fig.addItem(label, 1, 0)

    ax.setTitle(title, size=style["title_font_size"], bold=style["title_bold"])

    labelStyle = {
        "font-family": style["font_family"],
        "font-size": style["label_font_size"],
    }
    ax.setLabel("left", ylabel, **labelStyle)
    ax.setLabel("bottom", xlabel, **labelStyle)

    font = QFont(style["font_family"], style["tick_font_size"])
    ax.getAxis("bottom").tickFont = font
    ax.getAxis("bottom").setStyle(tickTextOffset=style["tickTextOffset"])
    ax.getAxis("bottom").setPen(
        pg.getConfigOption("foreground"), width=style["tick_thickness"]
    )

    ax.getAxis("left").tickFont = font
    ax.getAxis("left").setStyle(tickTextOffset=style["tickTextOffset"])
    ax.getAxis("left").setPen(
        pg.getConfigOption("foreground"), width=style["tick_thickness"]
    )

    if grid is True:
        ax.showGrid(x=True, y=True, alpha=0.1)

    ax.set_mouseMoved_proxy(label)

    return fig, ax


if __name__ == "__main__":
    from mkl_random import randn

    QAPP = pg.mkQApp()

    data = randn(2048, 2048)
    x = arange(data.shape[1]) * 5
    y = arange(data.shape[0]) * 1
    imv_gray0 = imagescpg(
        x,
        y,
        data,
        cmap="viridis",
        title="Gray Image with gray Colormap",
        xlabel="Column",
        ylabel="Row",
        colorbar=True,
    )

    QAPP.exec()
