#!/usr/bin/env python3

import sys
import os
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.ax = None

        self.orig_img = None
        self.img = None

        self.initUI()

    def initUI(self):
        spacer = QSpacerItem(50, 0, QSizePolicy.Minimum)

        # File selector
        lbl_file = QLabel("File:")
        self.txt_file = QLineEdit()
        self.txt_file.setPlaceholderText("Select file ...")
        btn_file = QPushButton("Select")
        btn_file.clicked.connect(self.show_open_dialog)

        self.btn_show = QPushButton("Show")
        self.btn_show.setToolTip("Show originally loaded image (reset all modifications)")
        self.btn_show.setDisabled(True)
        self.btn_show.clicked.connect(lambda: self.plot_image(self.orig_img))

        # Save
        self.btn_save = QPushButton("Save")
        self.btn_save.setDisabled(True)
        self.btn_save.clicked.connect(self.show_save_dialog)

        # Graph space
        self.figure = Figure()
        FigureCanvas(self.figure)
        self.figure.canvas.setMinimumHeight(400)

        # Graph toolbar
        self.plotnav = NavigationToolbar(self.figure.canvas, self.figure.canvas)
        self.plotnav.setStyleSheet("QToolBar { border: 0px }")
        self.plotnav.setOrientation(Qt.Vertical)

        # Histogram
        self.btn_hist = QPushButton("Histogram")
        self.btn_hist.setToolTip("Draw histogram of current image")
        self.btn_hist.setDisabled(True)
        self.btn_hist.clicked.connect(self.histogram)

        # Conversion to Grayscale
        self.btn_gray = QPushButton("Grayscale")
        self.btn_gray.setToolTip("Convert loaded image to grayscale image")
        self.btn_gray.setDisabled(True)
        self.btn_gray.clicked.connect(self.grayscale)

        # Layout
        hbox_top = QHBoxLayout()
        hbox_top.addWidget(lbl_file)
        hbox_top.addWidget(self.txt_file)
        hbox_top.addWidget(btn_file)
        hbox_top.addWidget(self.btn_show)
        hbox_top.addWidget(self.btn_save)
        hbox_top.addStretch()
        hbox_top.addSpacerItem(spacer)

        hbox_bot = QHBoxLayout()
        hbox_bot.addWidget(self.btn_hist)
        hbox_bot.addStretch()
        hbox_bot.addWidget(self.btn_gray)
        hbox_bot.addStretch()

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_top)
        vbox.addWidget(self.figure.canvas)
        vbox.addLayout(hbox_bot)

        # Window
        self.setLayout(vbox)
        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle("Signal Processor - Image")
        self.show()

    # Overriden resize event
    def resizeEvent(self, resizeEvent):
        self.plotnav.move(self.width() - 55, 0)

    def update_ui(self):
        block_general = not self.is_image_loaded()

        self.btn_save.setDisabled(block_general)
        self.btn_show.setDisabled(block_general)
        self.btn_hist.setDisabled(block_general)
        self.btn_gray.setDisabled(block_general)

    def show_open_dialog(self):
        fname, ext = QFileDialog.getOpenFileName(self, "Open file", filter="Image (*.png *.jpg *.bmp)")
        if fname and self.load_image(fname):
            self.txt_file.setText(fname)

    def show_save_dialog(self):
        fname, ext = QFileDialog.getSaveFileName(self, "Save file", filter="Image (*.png *.jpg *.bmp)")
        if fname and self.is_image_loaded():
            # Save as PNG if not set
            if '.' not in fname:
                fname += ".png"

            cv2.imwrite(fname, self.img)
            self.txt_file.setText(fname)

    def load_image(self, file):
        if not os.path.isfile(file):
            return False

        # Read image and convert from BGR (OpenCV default) to RGB
        self.orig_img = cv2.imread(file)
        self.orig_img = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB)
        self.img = self.orig_img

        self.plot_image(self.orig_img)
        self.update_ui()
        return True

    def is_image_loaded(self):
        return self.img is not None

    def reset_plot(self):
        for ax in self.figure.axes:
            self.figure.delaxes(ax)
        self.ax = self.figure.add_subplot(1, 1, 1)

    def plot_image(self, img, gray=False):
        self.reset_plot()
        self.ax.axis("off")
        self.ax.imshow(img, cmap='gray' if gray else None)
        self.figure.canvas.draw()

        self.img = img

    def histogram(self):
        self.reset_plot()

        # Plot each channel on RGB image or only first channel on grayscale image
        colors = ('r', 'g', 'b') if len(self.img.shape) > 2 else ('b',)
        for i, color in enumerate(colors):
            hist = cv2.calcHist([self.img], [i], None, [256], [0, 256])
            self.ax.plot(hist, color=color)

        self.figure.canvas.draw()

    def grayscale(self):
        self.img = cv2.cvtColor(self.orig_img, cv2.COLOR_RGB2GRAY)
        self.plot_image(self.img, gray=True)


if __name__ == "__main__":
    # Create Qt application with window
    app = QApplication(sys.argv)
    main_win = MainWindow()

    # Execute application (blocking)
    app.exec_()

    sys.exit(0)
