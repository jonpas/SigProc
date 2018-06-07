#!/usr/bin/env python3

import sys
import os
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.img = None

        self.initUI()

    def initUI(self):
        # File selector
        lbl_file = QLabel("File:")
        self.txt_file = QLineEdit()
        self.txt_file.setPlaceholderText("Select file ...")
        btn_file = QPushButton("Select")
        btn_file.clicked.connect(self.show_open_dialog)

        # Save
        self.btn_save = QPushButton("Save")
        self.btn_save.setDisabled(True)
        self.btn_save.clicked.connect(self.show_save_dialog)

        # Graph space
        self.figure = Figure()
        FigureCanvas(self.figure)
        self.figure.canvas.setMinimumHeight(400)
        self.subplot = self.figure.add_subplot(1, 1, 1)
        self.subplot.axis("off")

        # Graph toolbar
        self.plotnav = NavigationToolbar(self.figure.canvas, self.figure.canvas)
        self.plotnav.setStyleSheet("QToolBar { border: 0px }")
        self.plotnav.setOrientation(Qt.Vertical)

        # Layout
        hbox_top = QHBoxLayout()
        hbox_top.addWidget(lbl_file)
        hbox_top.addWidget(self.txt_file)
        hbox_top.addWidget(btn_file)
        hbox_top.addWidget(self.btn_save)
        hbox_top.addStretch()

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_top)
        vbox.addWidget(self.figure.canvas)

        # Window
        self.setLayout(vbox)
        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle("Signal Processor - Image")
        self.show()

    # Overriden resize event
    def resizeEvent(self, resizeEvent):
        self.plotnav.move(self.width() - 55, 0)

    def update_ui(self):
        self.btn_save.setDisabled(not self.is_image_loaded())

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

        self.img = cv2.imread(file)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.plot(self.img)
        self.update_ui()
        return True

    def is_image_loaded(self):
        return self.img is not None

    def plot(self, img):
        self.subplot.imshow(img)
        self.figure.canvas.draw()


if __name__ == "__main__":
    # Create Qt application with window
    app = QApplication(sys.argv)
    main_win = MainWindow()

    # Execute application (blocking)
    app.exec_()

    sys.exit(0)
