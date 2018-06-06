#!/usr/bin/env python3

import sys
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

        # Image space
        self.canvas = QLabel()
        self.canvas.setStyleSheet("QLabel { background-color: white; }")

        # Layout
        hbox_top = QHBoxLayout()
        hbox_top.addWidget(lbl_file)
        hbox_top.addWidget(self.txt_file)
        hbox_top.addWidget(btn_file)
        hbox_top.addWidget(self.btn_save)
        hbox_top.addStretch()

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_top)
        vbox.addWidget(self.canvas)

        # Window
        self.setLayout(vbox)
        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle("Signal Processor - Image")
        self.show()

    def update_ui(self):
        self.btn_save.setDisabled(not self.is_image_loaded())

    def show_open_dialog(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", filter="Image (*.png *.jpg *.bmp)")
        if fname[0] and self.load_image(fname[0]):
            self.txt_file.setText(fname[0])

    def show_save_dialog(self):
        fname = QFileDialog.getSaveFileName(self, "Save file", filter="Image (*.png *.jpg *.bmp)")
        if fname[0] and self.is_image_loaded():
            try:
                self.img.save(fname[0])
            except exceptions.CouldntEncodeError:
                print("Failed to save image!")
            else:
                self.txt_file.setText(fname[0])

    def load_image(self, file):
        try:
            self.img = QPixmap(file)
            self.canvas.setPixmap(self.img.scaled(self.canvas.width(), self.canvas.height(), Qt.KeepAspectRatio))
        except exceptions.CouldntDecodeError:
            print("Failed to load image!")
            return False
        else:
            self.update_ui()
            return True

    def is_image_loaded(self):
        return self.img is not None


if __name__ == "__main__":
    # Create Qt application with window
    app = QApplication(sys.argv)
    main_win = MainWindow()

    # Execute application (blocking)
    app.exec_()

    sys.exit(0)
