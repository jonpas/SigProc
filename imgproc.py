#!/usr/bin/env python3

import sys
import os
import numpy as np
import cv2
import scipy as sp
from scipy import signal
from scipy.ndimage import morphology
from skimage.exposure import rescale_intensity
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIntValidator


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.ax = None

        self.orig_img = None
        self.img = None

        self.initUI()

    def initUI(self):
        spacer = QSpacerItem(50, 0, QSizePolicy.Minimum)
        spacer_small = QSpacerItem(10, 0, QSizePolicy.Minimum)

        # File selector
        lbl_file = QLabel("File:")
        self.txt_file = QLineEdit()
        self.txt_file.setPlaceholderText("Select file ...")
        btn_file = QPushButton("Select")
        btn_file.clicked.connect(self.show_open_dialog)

        # Save
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.show_save_dialog)

        # Reset
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Show originally loaded image (reset all modifications)")
        self.btn_reset.clicked.connect(lambda: self.plot_image(self.orig_img))

        # Histogram
        self.btn_hist = QPushButton("Histogram")
        self.btn_hist.setToolTip("Draw histogram of current image")
        self.btn_hist.clicked.connect(self.histogram)

        # Graph space
        self.figure = Figure()
        FigureCanvas(self.figure)
        self.figure.canvas.setMinimumHeight(300)

        # Conversion to Grayscale
        self.cb_gray = QComboBox()
        self.cb_gray.setToolTip("Grayscale conversion method")
        self.cb_gray.addItems(["Average", "Red", "Green", "Blue"])
        self.btn_gray = QPushButton("Grayscale")
        self.btn_gray.setToolTip("Convert loaded image to grayscale image")
        self.btn_gray.clicked.connect(lambda: self.grayscale(self.cb_gray.currentIndex() - 1))

        # Segmentation / Binarization
        self.segment_thresh = QLineEdit()
        self.segment_thresh.setText("100")
        self.segment_thresh.setToolTip("Segmentation threshold")
        self.segment_thresh.setMaximumWidth(30)
        self.segment_thresh.setValidator(QIntValidator(0, 255))
        self.btn_segment = QPushButton("Binarize")
        self.btn_segment.setToolTip("Convert loaded image to binary image using segmentation")
        self.btn_segment.clicked.connect(lambda: self.binarize(int(self.segment_thresh.text())))

        # Graph toolbar
        self.plotnav = NavigationToolbar(self.figure.canvas, self.figure.canvas)
        self.plotnav.setStyleSheet("QToolBar { border: 0px }")
        self.plotnav.setOrientation(Qt.Vertical)

        # Image processing implementation
        self.cb_imgproc_impl = QComboBox()
        self.cb_imgproc_impl.setToolTip("Processing implementation")
        self.cb_imgproc_impl.addItems(["OpenCV", "SciPy", "Manual"])

        # Smooth / Blur
        self.smooth_intensity = QLineEdit()
        self.smooth_intensity.setText("5")
        self.smooth_intensity.setToolTip("Smooth intensity (must at least 3 and odd)")
        self.smooth_intensity.setMaximumWidth(30)
        self.smooth_intensity.setValidator(QIntValidator(0, 255))
        self.btn_smooth = QPushButton("Smooth")
        self.btn_smooth.setToolTip("Smooth (blur) current image")
        self.btn_smooth.clicked.connect(lambda: self.smooth(int(self.smooth_intensity.text())))

        # Sharpen
        self.sharpen_intensity = QLineEdit()
        self.sharpen_intensity.setText("5")
        self.sharpen_intensity.setToolTip("Sharpen intensity (must be at least 5)")
        self.sharpen_intensity.setMaximumWidth(30)
        self.sharpen_intensity.setValidator(QIntValidator(0, 255))
        self.btn_sharpen = QPushButton("Sharpen")
        self.btn_sharpen.setToolTip("Sharpen current image")
        self.btn_sharpen.clicked.connect(lambda: self.sharpen(int(self.sharpen_intensity.text())))

        # Edge detection
        self.edge_intensity = QLineEdit()
        self.edge_intensity.setText("4")
        self.edge_intensity.setToolTip("Edge detection intensity (must be at least 4)")
        self.edge_intensity.setMaximumWidth(30)
        self.edge_intensity.setValidator(QIntValidator(0, 255))
        self.btn_edge = QPushButton("Detect Edges")
        self.btn_edge.setToolTip("Detect edges on current image")
        self.btn_edge.clicked.connect(lambda: self.detect_edges(int(self.edge_intensity.text())))

        # Dilate
        self.dilate_intensity = QLineEdit()
        self.dilate_intensity.setText("5")
        self.dilate_intensity.setToolTip("Dilation intensity (must be at least 5)")
        self.dilate_intensity.setMaximumWidth(30)
        self.dilate_intensity.setValidator(QIntValidator(0, 255))
        self.btn_dilate = QPushButton("Dilate")
        self.btn_dilate.setToolTip("Dilate current image")
        self.btn_dilate.clicked.connect(lambda: self.dilate(int(self.dilate_intensity.text())))

        # Erode
        self.erode_intensity = QLineEdit()
        self.erode_intensity.setText("5")
        self.erode_intensity.setToolTip("Erosion intensity (must be at least 5)")
        self.erode_intensity.setMaximumWidth(30)
        self.erode_intensity.setValidator(QIntValidator(0, 255))
        self.btn_erode = QPushButton("Erode")
        self.btn_erode.setToolTip("Erode current image")
        self.btn_erode.clicked.connect(lambda: self.erode(int(self.erode_intensity.text())))

        # Layout
        hbox_top = QHBoxLayout()
        hbox_top.addWidget(lbl_file)
        hbox_top.addWidget(self.txt_file)
        hbox_top.addWidget(btn_file)
        hbox_top.addWidget(self.btn_save)
        hbox_top.addWidget(self.btn_reset)
        hbox_top.addStretch()
        hbox_top.addSpacerItem(spacer)
        hbox_top.addWidget(self.btn_hist)
        hbox_top.addStretch()
        hbox_top.addSpacerItem(spacer)
        hbox_top.addWidget(self.cb_gray)
        hbox_top.addWidget(self.btn_gray)
        hbox_top.addSpacerItem(spacer_small)
        hbox_top.addWidget(self.segment_thresh)
        hbox_top.addWidget(self.btn_segment)

        hbox_bot = QHBoxLayout()
        hbox_bot.addWidget(self.cb_imgproc_impl)
        hbox_bot.addStretch()
        hbox_bot.addSpacerItem(spacer)
        hbox_bot.addWidget(self.smooth_intensity)
        hbox_bot.addWidget(self.btn_smooth)
        hbox_bot.addWidget(self.sharpen_intensity)
        hbox_bot.addWidget(self.btn_sharpen)
        hbox_bot.addWidget(self.edge_intensity)
        hbox_bot.addWidget(self.btn_edge)
        hbox_bot.addStretch()
        hbox_bot.addSpacerItem(spacer)
        hbox_bot.addWidget(self.dilate_intensity)
        hbox_bot.addWidget(self.btn_dilate)
        hbox_bot.addWidget(self.erode_intensity)
        hbox_bot.addWidget(self.btn_erode)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_top)
        vbox.addWidget(self.figure.canvas)
        vbox.addLayout(hbox_bot)

        self.update_ui()

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
        self.btn_reset.setDisabled(block_general)
        self.btn_hist.setDisabled(block_general)
        self.btn_gray.setDisabled(block_general)
        self.btn_segment.setDisabled(block_general)
        self.btn_smooth.setDisabled(block_general)
        self.btn_sharpen.setDisabled(block_general)
        self.btn_dilate.setDisabled(block_general)
        self.btn_erode.setDisabled(block_general)
        self.btn_edge.setDisabled(block_general)

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
        self.figure.clear()
        self.ax = self.figure.add_subplot(1, 1, 1)

    def plot_image(self, img):
        self.reset_plot()
        self.ax.axis("off")

        self.ax.imshow(img, cmap='gray' if len(img.shape) < 3 else None)
        self.figure.canvas.draw()

        self.img = img

    # Draw histogram of current image
    def histogram(self):
        self.reset_plot()
        self.ax.margins(0)

        # Plot each channel on RGB image or only first channel on grayscale image
        colors = ('r', 'g', 'b') if len(self.img.shape) > 2 else ('b',)
        for i, color in enumerate(colors):
            hist = cv2.calcHist([self.img], [i], None, [256], [0, 256])
            self.ax.plot(hist, color=color)

        self.figure.canvas.draw()

    # Convert current image to grayscale
    def grayscale(self, type=-1):  # -1 - Average, 0 - Red, 1 - Green, 2 - Blue
        # Do nothing if already grayscale
        if len(self.img.shape) < 3:
            return self.img

        if type < 0:
            # Convert to grayscale by averaging all channels
            img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        else:
            # Convert to grayscale by taking one channel
            img_gray = self.img[:, :, type]

        self.plot_image(img_gray)
        return img_gray

    # Binarize current image
    def binarize(self, threshold=0):
        # Make sure we are operating on grayscale image (applied to original image)
        self.grayscale()
        _, img_bin = cv2.threshold(self.img, threshold, 255, cv2.THRESH_BINARY_INV)

        self.plot_image(img_bin)
        return img_bin

    # Get convolution implementation from combo box (lower-case text)
    def get_imgproc_impl(self):
        return self.cb_imgproc_impl.currentText().lower()

    # Convolve current image
    def convolve2d(self, kernel):
        imgproc = self.get_imgproc_impl()
        if imgproc == "opencv":
            # OpenCV filter2D
            return cv2.filter2D(self.img, -1, kernel)
        elif imgproc == "scipy":
            # SciPy convolve2d
            if len(self.img.shape) < 3:
                # Grayscale
                return signal.convolve2d(self.img, kernel, mode="same", boundary="symm")
            else:
                # Color - convolve each channel
                img_conv = []
                for ch in range(self.img.shape[2]):
                    img_conv_ch = signal.convolve2d(self.img[:, :, ch], kernel, mode="same", boundary="symm")
                    img_conv.append(img_conv_ch)

                # Stack channels, clip them to [0, 255] and represent as original image (prevent invalid range)
                return np.clip(np.stack(img_conv, axis=2), 0, 255).astype(self.img.dtype)
        elif imgproc == "manual":
            # Manual convolve
            # Get spatial dimensions of the image and kernel
            (img_h, img_w) = self.img.shape[:2]
            (kern_h, kern_w) = kernel.shape[:2]

            # Pad border
            pad = int((kern_w - 1) / 2)
            self.img = cv2.copyMakeBorder(self.img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

            # Slide the kernel over the image from left to right and top to bottom
            if len(self.img.shape) < 3:
                # Grayscale
                img_conv = np.zeros((img_h, img_w))
                for y in np.arange(pad, img_h + pad):
                    for x in np.arange(pad, img_w + pad):
                        # Extract region of interest (ROI) of the image by extracting the center region
                        roi = self.img[y - pad:y + pad + 1, x - pad:x + pad + 1]
                        # Perform convolution (element-wise multiplication between ROI and kernel and sum of matrix)
                        k = (roi * kernel).sum()
                        # Store convolved value in the current coordinate
                        img_conv[y - pad, x - pad] = k

                # Rescale convolved image to be in range [0, 255]
                return rescale_intensity(img_conv, in_range=(0, 255)) * 255
            else:
                # Color - convolve each channel
                img_conv = []
                for ch in range(self.img.shape[2]):
                    img_ch = self.img[:, :, ch]

                    img_conv_ch = np.zeros((img_h, img_w))
                    for y in np.arange(pad, img_h + pad):
                        for x in np.arange(pad, img_w + pad):
                            # Extract region of interest (ROI) of the image by extracting the center region
                            roi = img_ch[y - pad:y + pad + 1, x - pad:x + pad + 1]
                            # Perform convolution (element-wise multiplication between ROI and kernel and sum of matrix)
                            k = (roi * kernel).sum()
                            # Store convolved value in the current coordinate
                            img_conv_ch[y - pad, x - pad] = k

                    # Rescale convolved image to be in range [0, 255]
                    img_conv_ch = rescale_intensity(img_conv_ch, in_range=(0, 255)) * 255
                    img_conv.append(img_conv_ch)

                # Stack channels, clip them to [0, 255] and represent as original image (prevent invalid range)
                return np.clip(np.stack(img_conv, axis=2), 0, 255).astype(self.img.dtype)

        print("Error! Unknown image processing implementation!")
        return self.img

    # Smooth (blur) current image
    def smooth(self, intensity=5):
        if intensity < 3 or intensity % 2 == 0:
            print("Error! Smooth intensity should be at least 3 and an odd integer!")

        kernel = np.ones((intensity, intensity)) / intensity**2
        img_smooth = self.convolve2d(kernel)

        self.plot_image(img_smooth)
        return img_smooth

    # Sharpen current image
    def sharpen(self, intensity=5):
        if intensity < 5:
            print("Warning! Sharpen intensity should be at least 5! Defaulting to 5!")

        kernel = np.array((
            [0, -1, 0],
            [-1, max(intensity, 5), -1],
            [0, -1, 0]))
        img_sharp = self.convolve2d(kernel)

        self.plot_image(img_sharp)
        return img_sharp

    # Detect edges on current image
    def detect_edges(self, intensity=5):
        if intensity < 4:
            print("Warning! Edge detection intensity should be at least 4! Defaulting to 4!")

        kernel = np.array((
            [0, 1, 0],
            [1, -max(intensity, 4), 1],
            [0, 1, 0]))
        img_edges = self.convolve2d(kernel)

        self.plot_image(img_edges)
        return img_edges

    # Dilate current image
    def dilate(self, intensity=5):
        kernel = np.ones((intensity, intensity))

        imgproc = self.get_imgproc_impl()
        if imgproc == "opencv":
            img_dilate = cv2.dilate(self.img, kernel)
        elif imgproc == "scipy":
            if len(self.img.shape) < 3:
                # Grayscale
                img_dilate = morphology.grey_dilation(self.img, structure=kernel)
            else:
                # Color - erode each channel
                # TODO Make work correctly (green pixels appear)
                img_dilate = []
                for ch in range(self.img.shape[2]):
                    img_dilate_ch = morphology.grey_dilation(
                        self.img[:, :, ch], structure=kernel).astype(self.img.dtype)
                    img_dilate.append(img_dilate_ch)

                # Stack channels, clip them to [0, 255] and represent as original image (prevent invalid range)
                img_dilate = np.clip(np.stack(img_dilate, axis=2), 0, 255).astype(self.img.dtype)
        elif imgproc == "manual":
            print("manual")
            img_dilate = self.img
        else:
            print("Error! Unknown image processing implementation!")
            img_dilate = self.img

        self.plot_image(img_dilate)
        return img_dilate

    # Erode current image
    def erode(self, intensity=5):
        kernel = np.ones((intensity, intensity))

        imgproc = self.get_imgproc_impl()
        if imgproc == "opencv":
            img_erode = cv2.erode(self.img, kernel)
        elif imgproc == "scipy":
            if len(self.img.shape) < 3:
                # Grayscale
                img_erode = morphology.grey_erosion(self.img, structure=kernel)
            else:
                # Color - erode each channel
                # TODO Make work correctly (green pixels appear)
                img_erode = []
                for ch in range(self.img.shape[2]):
                    img_erode_ch = morphology.grey_erosion(self.img[:, :, ch], structure=kernel).astype(self.img.dtype)
                    img_erode.append(img_erode_ch)

                # Stack channels, clip them to [0, 255] and represent as original image (prevent invalid range)
                img_erode = np.clip(np.stack(img_erode, axis=2), 0, 255).astype(self.img.dtype)
        elif imgproc == "manual":
            print("manual")
            img_erode = self.img
        else:
            print("Error! Unknown image processing implementation!")
            img_erode = self.img

        self.plot_image(img_erode)
        return img_erode


if __name__ == "__main__":
    # Create Qt application with window
    app = QApplication(sys.argv)
    main_win = MainWindow()

    # Execute application (blocking)
    app.exec_()

    sys.exit(0)
