#!/usr/bin/env python3

"""
Requirements:
- numpy, pydub, matplotlib, PyQt5 Python packages
- ffmpeg installed on system
"""

import sys
import numpy as np
import pyaudio
from pydub import AudioSegment, exceptions
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

PyAudio = pyaudio.PyAudio


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.sound = None
        self.signal = None

        self.subplots = []
        self.lines_click = []
        self.lines_over = []
        self.line_over = None

        self.initUI()

    def initUI(self):
        # File selector
        lbl_file = QLabel("File:")
        self.txt_file = QLineEdit()
        self.txt_file.setPlaceholderText("Select file ...")
        btn_file = QPushButton("Select")
        btn_file.clicked.connect(self.show_file_dialog)

        spacer = QSpacerItem(100, 0, QSizePolicy.Minimum)

        # Audio controls
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setDisabled(True)

        self.btn_play = QPushButton("Play")
        self.btn_play.setDisabled(True)
        self.btn_play.clicked.connect(self.sound_play)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setDisabled(True)

        # Graph space
        self.figure = Figure()
        FigureCanvas(self.figure)
        self.figure.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_plot_over)

        # Layout
        hbox_file = QHBoxLayout()
        hbox_file.addWidget(lbl_file)
        hbox_file.addWidget(self.txt_file)
        hbox_file.addWidget(btn_file)
        hbox_file.addSpacerItem(spacer)
        hbox_file.addWidget(self.btn_pause)
        hbox_file.addWidget(self.btn_play)
        hbox_file.addWidget(self.btn_stop)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_file)
        vbox.addWidget(self.figure.canvas)

        # Window
        self.setLayout(vbox)
        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle("Audio Visualizer")
        self.show()

    def update_ui(self, block):
        self.btn_pause.setDisabled(block)
        self.btn_play.setDisabled(block)
        self.btn_stop.setDisabled(block)

    def show_file_dialog(self):
        fname = QFileDialog.getOpenFileName(self, "Select file")
        if fname[0]:
            self.txt_file.setText(fname[0])
            self.load_signal(fname[0])
            if self.is_signal_loaded():
                self.plot(self.signal, self.sound)
        else:
            self.txt_file.setText("")

    def load_signal(self, file):
        try:
            self.sound = AudioSegment.from_file(file)
            signal = self.sound.get_array_of_samples()
            self.signal = np.array(signal)
        except exceptions.CouldntDecodeError:
            print("Failed to load signal!")

        self.update_ui(self.signal is None)

    def is_signal_loaded(self):
        return self.sound is not None and self.signal is not None

    def plot(self, signal, sound):
        self.figure.clear()
        self.subplots = []
        self.lines_click = []
        self.lines_over = []

        # X axis as time in seconds
        time = np.linspace(0, sound.duration_seconds, num=len(signal))

        for i in range(0, sound.channels):
            ax = self.figure.add_subplot(sound.channels, 1, i + 1)
            samples = sound.get_array_of_samples()  # [samp1L, samp1R, samp2L, samp2R]
            # Plot current channel, slicing it away
            ax.plot(time[i::sound.channels], samples[i::sound.channels])
            ax.margins(x=0)

            # Hide X axis on all but last channel
            if i + 1 < sound.channels:
                ax.get_xaxis().set_visible(False)
            # Display Y label somewhere in the middle
            if i == int(sound.channels / 2) + 1:
                ax.set_ylabel("Amplitude")
            self.subplots.append(ax)

        self.figure.subplots_adjust(hspace=.0)
        ax.set_xlabel("Time (s)")
        self.figure.canvas.draw()

    def on_plot_click(self, event):
        for line in self.lines_click:
            line.remove()
        self.lines_click = []
        self.figure.canvas.draw_idle()  # TODO Still too slow
        if event.xdata is not None and event.ydata is not None:
            for ax in self.subplots:
                line = ax.axvline(event.xdata, linewidth=1, color="black")
                self.lines_click.append(line)
                self.plot_update(line, ax)

    def on_plot_over(self, event):
        # Remove vertical lines
        for line in self.lines_over:
            line.remove()
        self.lines_over = []
        self.figure.canvas.draw_idle()  # TODO Still too slow
        if event.xdata is not None and event.ydata is not None:
            # Add vertical lines
            for ax in self.subplots:
                line = ax.axvline(event.xdata, linewidth=1, color="grey")
                self.lines_over.append(line)
                self.plot_update(line, ax)

    def plot_update(self, element, ax):
        ax.draw_artist(element)
        self.figure.canvas.blit(ax.bbox)

    def sound_play(self):
        if self.is_signal_loaded():
            p = PyAudio()
            stream = p.open(
                format=p.get_format_from_width(self.sound.sample_width),
                channels=self.sound.channels,
                rate=self.sound.frame_rate,
                output=True)
            stream.write(self.signal)
            stream.close()
            p.terminate()


if __name__ == "__main__":
    # Create Qt application with window
    app = QApplication(sys.argv)
    main_win = MainWindow()

    # Execute application (blocking)
    app.exec_()

    sys.exit(0)
