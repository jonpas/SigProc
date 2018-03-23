#!/usr/bin/env python3

import sys
import itertools
import numpy as np
import pyaudio
from pydub import AudioSegment, exceptions
from pydub.utils import make_chunks
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt5.QtWidgets import *

PyAudio = pyaudio.PyAudio


class MainWindow(QWidget):
    sig_sound_play_at = pyqtSignal(float)
    sig_sound_pause = pyqtSignal()
    sig_sound_stop = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.sound = None
        self.signal = None

        self.plotbackground = None

        self.sound_thread = None
        self.sound_paused = False
        self.sound_start_at = 0

        self.initUI()

    def initUI(self):
        spacer = QSpacerItem(50, 0, QSizePolicy.Minimum)

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

        # Audio controls
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setDisabled(True)
        self.btn_pause.clicked.connect(self.sound_pause)
        self.sound_mutex = QMutex()
        self.sound_pause_cond = QWaitCondition()

        self.btn_play = QPushButton("Play")
        self.btn_play.setDisabled(True)
        self.btn_play.clicked.connect(self.sound_play)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setDisabled(True)
        self.btn_stop.clicked.connect(self.sound_stop)

        # Doppler Shift simulation
        cb_source_speed = QComboBox()
        cb_source_speed.setToolTip("Source speed")
        cb_source_speed.addItems(["20 km/h", "50 km/h", "100 km/h", "250 km/h", "500 km/h"])
        self.source_speeds = [20, 50, 100, 250, 500]  # Same indexes as text above
        btn_doppler = QPushButton("Simulate Doppler Shift")

        # Effects
        btn_effect = QPushButton("Apply Effect")
        # self.btn_effects.clicked.connect(self.effect_select)

        # Recording
        cb_sample_rate = QComboBox()
        cb_sample_rate.setToolTip("Sampling rate")
        cb_sample_rate.addItems(["8.000 Hz", "11.025 Hz", "22.050 Hz", "44.100 Hz"])
        self.sampling_rates = [8000, 11025, 22050, 44100]  # Same indexes as text above
        btn_record = QPushButton("Record")
        # self.btn_record.clicked.connect(self.sound_record)

        cb_bit_depth = QComboBox()
        cb_bit_depth.setToolTip("Bit depth")
        cb_bit_depth.addItems(["8 b", "16 b"])
        self.bit_depths = [pyaudio.paInt8, pyaudio.paInt16]  # Same indexes as text above

        # Graph space
        self.figure = Figure()
        FigureCanvas(self.figure)
        self.figure.canvas.setMinimumHeight(400)
        self.figure.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_plot_over)

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
        hbox_top.addSpacerItem(spacer)
        hbox_top.addWidget(self.btn_pause)
        hbox_top.addWidget(self.btn_play)
        hbox_top.addWidget(self.btn_stop)

        hbox_bot = QHBoxLayout()
        hbox_bot.addWidget(cb_source_speed)
        hbox_bot.addWidget(btn_doppler)
        hbox_bot.addStretch()
        hbox_bot.addSpacerItem(spacer)
        hbox_bot.addWidget(btn_effect)
        hbox_bot.addStretch()
        hbox_bot.addSpacerItem(spacer)
        hbox_bot.addWidget(cb_sample_rate)
        hbox_bot.addWidget(cb_bit_depth)
        hbox_bot.addWidget(btn_record)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_top)
        vbox.addWidget(self.figure.canvas)
        vbox.addLayout(hbox_bot)

        # Window
        self.setLayout(vbox)
        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle("Signal Processor")
        self.show()

    # Overriden resize event
    def resizeEvent(self, resizeEvent):
        if self.is_signal_loaded():
            self.on_plot_change(None)
        self.plotnav.move(self.width() - 55, 0)

    def update_ui(self, block):
        self.btn_save.setDisabled(not self.is_signal_loaded())
        self.btn_pause.setDisabled(not block)
        self.btn_pause.setText("Resume" if self.sound_paused else "Pause")
        self.btn_play.setDisabled(block)
        self.btn_stop.setDisabled(not block)
        self.plotnav.setDisabled(self.is_sound_playing())

    def show_open_dialog(self):
        fname = QFileDialog.getOpenFileName(self, "Select file")
        if fname[0]:
            if self.load_signal(fname[0]):
                self.txt_file.setText(fname[0])
                self.plot(self.signal, self.sound)

    def show_save_dialog(self):
        fname = QFileDialog.getSaveFileName(self, "Save file")
        if fname[0]:
            if self.is_signal_loaded():
                ext = fname[0].rsplit(".", 1)[-1]
                try:
                    self.sound.export(fname[0], format=ext)
                except exceptions.CouldntEncodeError:
                    print("Failed to save signal!")
                else:
                    self.txt_file.setText(fname[0])

    def load_signal(self, file):
        self.sound_stop()

        try:
            self.sound = AudioSegment.from_file(file)
            self.signal = np.array(self.sound.get_array_of_samples())
        except exceptions.CouldntDecodeError:
            print("Failed to load signal!")
            return False
        else:
            self.update_ui(self.signal is None)
            return True

    def is_signal_loaded(self):
        return self.sound is not None and self.signal is not None

    def plot(self, signal, sound):
        self.figure.clear()
        self.subplots = []
        self.lclick = []
        self.lclick_pos = 0
        self.lover = []
        self.lover_pos = 0
        self.lframe = []
        self.lframe_pos = 0
        self.sound_start_at = 0

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
            if i == int(sound.channels / 2):
                ax.set_ylabel("Amplitude")

            self.subplots.append(ax)

            # Handle zoom/pan events
            ax.callbacks.connect("xlim_changed", self.on_plot_change)
            ax.callbacks.connect("ylim_changed", self.on_plot_change)

        self.figure.subplots_adjust(hspace=0.0)
        ax.set_xlabel("Time (s)")
        self.figure.canvas.draw()

        # Save background for updating on the fly
        self.plotbackground = self.figure.canvas.copy_from_bbox(self.figure.bbox)

        # Create lines (for later use, hidden until first update)
        for ax in self.subplots:
            line = ax.axvline(0, linewidth=1, color="black")
            self.lclick.append(line)
            line = ax.axvline(0, linewidth=1, color="grey")
            self.lover.append(line)
            line = ax.axvline(0, linewidth=1, color="blue")
            self.lframe.append(line)

    def on_plot_change(self, axes):
        # Hide all lines to not save them as part of background
        for line in itertools.chain(self.lclick, self.lover, self.lframe):
            line.set_visible(False)

        # Redraw and resave new layout background
        self.figure.canvas.draw()
        self.plotbackground = self.figure.canvas.copy_from_bbox(self.figure.bbox)

        # Reshow all lines
        for line in itertools.chain(self.lclick, self.lover, self.lframe):
            line.set_visible(True)

    def is_plotnav_active(self):
        return self.plotnav._active is None

    def on_plot_click(self, event):
        if not self.is_plotnav_active():
            return

        if event.xdata is not None and event.ydata is not None:
            self.sound_start_at = event.xdata
            self.sound_play()
            self.update_ui(True)

            # Update lines
            self.lclick_pos = event.xdata
            self.plot_update()

    def on_plot_over(self, event):
        if not self.is_plotnav_active():
            return

        # Update lines
        if event.xdata is not None and event.ydata is not None:
            self.lover_pos = event.xdata
        else:
            self.lover_pos = 0

        if self.plotbackground is not None:
            self.plot_update()

    def plot_frame(self, x):
        # Update lines
        self.lframe_pos = x
        self.plot_update()

    def plot_update(self):
        self.figure.canvas.restore_region(self.plotbackground)
        for i, (lclick, lover, lframe) in enumerate(zip(self.lclick, self.lover, self.lframe)):
            lclick.set_xdata([self.lclick_pos])
            lover.set_xdata([self.lover_pos])
            lframe.set_xdata([self.lframe_pos])
            self.subplots[i].draw_artist(lclick)
            self.subplots[i].draw_artist(lover)
            self.subplots[i].draw_artist(lframe)
        self.figure.canvas.blit(self.figure.bbox)

    def sound_play(self):
        if self.sound_thread:
            self.sig_sound_play_at.emit(self.sound_start_at)
        elif self.is_signal_loaded():
            self.sound_thread = SoundThread(self.sound, self.sound_start_at, self.sound_mutex, self.sound_pause_cond)
            self.sig_sound_play_at.connect(self.sound_thread.play_at)
            self.sig_sound_pause.connect(self.sound_thread.pause)
            self.sig_sound_stop.connect(self.sound_thread.stop)
            self.sound_thread.sig_frame.connect(self.plot_frame)
            self.sound_thread.finished.connect(self.on_sound_done)
            self.sound_thread.start()
            self.update_ui(True)

    def is_sound_playing(self):
        return self.sound_thread is not None and not self.sound_paused

    def sound_stop(self):
        self.sig_sound_stop.emit()
        self.sound_mutex.lock()
        self.sound_pause_cond.wakeAll()
        self.sound_mutex.unlock()

    def sound_pause(self):  # Toggle
        if self.sound_paused:
            self.sig_sound_pause.emit()
            self.sound_mutex.lock()
            self.sound_pause_cond.wakeAll()
            self.sound_mutex.unlock()
        else:
            self.sig_sound_pause.emit()
        self.sound_paused = not self.sound_paused
        self.update_ui(True)

    def on_sound_done(self):
        self.sound_thread = None
        self.update_ui(False)
        self.lframe_pos = 0
        self.plot_update()


class SoundThread(QThread):
    sig_frame = pyqtSignal(float)

    def __init__(self, sound, start_at, mutex, pause_cond):
        QThread.__init__(self)

        self.sound = sound
        self.start_at = start_at
        self.restart = True  # Start on True for first start

        self.running = True
        self.paused = False
        self.mutex = mutex
        self.pause_cond = pause_cond

    def __del__(self):
        self.wait()

    def run(self):
        p = PyAudio()
        stream = p.open(
            format=p.get_format_from_width(self.sound.sample_width),
            channels=self.sound.channels,
            rate=self.sound.frame_rate,
            output=True)

        while self.restart:
            self.restart = False

            # Break into 0.5 second chunks
            start = self.start_at * 1000
            current_time = self.start_at
            for chunk in make_chunks(self.sound[start:], 50):
                if not self.running or self.restart:
                    break

                stream.write(chunk._data)

                current_time += 0.05
                self.sig_frame.emit(current_time)

                if self.paused:
                    self.mutex.lock()
                    self.pause_cond.wait(self.mutex)
                    self.mutex.unlock()

        stream.close()
        p.terminate()

    def play_at(self, start_at=0):
        self.start_at = start_at
        self.restart = True

    def pause(self):  # Toggle
        self.paused = not self.paused

    def stop(self):
        self.running = False


if __name__ == "__main__":
    # Create Qt application with window
    app = QApplication(sys.argv)
    main_win = MainWindow()

    # Execute application (blocking)
    app.exec_()

    sys.exit(0)
