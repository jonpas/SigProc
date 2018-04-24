#!/usr/bin/env python3

import sys
import os
import itertools
import numpy as np
from scipy import signal, constants, fftpack
import pyaudio
from pydub import AudioSegment, exceptions
from pydub.utils import make_chunks
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIntValidator

PyAudio = pyaudio.PyAudio

# Development switches and non-exposed settings
MANUAL_CONVOLVE = False  # Use manual convolution instead of scipy.signal.fftconvolve


class MainWindow(QWidget):
    sig_sound_play_at = pyqtSignal(float)
    sig_sound_pause = pyqtSignal()
    sig_sound_stop = pyqtSignal()
    sig_record_stop = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.sound = None
        self.signal = None

        self.plotbackground = None

        self.playing = False
        self.sound_paused = False
        self.sound_start_at = 0

        self.recording = False
        self.doppler = False  # Doppler simulation

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
        self.cb_source_speed = QComboBox()
        self.cb_source_speed.setToolTip("Source speed")
        self.cb_source_speed.addItems(["20 km/h", "50 km/h", "100 km/h", "150 km/h", "200 km/h"])
        self.cb_source_speed.setCurrentIndex(2)
        self.source_speeds = [5.56, 13.89, 27.78, 41.67, 55.56]  # Same indexes as text above (in m/s)
        self.btn_doppler = QPushButton("Simulate Doppler")
        self.btn_doppler.setToolTip("Apply simple Doppler Shift simulation")
        self.btn_doppler.setDisabled(True)
        self.btn_doppler.clicked.connect(self.doppler_simulate)

        # Effects
        self.cb_effect = QComboBox()
        self.cb_effect.setToolTip("Preset effects")
        self.cb_effect.setMaximumWidth(150)
        self.effects = []
        for root, dirs, files in os.walk("resources/impulses"):
            for file in files:
                if file.endswith(".wav") or file.endswith(".mp3"):
                    self.cb_effect.addItem(file.split(".")[0])
                    self.effects.append(os.path.join(root, file))
        self.btn_effect = QPushButton("Apply Effect")
        self.btn_effect.clicked.connect(self.effect_apply)
        self.btn_effect_load = QPushButton("Load Effect")
        self.btn_effect_load.clicked.connect(self.effect_load)

        # Recording
        self.cb_sample_rate = QComboBox()
        self.cb_sample_rate.setToolTip("Sampling rate")
        self.cb_sample_rate.addItems(["8.000 Hz", "11.025 Hz", "22.050 Hz", "44.100 Hz"])
        self.cb_sample_rate.setCurrentIndex(3)
        self.sampling_rates = [8000, 11025, 22050, 44100]  # Same indexes as text above
        self.btn_record = QPushButton("Record")
        self.btn_record.setMinimumWidth(100)
        self.btn_record.clicked.connect(self.record)

        self.cb_bit_depth = QComboBox()
        self.cb_bit_depth.setToolTip("Bit depth")
        self.cb_bit_depth.addItems(["8 b", "16 b"])
        self.cb_bit_depth.setCurrentIndex(1)
        self.bit_depths = [pyaudio.paUInt8, pyaudio.paInt16]  # Same indexes as text above

        # Analysis (ST-DFT)
        self.stdft_window = QLineEdit()
        self.stdft_window.setText("256")
        self.stdft_window.setToolTip("Window length")
        self.stdft_window.setMaximumWidth(35)
        self.stdft_window.setValidator(QIntValidator(0, 2147483647))
        self.stdft_noverlap = QLineEdit()
        self.stdft_noverlap.setText("128")
        self.stdft_noverlap.setMaximumWidth(35)
        self.stdft_noverlap.setValidator(QIntValidator(0, 2147483647))
        self.stdft_noverlap.setToolTip("Overlap between windows (must be smaller than window length)")
        self.btn_analyse = QPushButton("Analyse")
        self.btn_analyse.setToolTip("Perform Short Time Discrete Fourier Transform analysis (spectrogram)")
        self.btn_analyse.setDisabled(True)
        self.btn_analyse.clicked.connect(self.analyse)

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
        hbox_bot.addWidget(self.cb_source_speed)
        hbox_bot.addWidget(self.btn_doppler)
        hbox_bot.addStretch()
        hbox_bot.addSpacerItem(spacer)
        hbox_bot.addWidget(self.cb_effect)
        hbox_bot.addWidget(self.btn_effect)
        hbox_bot.addWidget(self.btn_effect_load)
        hbox_bot.addStretch()
        hbox_bot.addSpacerItem(spacer)
        hbox_bot.addWidget(self.cb_sample_rate)
        hbox_bot.addWidget(self.cb_bit_depth)
        hbox_bot.addWidget(self.btn_record)

        hbox_bot2 = QHBoxLayout()
        hbox_bot2.addWidget(self.stdft_window)
        hbox_bot2.addWidget(self.stdft_noverlap)
        hbox_bot2.addWidget(self.btn_analyse)
        hbox_bot2.addStretch()

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_top)
        vbox.addWidget(self.figure.canvas)
        vbox.addLayout(hbox_bot)
        vbox.addLayout(hbox_bot2)

        # Window
        self.setLayout(vbox)
        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle("Signal Processor")
        self.show()

    # Overriden resize event
    def resizeEvent(self, resizeEvent):
        if self.is_sound_loaded():
            self.on_plot_change(None)
        self.plotnav.move(self.width() - 55, 0)

    def update_ui(self):
        block_general = self.playing or self.sound_paused or self.recording

        self.btn_save.setDisabled(not self.is_sound_loaded())

        self.btn_pause.setDisabled(not self.playing)
        self.btn_pause.setText("Resume" if self.sound_paused else "Pause")
        self.btn_play.setDisabled(self.playing or self.recording)
        self.btn_stop.setDisabled(not self.playing or self.recording)

        self.plotnav.setDisabled(self.playing and not self.sound_paused)

        self.cb_source_speed.setDisabled(block_general or self.doppler)
        self.btn_doppler.setDisabled(not self.is_sound_loaded() or self.doppler)

        self.cb_effect.setDisabled(block_general)
        self.btn_effect.setDisabled(block_general)
        self.btn_effect_load.setDisabled(block_general)

        self.cb_sample_rate.setDisabled(block_general)
        self.cb_bit_depth.setDisabled(block_general)
        self.btn_record.setDisabled(self.playing or self.sound_paused)
        self.btn_record.setText("Stop Recording" if self.recording else "Record")

        self.stdft_window.setDisabled(block_general)
        self.stdft_noverlap.setDisabled(block_general)
        self.btn_analyse.setDisabled(block_general)

    def show_open_dialog(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", filter="Audio (*.wav *.mp3)")
        if fname[0] and self.load_sound(fname[0]):
            self.txt_file.setText(fname[0])

    def show_save_dialog(self):
        fname = QFileDialog.getSaveFileName(self, "Save file", filter="Audio (*.wav *.mp3)")
        if fname[0] and self.is_sound_loaded():
            ext = fname[0].rsplit(".", 1)[-1]
            try:
                self.sound.export(fname[0], format=ext)
            except exceptions.CouldntEncodeError:
                print("Failed to save signal!")
            else:
                self.txt_file.setText(fname[0])

    def load_sound(self, file):
        self.sound_stop()
        self.doppler = False

        try:
            self.sound = AudioSegment.from_file(file)
            self.signal = np.array(self.sound.get_array_of_samples())
        except exceptions.CouldntDecodeError:
            print("Failed to load sound!")
            self.sound = None
            self.signal = None
            return False
        else:
            self.update_ui()
            self.plot(self.signal, self.sound)
            return True

    def is_sound_loaded(self):
        return self.sound is not None and self.signal is not None

    def load_signal(self, data, sample_width, rate, channels):
        self.sound = AudioSegment(
            data=data,
            sample_width=sample_width,  # 3 (24-bit) not supported by pydub
            frame_rate=rate,
            channels=channels)
        self.signal = np.array(self.sound.get_array_of_samples())
        self.update_ui()
        self.plot(self.signal, self.sound)

    def effect_load(self):
        feffect = self.effects[self.cb_effect.currentIndex()]
        if self.load_sound(feffect):
            self.txt_file.setText(feffect)
            self.plot(self.signal, self.sound)

    def effect_apply(self):
        if not self.is_sound_loaded():
            print("Failed to apply effect! No sound loaded!")
            return

        if self.sound.channels > 2:
            print("Failed to apply effect! Sound has more than 2 channels!")
            return

        feffect = self.effects[self.cb_effect.currentIndex()]
        try:
            effect_sound = AudioSegment.from_file(feffect)
            effect_signal = np.array(effect_sound.get_array_of_samples())
        except exceptions.CouldntDecodeError:
            print("Failed to load effect!")

        if effect_sound.frame_rate != self.sound.frame_rate:
            print("Failed to apply effect! Effect rate ({}) not same as sound rate ({})!"
                  .format(effect_sound.frame_rate, self.sound.frame_rate))
            return

        # Create stereo in case original sound is mono
        sound_channels = self.sound.channels
        if self.sound.channels < 2:
            self.sound = AudioSegment.from_mono_audiosegments(self.sound, self.sound)
            self.signal = np.array(self.sound.get_array_of_samples())

        # Convolve signals using fast fourier transform (into stereo, each channel separately)
        step = effect_sound.channels
        left = None
        right = None

        for i in range(0, sound_channels):
            if MANUAL_CONVOLVE:
                # Manual convolve
                n = fftpack.helper.next_fast_len(len(self.signal[i::step]) + len(effect_signal[i::step]) - 1)
                x = np.fft.rfft(np.append(self.signal[i::step], np.zeros(len(effect_signal[i::step]) - 1)), n)
                y = np.fft.rfft(np.append(effect_signal[i::step], np.zeros(len(self.signal[i::step]) - 1)), n)
                ch = np.fft.irfft(x * y)
            else:
                # SciPy fftconvolve
                ch = signal.fftconvolve(self.signal[i::step], effect_signal[i::step])

            # Normalize and amplify
            ch = np.array(ch / np.linalg.norm(ch))
            ch = np.multiply(ch, 65535)  # float to int
            volume_diff = np.max(self.signal[i::step]) / np.max(ch)
            ch = np.multiply(ch, volume_diff)

            if i == 0:
                left = ch
                if sound_channels == 1:
                    right = left  # Mono input, copy channel
            else:
                right = ch

        # Join channels back together and load signal
        final = np.empty(left.size + right.size, np.int16)
        final[0::step] = left.astype(np.int16)
        final[1::step] = right.astype(np.int16)
        self.load_signal(b''.join(final), 2, self.sound.frame_rate, effect_sound.channels)

    def doppler_simulate(self):
        self.doppler = True
        self.update_ui()

        speed_source = self.source_speeds[self.cb_source_speed.currentIndex()]

        # Frequency manipulation
        speed_sound = constants.speed_of_sound
        freq_in = speed_sound / (speed_sound - speed_source) * self.sound.frame_rate
        freq_out = speed_sound / (speed_sound + speed_source) * self.sound.frame_rate

        half1 = self.sound[0:int(len(self.sound) * 0.5)]
        half1 = AudioSegment(
            data=half1.get_array_of_samples(),
            sample_width=self.sound.sample_width,
            frame_rate=int(freq_in),
            channels=self.sound.channels)

        half2 = self.sound[int(len(self.sound) * 0.5):]
        half2 = AudioSegment(
            data=half2.get_array_of_samples(),
            sample_width=self.sound.sample_width,
            frame_rate=int(freq_out),
            channels=self.sound.channels)

        self.sound = half1.append(half2, crossfade=100)
        self.signal = np.array(self.sound.get_array_of_samples())

        # Volume manipulation (decrease with distance)
        half_time = half1.duration_seconds
        dist_max = speed_source * half_time
        print("Maximum distance: {} m".format(dist_max))

        distances = np.linspace(
            0.0, speed_source * (len(self.signal) / self.sound.frame_rate / self.sound.channels),
            num=int(len(self.signal) / self.sound.channels))  # Plot distances
        distances -= dist_max  # Take away maximum distance to get relative from center
        distances = np.absolute(distances)  # Make positive in both directions (_/^\_)
        distances = np.maximum(distances, 1.0)  # Prevent center clipping

        new_volumes = np.power(distances, -1.0)  # Scale volume with distance
        for i in range(0, self.sound.channels):  # Apply to all channels
            self.signal[i::self.sound.channels] = np.multiply(self.signal[i::self.sound.channels], new_volumes)

        self.signal = self.signal.astype(np.int16)

        # Load and plot new signal with doppler and visualization subplot
        self.load_signal(b''.join(self.signal), self.sound.sample_width, self.sound.frame_rate, self.sound.channels)
        self.plot(self.signal, self.sound, doppler_max=half_time)

    def analyse(self):
        if not self.is_sound_loaded():
            return

        if not self.stdft_window.text() or not self.stdft_noverlap.text():
            print("Failed to analyse! Invalid input (must be integers)!")
            return

        window = int(self.stdft_window.text())
        noverlap = int(self.stdft_noverlap.text())

        if window <= 0 or noverlap <= 0:
            print("Failed to analyse! Invalid input (must be integers greater than 0)!")
            return

        if noverlap >= window:
            print("Failed to analyse! Overlap must be less than window size!")
            return

        if self.sound.channels > 1:
            print("Warning! Analysing only first channel!")

        self.plot(self.signal, self.sound, stdft_window=window, stdft_noverlap=noverlap)

    def plot(self, sig, sound, doppler_max=-1.0, stdft_window=-1, stdft_noverlap=-1):
        self.figure.clear()
        self.subplots = []
        self.lclick = []
        self.lclick_pos = 0
        self.lover = []
        self.lover_pos = 0
        self.lframe = []
        self.lframe_pos = 0
        self.sound_start_at = 0

        doppler = doppler_max != -1.0
        analysis = stdft_window != -1 and stdft_noverlap != -1

        # X axis as time in seconds
        time = np.linspace(0, sound.duration_seconds, num=len(sig))

        for i in range(0, sound.channels):
            ax = self.figure.add_subplot(sound.channels + doppler + analysis, 1, i + 1)

            # Plot current channel, slicing it away
            ax.plot(time[i::sound.channels], sig[i::sound.channels])  # [samp1L, samp1R, samp2L, samp2R]
            ax.margins(0)

            # Hide X axis on all but last channel
            if i + 1 < sound.channels + doppler + analysis:
                ax.get_xaxis().set_visible(False)
            # Display Y label somewhere in the middle
            if i == int(sound.channels / 2):
                ax.set_ylabel("Amplitude")

            self.subplots.append(ax)

        if doppler:
            ax = self.figure.add_subplot(sound.channels + analysis + 1, 1, sound.channels + analysis + 1)
            ax.margins(0)
            ax.plot(time, sig * [0])
            ax.axhline(0, linewidth=2, color="black")
            ax.axvline(doppler_max, ymin=0.25, ymax=0.75, linewidth=2, color="blue")
            ax.set_ylim([-1, 1])
            ax.get_yaxis().set_ticks([])
            ax.set_ylabel("Doppler Sim")
            self.subplots.append(ax)

        if analysis:
            ax = self.figure.add_subplot(sound.channels + doppler + 1, 1, sound.channels + doppler + 1)
            ax.margins(0)
            ax.specgram(sig[0::sound.channels], Fs=self.sound.frame_rate,
                        NFFT=stdft_window, noverlap=stdft_noverlap)
            ax.set_ylabel("Frequency (Hz)")
            self.subplots.append(ax)

        # Handle zoom/pan events
        for ax in self.subplots:
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
            self.update_ui()

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
        if self.playing:
            self.sig_sound_play_at.emit(self.sound_start_at)
        elif self.is_sound_loaded():
            self.sound_thread = SoundThread(self.sound, self.sound_start_at, self.sound_mutex, self.sound_pause_cond)
            self.sig_sound_play_at.connect(self.sound_thread.play_at)
            self.sig_sound_pause.connect(self.sound_thread.pause)
            self.sig_sound_stop.connect(self.sound_thread.stop)
            self.sound_thread.sig_frame.connect(self.plot_frame)
            self.sound_thread.finished.connect(self.on_sound_done)
            self.sound_thread.start()

            self.playing = True
            self.update_ui()

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
        self.update_ui()

    def on_sound_done(self):
        self.playing = False
        self.sound_paused = False
        self.update_ui()
        self.lframe_pos = 0
        self.plot_update()

    def record(self):  # Toggle
        if self.recording:
            self.sig_record_stop.emit()
        else:
            self.recording = True
            bit_depth = self.bit_depths[self.cb_bit_depth.currentIndex()]
            rate = self.sampling_rates[self.cb_sample_rate.currentIndex()]
            self.record_thread = RecordThread(bit_depth, rate, 2)  # Always record in stereo (2 channels)
            self.sig_record_stop.connect(self.record_thread.stop)
            self.record_thread.sig_return.connect(self.on_record_return)
            self.record_thread.start()
            self.update_ui()

    def on_record_return(self, data, sample_width, rate, channels):
        self.load_signal(data, sample_width, rate, channels)
        self.recording = False
        self.update_ui()


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

                    # Reopen stream due to an issue on Linux
                    # where stream stops begin read by backend
                    stream = p.open(
                        format=p.get_format_from_width(self.sound.sample_width),
                        channels=self.sound.channels,
                        rate=self.sound.frame_rate,
                        output=True)

        stream.close()
        p.terminate()

    def play_at(self, start_at=0):
        self.start_at = start_at
        self.restart = True

    def pause(self):  # Toggle
        self.paused = not self.paused

    def stop(self):
        self.running = False


class RecordThread(QThread):
    sig_return = pyqtSignal(bytes, int, int, int)

    def __init__(self, bit_depth, rate, channels):
        QThread.__init__(self)

        self.bit_depth = bit_depth
        self.rate = rate
        self.channels = channels

        self.running = True

    def __del__(self):
        self.wait()

    def run(self):
        p = PyAudio()
        stream = p.open(
            format=self.bit_depth,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=1024)

        data = []
        while self.running:
            data.append(stream.read(1024))

        stream.close()

        # Return recording data
        self.sig_return.emit(b''.join(data), p.get_sample_size(self.bit_depth), self.rate, self.channels)
        p.terminate()

    def stop(self):
        self.running = False


if __name__ == "__main__":
    # Create Qt application with window
    app = QApplication(sys.argv)
    main_win = MainWindow()

    # Execute application (blocking)
    app.exec_()

    sys.exit(0)
