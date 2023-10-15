import time
from scipy.signal import savgol_filter

from fft_analyzer.src.stream_reader_pyaudio import StreamReader
from fft_analyzer.src.visualizer import Spectrum_Visualizer
from fft_analyzer.src.fft import getFFT
from fft_analyzer.src.utils import *


class StreamAnalyzer:
    """
    The Audio_Analyzer class provides access to continuously recorded
    (and mathematically processed) audio data.

    Arguments:

        device: int or None:      Select which audio stream to read .
        rate: float or None:      Sample rate to use. Defaults to something supported.
        FFT_window_size_ms: int:  Time window size (in ms) to use for the FFT transform
        updatesPerSecond: int:    How often to record new data.

    """

    def __init__(self,
                 FFT_window_size_ms=50,
                 audio_path=None,
                 updates_per_second=100,
                 smoothing_length_ms=50,
                 n_frequency_bins=51,
                 height=450,
                 window_ratio=24 / 9,
                 ready_frame_callback=None):

        self.progress_percentage = 0
        self.n_frequency_bins = n_frequency_bins
        self.height = height
        self.window_ratio = window_ratio
        self.ready_frame_callback = ready_frame_callback
        self.audio_path = audio_path

        # Creating a stream instance to calculate and obtain the frame rate, etc.:
        self.stream_reader = StreamReader(
            audio_path=audio_path,
            updates_per_second=updates_per_second,
            analyzer_cls_link=self
        )

        self.rate = self.stream_reader.rate

        # Custom settings:
        self.rolling_stats_window_s = 20  # The axis range of the FFT features will adapt dynamically using a window of N seconds
        self.equalizer_strength = 0.20  # [0-1] --> gradually rescales all FFT features to have the same mean
        self.apply_frequency_smoothing = True  # Apply a postprocessing smoothing filter over the FFT outputs

        if self.apply_frequency_smoothing:
            self.filter_width = round_up_to_even(0.03 * self.n_frequency_bins) - 1

        """
        Sound Visualization: The FFT_window_size parameter can also affect how accurately and faithfully 
        the animation represents the sound. For example, if the audio has many high-frequency components, 
        using a larger FFT_window_size will help detect these frequencies and display them in the animation.

        """

        self.FFT_window_size = round_up_to_even(self.rate * FFT_window_size_ms / 1000)
        self.FFT_window_size_ms = 1000 * self.FFT_window_size / self.rate
        self.fft = np.ones(int(self.FFT_window_size / 2), dtype=float)
        self.fftx = np.arange(int(self.FFT_window_size / 2), dtype=float) * self.rate / self.FFT_window_size

        self.data_windows_to_buffer = math.ceil(self.FFT_window_size / self.stream_reader.update_window_n_frames)
        self.data_windows_to_buffer = max(1, self.data_windows_to_buffer)

        # Temporal smoothing:
        # Currently the buffer acts on the FFT_features (which are computed only occasionally eg 30 fps)
        # This is bad since the smoothing depends on how often the .get_audio_features() method is called...
        self.smoothing_length_ms = smoothing_length_ms
        if self.smoothing_length_ms > 0:
            self.smoothing_kernel = get_smoothing_filter(self.FFT_window_size_ms, self.smoothing_length_ms, verbose=1)
            self.feature_buffer = numpy_data_buffer(len(self.smoothing_kernel), len(self.fft), dtype=np.float32,
                                                    data_dimensions=2)

        # This can probably be done more elegantly...
        self.fftx_bin_indices = np.logspace(np.log2(len(self.fftx)), 0, len(self.fftx), endpoint=True, base=2,
                                            dtype=None) - 1
        self.fftx_bin_indices = np.round(
            ((self.fftx_bin_indices - np.max(self.fftx_bin_indices)) * -1) / (len(self.fftx) / self.n_frequency_bins),
            0).astype(int)
        self.fftx_bin_indices = np.minimum(np.arange(len(self.fftx_bin_indices)),
                                           self.fftx_bin_indices - np.min(self.fftx_bin_indices))

        self.frequency_bin_energies = np.zeros(self.n_frequency_bins)
        self.frequency_bin_centres = np.zeros(self.n_frequency_bins)
        self.fftx_indices_per_bin = []
        for bin_index in range(self.n_frequency_bins):
            bin_frequency_indices = np.where(self.fftx_bin_indices == bin_index)
            self.fftx_indices_per_bin.append(bin_frequency_indices)
            fftx_frequencies_this_bin = self.fftx[bin_frequency_indices]
            self.frequency_bin_centres[bin_index] = np.mean(fftx_frequencies_this_bin)

        # Hardcoded parameters:
        self.fft_fps = 30
        self.log_features = False  # Plot log(FFT features) instead of FFT features --> usually pretty bad
        self.num_ffts = 0
        self.strongest_frequency = 0

        # Assume the incoming sound follows a pink noise spectrum:
        self.power_normalization_coefficients = np.logspace(np.log2(1), np.log2(np.log2(self.rate / 2)), len(self.fftx),
                                                            endpoint=True, base=2, dtype=None)
        self.rolling_stats_window_n = self.rolling_stats_window_s * self.fft_fps  # Assumes ~30 FFT features per second
        print('fft_fps:', self.fft_fps)
        print('rolling_stats_window_s:', self.rolling_stats_window_s)
        print('rolling_stats_window_n:', self.rolling_stats_window_n)
        self.rolling_bin_values = numpy_data_buffer(self.rolling_stats_window_n, self.n_frequency_bins,
                                                    start_value=25000)
        self.bin_mean_values = np.ones(self.n_frequency_bins)

        print("Using FFT_window_size length of %d for FFT ---> window_size = %dms" % (
        self.FFT_window_size, self.FFT_window_size_ms))
        print("##################################################################################################")

        # We run a visualizer (actually a game) that provides the update() method to update the image based on the provided data.
        # The data is read from an audio file and placed in a common buffer, from which they are pulled into the game during update():

        self.visualizer = Spectrum_Visualizer(self)
        self.visualizer.start()
        self.stream_reader.start_audio_reading(self.data_windows_to_buffer)
        self.stream_reader.terminate()
        print("Animation clips generated")

    def update_rolling_stats(self):
        pass
        self.rolling_bin_values.append_data(self.frequency_bin_energies)
        self.bin_mean_values = np.mean(self.rolling_bin_values.get_buffer_data(), axis=0)
        self.bin_mean_values = np.maximum((1 - self.equalizer_strength) * np.mean(self.bin_mean_values),
                                          self.bin_mean_values)

    def update_features(self, n_bins=3):
        latest_data_window = self.stream_reader.data_buffer.get_most_recent(self.FFT_window_size)

        self.fft = getFFT(latest_data_window, self.rate, self.FFT_window_size, log_scale=self.log_features)
        # Equalize pink noise spectrum falloff:
        self.fft = self.fft * self.power_normalization_coefficients
        self.num_ffts += 1
        self.fft_fps = self.num_ffts / (time.time() - self.stream_reader.stream_start_time)

        if self.smoothing_length_ms > 0:
            self.feature_buffer.append_data(self.fft)
            buffered_features = self.feature_buffer.get_most_recent(len(self.smoothing_kernel))
            if len(buffered_features) == len(self.smoothing_kernel):
                buffered_features = self.smoothing_kernel * buffered_features
                self.fft = np.mean(buffered_features, axis=0)

        self.strongest_frequency = self.fftx[np.argmax(self.fft)]

        # ToDo: replace this for-loop with pure numpy code
        for bin_index in range(self.n_frequency_bins):
            self.frequency_bin_energies[bin_index] = np.mean(self.fft[self.fftx_indices_per_bin[bin_index]])

        # Beat detection ToDo:
        # https://www.parallelcube.com/2018/03/30/beat-detection-algorithm/
        # https://github.com/shunfu/python-beat-detector
        # https://pypi.org/project/vamp/

        return

    def get_audio_features(self):
        if self.stream_reader.new_data is ENDE_FLAG:
            self.visualizer.stop()
        elif self.stream_reader.new_data is True:
            self.update_features()
            self.update_rolling_stats()
            self.stream_reader.new_data = False

            self.frequency_bin_energies = np.nan_to_num(self.frequency_bin_energies, copy=True)
            if self.apply_frequency_smoothing:
                if self.filter_width > 3:
                    self.frequency_bin_energies = savgol_filter(self.frequency_bin_energies, self.filter_width, 3)
            self.frequency_bin_energies[self.frequency_bin_energies < 0] = 0

            if self.visualizer._is_running:
                self.visualizer.update(self.progress_percentage)
