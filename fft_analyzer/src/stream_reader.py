import wave
import pyaudio
import time
from collections import deque
from pydub import AudioSegment
from utils import *
import os

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TEMP_AUDIO_WAV_PATH = os.path.join(ROOT_DIRECTORY, '../temp/audio/audio_temp.wav')


class StreamReader:
    calls = 0

    def __init__(self,
                 audio_path=None,
                 updates_per_second=1000,
                 analyzer_cls_link=None
                 ):

        self.stream_start_time = None
        self.analyzer_cls_link = analyzer_cls_link
        self.pa = pyaudio.PyAudio()

        # Temporary variables #hacks!
        self.update_window_n_frames = 1024  # Don't remove this, needed for device testing!
        self.data_buffer = None

        self.audio = AudioSegment.from_mp3(audio_path)

        self.rate = self.audio.frame_rate
        self.channels = self.audio.channels

        output_audio = AudioSegment(self.audio.raw_data, frame_rate=self.audio.frame_rate,
                                    sample_width=self.audio.sample_width,
                                    channels=self.audio.channels)

        output_audio.export(TEMP_AUDIO_WAV_PATH, format="wav")

        self.wf = wave.open(TEMP_AUDIO_WAV_PATH, 'rb')

        data_capture_per_mal_size = updates_per_second / 2
        self.update_window_n_frames = round_up_to_even(self.rate / data_capture_per_mal_size)
        self.data_capture_delays = deque(maxlen=20)
        self.new_data = False

    def start_audio_reading(self, data_windows_to_buffer=None):
        self.data_windows_to_buffer = data_windows_to_buffer
        # The accuracy of high and low note animations in update_window_n_frames depends on the number of windows in one buffer element, n_windows in data_windows_to_buffer.
        # The self.update_window_n_frames is the size of one buffer element. Since the data is read from a file, not from a microphone,
        # it should be twice as large.
        self.data_buffer = numpy_data_buffer(self.data_windows_to_buffer, self.update_window_n_frames)

        print("\n-- Starting audio reading...\n")

        self.stream_start_time = time.time()
        time.sleep(math.nextafter(0, 1))  # smallest positive value
        total_frames_to_read = int(self.audio.duration_seconds * self.audio.frame_rate)
        frame_count_total = 0
        while True:
            frame_count = int(self.update_window_n_frames / 2)
            data = self.wf.readframes(frame_count)
            if self.data_buffer is not None:
                try:
                    self.data_buffer.append_data(np.frombuffer(data, dtype=np.int16))
                    frame_count_total += frame_count
                    frames_remaining = total_frames_to_read - frame_count_total
                    progress_percentage = (frame_count_total / total_frames_to_read) * 100
                    self.analyzer_cls_link.progress_percentage = progress_percentage
                    print('Progress: {:.2f}% (Frames Read: {}, Frames Remaining: {})'.format(progress_percentage,
                                                                                             frame_count_total,
                                                                                             frames_remaining))
                except ValueError as e:
                    # At the end, it attempts to write 0 bytes, triggering a hook to react to the end of the file:
                    print(e)
                    self.new_data = ENDE_FLAG
                    self.analyzer_cls_link.visualizer.stop()
                    break

                if self.analyzer_cls_link is not None:
                    self.new_data = True
                    self.analyzer_cls_link.get_audio_features()
