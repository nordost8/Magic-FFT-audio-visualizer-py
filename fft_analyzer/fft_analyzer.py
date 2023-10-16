import argparse
from fft_analyzer.src.streamanalyzer import StreamAnalyzer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=int, dest='audio_path',
                        help='Audio file path in system (ex.: ./resource/my_audio.mp3'),
    parser.add_argument('--background_path', type=int, dest='background_path',
                        help='Background image path in system (ex.: ./resource/image.jpg'),
    parser.add_argument('--height', type=int, default=1080, dest='height',
                        help='height, in pixels, of the visualizer window'),
    parser.add_argument('--updates_per_second', type=int, default=24, dest='updates_per_second',
                        help='Updates per second')
    parser.add_argument('--smoothing_length_ms', type=int, default=300, dest='smoothing_length_ms',
                        help='Smoothing length / ms')
    parser.add_argument('--n_frequency_bins', type=int, default=300, dest='frequency_bins',
                        help='The FFT features are grouped in bins')
    parser.add_argument('--window_ratio', default='16/9', dest='window_ratio',
                        help='float ratio of the visualizer window. e.g. 16/9 (popular, fullHD)')
    return parser.parse_args()


def convert_window_ratio(window_ratio):
    if '/' in window_ratio:
        dividend, divisor = window_ratio.split('/')
        try:
            float_ratio = float(dividend) / float(divisor)
        except:
            raise ValueError('window_ratio should be in the format: float/float')
        return float_ratio
    raise ValueError('window_ratio should be in the format: float/float')


def fft_analyzer(audio_path=None, background_path=None, updates_per_second=None, smoothing_length_ms=None, frequency_bins=None,
                 height=None, window_ratio=None, ready_frame_callback=None):
    args = parse_args()
    window_ratio = convert_window_ratio(args.window_ratio or window_ratio)

    StreamAnalyzer(
        FFT_window_size_ms=60, # Window size used for the FFT transform, HARDCORED.
        audio_path=args.audio_path or audio_path,
        background_path=args.background_path or background_path,
        updates_per_second=args.updates_per_second or updates_per_second, # Updates per second (FPS)
        smoothing_length_ms=args.smoothing_length_ms or smoothing_length_ms, # Apply some temporal smoothing to reduce noisy features
        n_frequency_bins=args.frequency_bins or frequency_bins,  # The FFT features are grouped in bins
        height=args.height or height,  # Height, in pixels, of the visualizer window,
        window_ratio=window_ratio,  # Float ratio of the visualizer window. e.g. 24/9
        ready_frame_callback=ready_frame_callback
    )

    return True

if __name__ == '__main__':
    fft_analyzer()
