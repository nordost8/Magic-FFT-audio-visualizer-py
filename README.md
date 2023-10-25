# Magic-FFT-audio-visualizer-py
This audio FFT analyzer magically transforms sound waves into stunning animation frames, allowing you to visualize the 'auras' of audio. 🎶✨Developed using ffmpeg and Python. 🐍

**Simply watch this YouTube video and judge for yourself (clickable):**

[![Watch the video](https://img.youtube.com/vi/PbPL1uChWfQ/maxresdefault.jpg)](https://youtu.be/PbPL1uChWfQ)

**Usage:**

- Place the audio files you want to include in your radio in the "resource/music_files" folder.
- After video generation, they will be stored in the "resource/ready_videos" folder.

**Running:**

You can use the video_master_example.py. The code is simple and easy to understand, just follow the comments for guidance!

**Note:**
You can intercept each frame of the video (while it's being generated) in the ready_frame_callback function and do anything you want with that frame! For example, you can see how to combine them into a final product (video). Alternatively, you can save animation frames individually and modify them to your liking. The fft_analyzer function has numerous standard parameters, such as screen dimensions, orientation, frames per second, frequency band density, and more. Feel free to ask me if you're interested in specific parameters! You can also easily customize the animation to your liking, creating beautiful animations. For example, pygame is used in this instance, and you can enhance existing animations or create new ones from scratch.

**"Under the hood":**
At the core of the program is the FFT (Fast Fourier Transform). The FFT algorithm efficiently computes the Discrete Fourier Transform (DFT) by recursively breaking the signal into smaller parts, which significantly speeds up the analysis. It involves complex numbers to represent the amplitudes and phases of different frequency components. This process provides the frequency-domain representation of the input signal, allowing for various types of analysis and processing.

[An example of a YouTube radio project created with this Magic-FFT](https://github.com/nordost8/Simple-Youtube-LoFi-radio-streamer-py)

[![Telegram](https://img.icons8.com/color/48/000000/telegram-app.png)](https://t.me/nordost8)
