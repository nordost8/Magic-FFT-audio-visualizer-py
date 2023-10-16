import os
import random
import shutil

import cv2
import numpy as np
import math
import pygame
from PIL import Image
from matplotlib import cm

class Spectrum_Visualizer:
    """
    The Spectrum_Visualizer visualizes spectral FFT data using a simple PyGame GUI
    """
    nummer = 1

    def __init__(self, ear):
        self.screen = None
        self.prev_additional_surface = None
        self.additional_surface = None
        self.background_image = None
        self.ear = ear
        window_ratio = self.ear.window_ratio

        # Divide the screen into 2 parts (half of the screen height), so the animation height will be equal to dividing
        # the screen into N parts:
        self.DIVISION_FACTOR = 1.6

        self.MAIN_SCREEN_HEIGHT = round(self.ear.height)
        self.WIDTH = round(window_ratio * self.MAIN_SCREEN_HEIGHT)
        self.BARS_SURFACE_HEIGHT = round(self.MAIN_SCREEN_HEIGHT / self.DIVISION_FACTOR)
        self.bottom_indent = 0  # round(0.05 * self.HEIGHT) Отступ знизу снизу extend
        self.y_ext = [self.bottom_indent, self.BARS_SURFACE_HEIGHT]
        self.cm = cm.plasma
        # self.cm = cm.inferno

        self.running_line_surface = None
        self.running_line_font = None

        self.running_line_text = os.path.basename(ear.audio_path).replace('.mp3', '')

        self.decay_speed = 0.10
        self.inter_bar_distance = 0
        self.avg_energy_height = 0.0120
        self.alpha_multiplier = 0.995
        self.move_fraction = 0.0039
        self.shrink_f = 0.994

        self.bar_width = (self.WIDTH / self.ear.n_frequency_bins) - self.inter_bar_distance

        # Configure the bars:
        self.slow_bars, self.fast_bars, self.bar_x_positions = [], [], []
        for i in range(self.ear.n_frequency_bins):
            x = int(i * self.WIDTH / self.ear.n_frequency_bins)
            fast_bar = [int(x), int(self.y_ext[0]), math.ceil(self.bar_width), None]
            slow_bar = [int(x), None, math.ceil(self.bar_width), None]
            self.bar_x_positions.append(x)
            self.fast_bars.append(fast_bar)
            self.slow_bars.append(slow_bar)

        self.add_slow_bars = 1
        self.add_fast_bars = 1
        self.slow_bar_thickness = max(0.00002 * self.BARS_SURFACE_HEIGHT, 1.25 / self.ear.n_frequency_bins)

        self.fast_bar_colors = [list((255 * np.array(self.cm(i))[:3]).astype(int)) for i in
                                np.linspace(0, 255, self.ear.n_frequency_bins).astype(int)]

        self.slow_bar_colors = [list(np.clip((255 * 3.5 * np.array(self.cm(i))[:3]).astype(int), 0, 255)) for i in
                                np.linspace(0, 255, self.ear.n_frequency_bins).astype(int)]

        self.fast_bar_colors = self.fast_bar_colors[::-1]
        self.slow_bar_colors = self.slow_bar_colors[::-1]

        self.slow_features = [0] * self.ear.n_frequency_bins
        self.frequency_bin_max_energies = np.zeros(self.ear.n_frequency_bins)
        self.frequency_bin_energies = self.ear.frequency_bin_energies
        self.bin_text_tags, self.bin_rectangles = [], []

        # Fixed init params:
        self.start_time = None
        self.vis_steps = 0
        self.fps_interval = 10
        self.fps = 0
        self._is_running = False

    def start(self):
        print("Starting spectrum visualizer...")
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # Not showing pygame window
        pygame.init()

        self.running_line_font = pygame.font.Font(None, 20)
        self.running_line_surface = self.running_line_font.render(self.running_line_text, True, (255, 255, 255))

        self.screen = pygame.display.set_mode((self.WIDTH, self.MAIN_SCREEN_HEIGHT))
        self.background_image = pygame.image.load(self.ear.background_path)
        self.background_image = pygame.transform.scale(self.background_image, (self.WIDTH, self.MAIN_SCREEN_HEIGHT))
        self.screen.blit(self.background_image, (0, 0))

        # Additional surface for bars (transparent):
        self.additional_surface = pygame.Surface((self.WIDTH, self.BARS_SURFACE_HEIGHT), pygame.SRCALPHA)

        self.additional_surface.set_alpha(255)
        self.prev_additional_surface = self.additional_surface

        self._is_running = True

    def stop(self):
        print("Stopping spectrum visualizer...")
        del self.screen
        del self.additional_surface
        del self.prev_additional_surface
        pygame.quit()
        self._is_running = False

    def update(self, progress_percentage=0):
        if np.min(self.ear.bin_mean_values) > 0:
            self.frequency_bin_energies = self.avg_energy_height * self.ear.frequency_bin_energies / self.ear.bin_mean_values

        new_w, new_h = int((2 + self.shrink_f) / 3 * self.WIDTH), int(self.shrink_f * self.BARS_SURFACE_HEIGHT)
        prev_additional_surface = pygame.transform.scale(self.prev_additional_surface, (new_w, new_h))

        self.additional_surface.fill((0, 0, 0, 0))

        new_pos = int(self.move_fraction * self.WIDTH - (0.0133 * self.WIDTH)), int(
            self.move_fraction * self.BARS_SURFACE_HEIGHT)
        self.additional_surface.blit(prev_additional_surface, new_pos)

        self.plot_bars()

        combined_surface = pygame.Surface((self.WIDTH, self.MAIN_SCREEN_HEIGHT), pygame.SRCALPHA)
        combined_surface.blit(self.screen, (0, 0))
        position_y = self.MAIN_SCREEN_HEIGHT * (self.DIVISION_FACTOR - 1) / self.DIVISION_FACTOR

        # "fade in", "fade out" video transparent:
        if progress_percentage <= 5:
            self.additional_surface.set_alpha(int((progress_percentage / 5) * 255))
        elif progress_percentage >= 97:
            self.additional_surface.set_alpha(0)
        elif progress_percentage >= 92:
            self.additional_surface.set_alpha(int(255 * (97 - progress_percentage) / 5))

        combined_surface.blit(self.additional_surface, (0, position_y))

        combined_surface.blit(self.running_line_surface,
                              (self.WIDTH - self.running_line_surface.get_width() - 20, self.MAIN_SCREEN_HEIGHT - 30))

        # Callback mode if ear.ready_frame_bacllback exists or save images:
        if hasattr(self.ear, 'ready_frame_callback'):
            self.ear.ready_frame_callback(combined_surface)
        else:
            pygame.image.save(combined_surface, TEMP_FRAMES_PATH + f's_{self.nummer:05}.jpg')
            self.nummer += 1

    def plot_bars(self):
        bars, slow_bars, new_slow_features = [], [], []
        local_height = self.y_ext[1] - self.y_ext[0]
        feature_values = self.frequency_bin_energies[::-1]
        for i in range(len(self.frequency_bin_energies)):
            feature_value = feature_values[i] * local_height

            self.fast_bars[i][3] = int(feature_value + 0.02 * self.BARS_SURFACE_HEIGHT)

            if self.add_slow_bars:
                self.decay = min(0.99, 1 - max(0, self.decay_speed * 60 / self.ear.fft_fps))
                slow_feature_value = max(self.slow_features[i] * self.decay, feature_value)
                new_slow_features.append(slow_feature_value)
                self.slow_bars[i][1] = int(self.fast_bars[i][1] + slow_feature_value)
                self.slow_bars[i][3] = min(2,
                                           int(self.slow_bar_thickness * local_height / 2) or 1)

        def draw_rect_alpha(surface, color, rect, width, alpha=225):
            shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
            color.append(alpha)
            color_with_alpha = color[:3] + [alpha]
            pygame.draw.rect(shape_surf, color_with_alpha, shape_surf.get_rect(), width)
            surface.blit(shape_surf, rect)

        if self.add_fast_bars:
            for i, fast_bar in enumerate(self.fast_bars):
                # pygame.draw.rect(self.additional_surface, self.fast_bar_colors[i], fast_bar, 0)
                draw_rect_alpha(self.additional_surface, self.fast_bar_colors[i], fast_bar, 0)

        self.prev_additional_surface = self.additional_surface.copy().convert_alpha()
        self.prev_additional_surface.set_alpha(self.prev_additional_surface.get_alpha() * self.alpha_multiplier)

        if self.add_slow_bars:
            for i, slow_bar in enumerate(self.slow_bars):
                pygame.draw.rect(self.additional_surface, self.slow_bar_colors[i], slow_bar, 0)

        self.slow_features = new_slow_features

        # Draw everything:
        temp_additional_surface = pygame.transform.rotate(self.additional_surface, 180)
        self.additional_surface.fill((0, 0, 0, 0))
        self.additional_surface.blit(temp_additional_surface, (0, 0))
