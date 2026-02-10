"""
Camera-Display Experiment

Эксперимент с виртуальной моделью камера-дисплей для проверки
рекурсивного замыкания в визуальной домене согласно УМС.
"""

from .core import CameraDisplayCore
from .qualia_engine import VisualQualiaEngine
from .controller import CameraController
from .visualizer import Visualizer

__all__ = ['CameraDisplayCore', 'VisualQualiaEngine', 'CameraController', 'Visualizer']