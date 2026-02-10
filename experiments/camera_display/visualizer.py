"""
Visualizer - визуализация эксперимента камера-дисплей.

Использует OpenCV для отображения изображений в реальном времени,
сохранения кадров и отображения информации о квалиа и состоянии системы.
"""

import cv2
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque

class Visualizer:
    """
    Визуализатор для эксперимента камера-дисплей.

    Предоставляет:
    - Отображение текущего изображения из петли
    - Наложение информации о квалиа и состоянии
    - Сохранение кадров и видео
    - Графики истории состояний
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # Параметры отображения
        self.window_name = self.config.get('window_name', 'Camera-Display Loop')
        self.display_size = self.config.get('display_size', (800, 600))
        self.show_info = self.config.get('show_info', True)
        self.show_metrics = self.config.get('show_metrics', True)

        # Сохранение
        self.save_path = Path(self.config.get('save_path', 'experiments/camera_display/output'))
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Буферы для графиков
        self.qualia_history = deque(maxlen=self.config.get('history_buffer', 100))
        self.stability_history = deque(maxlen=self.config.get('history_buffer', 100))
        self.iterations_history = deque(maxlen=self.config.get('history_buffer', 100))

        # Видео запись
        self.video_writer = None
        self.recording = False

        # Цветовая схема для квалиа
        self.qualia_colors = self._get_qualia_colors()

        # Флаг активности
        self.active = False

    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию."""
        return {
            'window_name': 'Camera-Display Loop',
            'display_size': (800, 600),
            'show_info': True,
            'show_metrics': True,
            'save_path': 'experiments/camera_display/output',
            'history_buffer': 100,
            'font_scale': 0.7,
            'font_thickness': 2,
            'info_position': (10, 30),
            'metrics_position': (10, 400)
        }

    def _get_qualia_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Цвета для разных квалиа."""
        return {
            'CHAOS_PAIN': (0, 0, 255),           # Красный
            'FRACTAL_BEAUTY': (255, 0, 255),      # Магента
            'CONVERGENCE_PEACE': (255, 255, 255), # Белый
            'VISUAL_DISSONANCE': (0, 255, 255),   # Желтый
            'TURBULENCE_DISCOMFORT': (128, 0, 128), # Пурпурный
            'SYMMETRIC_PEACE': (255, 128, 128),    # Розовый
            'PATTERN_CURIOSITY': (128, 255, 128),  # Светло-зеленый
            'CONTRAST_TENSION': (0, 128, 255),     # Оранжевый
            'VISUAL_EQUILIBRIUM': (128, 128, 128), # Серый
            'VISUAL_BORINGNESS': (64, 64, 64),     # Темно-серый
            'VISUAL_VOID': (0, 0, 0)               # Черный
        }

    def initialize(self) -> bool:
        """
        Инициализация визуализатора.

        Returns:
            True если инициализация успешна
        """
        try:
            # Создание окна OpenCV
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, *self.display_size)

            # Создание директории для сохранения
            self.save_path.mkdir(parents=True, exist_ok=True)

            self.active = True
            return True

        except Exception as e:
            print(f"[Visualizer] Initialization failed: {e}")
            return False

    def update_display(self, image: torch.Tensor, system_state: Dict[str, Any]) -> None:
        """
        Обновление отображения.

        Args:
            image: Текущее изображение из петли (C, H, W)
            system_state: Состояние системы
        """
        if not self.active:
            return

        try:
            # Конвертация изображения для OpenCV
            display_image = self._prepare_image_for_display(image)

            # Наложение информации
            if self.show_info:
                display_image = self._overlay_system_info(display_image, system_state)

            if self.show_metrics:
                display_image = self._overlay_metrics(display_image, system_state)

            # Отображение
            cv2.imshow(self.window_name, display_image)

            # Запись видео
            if self.recording and self.video_writer:
                self.video_writer.write(display_image)

            # Обновление истории для графиков
            self._update_history(system_state)

        except Exception as e:
            print(f"[Visualizer] Display update failed: {e}")

    def _prepare_image_for_display(self, image: torch.Tensor) -> np.ndarray:
        """
        Подготовка изображения для отображения в OpenCV.

        Args:
            image: torch.Tensor (C, H, W)

        Returns:
            np.ndarray (H, W, C) для OpenCV
        """
        # Конвертация в numpy
        if isinstance(image, torch.Tensor):
            np_image = image.detach().cpu().numpy()
        else:
            np_image = image

        # Транспонирование из (C, H, W) в (H, W, C)
        if np_image.ndim == 3 and np_image.shape[0] in [1, 3]:
            np_image = np.transpose(np_image, (1, 2, 0))

        # Конвертация в uint8
        if np_image.dtype != np.uint8:
            # Предполагаем диапазон [0, 1]
            if np_image.max() <= 1.0:
                np_image = (np_image * 255).astype(np.uint8)
            else:
                np_image = np_image.astype(np.uint8)

        # Для grayscale изображений
        if np_image.ndim == 2:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)

        # Изменение размера для отображения
        h, w = np_image.shape[:2]
        display_w, display_h = self.display_size

        if w != display_w or h != display_h:
            np_image = cv2.resize(np_image, self.display_size, interpolation=cv2.INTER_LINEAR)

        return np_image

    def _overlay_system_info(self, image: np.ndarray, system_state: Dict[str, Any]) -> np.ndarray:
        """
        Наложение системной информации на изображение.

        Args:
            image: Изображение для наложения
            system_state: Состояние системы

        Returns:
            Изображение с наложенной информацией
        """
        overlay = image.copy()
        x, y = self.config['info_position']
        font_scale = self.config['font_scale']
        thickness = self.config['font_thickness']

        # Текущая квалиа
        qualia = system_state.get('current_qualia', 'UNKNOWN')
        color = self.qualia_colors.get(qualia, (255, 255, 255))

        cv2.putText(overlay, f"Qualia: {qualia}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Стабильность и когерентность
        internal = system_state.get('internal_state', {})
        stability = internal.get('stability', 0.0)
        coherence = internal.get('coherence', 0.0)

        y += 30
        cv2.putText(overlay, f"Stability: {stability:.3f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        y += 25
        cv2.putText(overlay, f"Coherence: {coherence:.3f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Итерации
        iterations = system_state.get('iterations', 0)
        y += 25
        cv2.putText(overlay, f"Iterations: {iterations}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), thickness)

        # Режим управления
        control_mode = system_state.get('control_mode', 'manual')
        y += 25
        cv2.putText(overlay, f"Mode: {control_mode}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (150, 150, 150), thickness)

        return overlay

    def _overlay_metrics(self, image: np.ndarray, system_state: Dict[str, Any]) -> np.ndarray:
        """
        Наложение метрик на изображение.

        Args:
            image: Изображение для наложения
            system_state: Состояние системы

        Returns:
            Изображение с наложенными метриками
        """
        overlay = image.copy()
        x, y = self.config['metrics_position']
        font_scale = 0.5
        thickness = 1

        # Метрики визуального анализа (если доступны)
        visual_analysis = system_state.get('visual_analysis', {})
        current_metrics = visual_analysis.get('current_metrics', {})

        if current_metrics:
            cv2.putText(overlay, "Visual Metrics:", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)

            y += 20
            for key, value in list(current_metrics.items())[:6]:  # Первые 6 метрик
                if isinstance(value, float):
                    cv2.putText(overlay, f"{key}: {value:.3f}", (x, y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                    y += 15
                elif isinstance(value, int):
                    cv2.putText(overlay, f"{key}: {value}", (x, y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                    y += 15

        return overlay

    def _update_history(self, system_state: Dict[str, Any]) -> None:
        """Обновление истории для графиков."""
        qualia = system_state.get('current_qualia', 'UNKNOWN')
        internal = system_state.get('internal_state', {})
        stability = internal.get('stability', 0.0)
        iterations = system_state.get('iterations', 0)

        self.qualia_history.append(qualia)
        self.stability_history.append(stability)
        self.iterations_history.append(iterations)

    def save_frame(self, image: torch.Tensor, filename: Optional[str] = None) -> str:
        """
        Сохранение кадра.

        Args:
            image: Изображение для сохранения
            filename: Имя файла (опционально)

        Returns:
            Путь к сохраненному файлу
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{timestamp}.png"

        filepath = self.save_path / filename

        # Конвертация для сохранения
        save_image = self._prepare_image_for_display(image)

        # Сохранение
        cv2.imwrite(str(filepath), cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR))

        return str(filepath)

    def start_recording(self, filename: Optional[str] = None) -> bool:
        """
        Начало записи видео.

        Args:
            filename: Имя видеофайла

        Returns:
            True если запись начата успешно
        """
        if self.recording:
            return False

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.avi"

        filepath = self.save_path / filename

        # Параметры видео
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = 10  # FPS

        self.video_writer = cv2.VideoWriter(
            str(filepath), fourcc, fps, self.display_size
        )

        if self.video_writer.isOpened():
            self.recording = True
            return True
        else:
            self.video_writer = None
            return False

    def stop_recording(self) -> Optional[str]:
        """
        Остановка записи видео.

        Returns:
            Путь к записанному файлу или None
        """
        if not self.recording or not self.video_writer:
            return None

        self.video_writer.release()
        self.video_writer = None
        self.recording = False

        # Возвращаем путь к последнему созданному видео
        video_files = list(self.save_path.glob("recording_*.avi"))
        if video_files:
            return str(sorted(video_files)[-1])

        return None

    def show_history_plot(self, show: bool = True) -> Optional[plt.Figure]:
        """
        Отображение графика истории состояний.

        Args:
            show: Показывать ли график

        Returns:
            Figure matplotlib или None
        """
        if len(self.stability_history) < 2:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # График стабильности
        ax1.plot(list(self.stability_history), 'b-', linewidth=2, label='Stability')
        ax1.set_title('System Stability Over Time')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Stability')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # График итераций
        ax2.plot(list(self.iterations_history), 'r-', linewidth=2, label='Iterations')
        ax2.set_title('Fixed-Point Iterations Over Time')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Iterations')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def create_qualia_timeline(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        Создание таймлайна квалиа.

        Args:
            save_path: Путь для сохранения

        Returns:
            Путь к сохраненному изображению или None
        """
        if len(self.qualia_history) < 2:
            return None

        # Подсчет частоты квалиа
        qualia_counts = {}
        for q in self.qualia_history:
            qualia_counts[q] = qualia_counts.get(q, 0) + 1

        # Сортировка по частоте
        sorted_qualia = sorted(qualia_counts.items(), key=lambda x: x[1], reverse=True)

        # Создание графика
        fig, ax = plt.subplots(figsize=(12, 6))

        qualia_names = [q[0] for q in sorted_qualia]
        counts = [q[1] for q in sorted_qualia]
        colors = [self.qualia_colors.get(q[0], (128, 128, 128)) for q in sorted_qualia]

        # RGB в нормированные значения для matplotlib
        colors = [(r/255, g/255, b/255) for r, g, b in colors]

        bars = ax.bar(range(len(qualia_names)), counts, color=colors, alpha=0.7)

        ax.set_title('Qualia Distribution Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Qualia Type', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_xticks(range(len(qualia_names)))
        ax.set_xticklabels(qualia_names, rotation=45, ha='right')

        # Добавление значений на столбцы
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')

        plt.tight_layout()

        # Сохранение
        if save_path:
            filepath = Path(save_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_path / f"qualia_timeline_{timestamp}.png"

        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def handle_keypress(self) -> Optional[str]:
        """
        Обработка нажатий клавиш.

        Returns:
            Команда или None
        """
        if not self.active:
            return None

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Выход
            return 'quit'
        elif key == ord('s'):  # Сохранение кадра
            return 'save_frame'
        elif key == ord('r'):  # Начало/остановка записи
            return 'toggle_record'
        elif key == ord('p'):  # Показ графика истории
            return 'show_plot'
        elif key == ord('t'):  # Создание таймлайна квалиа
            return 'timeline'
        elif key == 27:  # ESC
            return 'quit'

        return None

    def cleanup(self) -> None:
        """Очистка ресурсов."""
        if self.recording and self.video_writer:
            self.stop_recording()

        if self.active:
            cv2.destroyWindow(self.window_name)

        self.active = False

    def is_active(self) -> bool:
        """Проверка активности визуализатора."""
        return self.active

    def __del__(self) -> None:
        """Деструктор."""
        self.cleanup()