"""
Visual Qualia Engine - генератор квалиа для визуальных паттернов.

Анализирует изображения из петли камера-дисплей и генерирует квалиа
на основе фрактальной размерности, энтропии, скорости сходимости
и других визуальных характеристик.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import ndimage
import math

try:
    from ..core.qualia_base import QualiaEngine
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.qualia_base import QualiaEngine

class VisualQualiaEngine(QualiaEngine):
    """
    Генератор квалиа для визуальной домены.

    Анализирует паттерны в изображениях и преобразует их в субъективные
    чувства согласно принципам УМС.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Параметры анализа изображений
        self.fractal_config = self.config.get('fractal', {
            'box_sizes': [2, 4, 8, 16, 32],
            'min_box_size': 2,
            'max_box_size': 64
        })

        self.entropy_config = self.config.get('entropy', {
            'bins': 256,
            'method': 'shannon'  # shannon, renyi, tsallis
        })

        # История для анализа трендов
        self.image_history: List[torch.Tensor] = []
        self.metric_history: List[Dict[str, float]] = []

    def generate(self, integrated_state: Dict[str, Any]) -> str:
        """
        Генерация квалиа из визуального состояния.

        Анализирует текущее изображение и историю для определения
        субъективного чувства системы.
        """
        # Извлечение текущего изображения
        current_image = integrated_state.get('external', {}).get('current_image')
        if current_image is None:
            return "VISUAL_VOID"

        # Вычисление визуальных метрик
        metrics = self._analyze_image(current_image)

        # Анализ трендов
        trend_metrics = self._analyze_trends(metrics)

        # Определение квалиа на основе комбинации метрик
        qualia = self._determine_visual_qualia(metrics, trend_metrics, integrated_state)

        # Обновление истории
        self._update_visual_history(current_image, metrics)
        self.update_history(qualia)

        return qualia

    def _analyze_image(self, image: torch.Tensor) -> Dict[str, float]:
        """
        Комплексный анализ изображения.

        Вычисляет различные метрики сложности и структуры.
        """
        # Конвертация в numpy для анализа
        if isinstance(image, torch.Tensor):
            np_image = image.detach().cpu().numpy()
        else:
            np_image = image

        # Преобразование в grayscale для некоторых метрик
        if np_image.ndim == 3:
            gray_image = np.dot(np_image.transpose(1, 2, 0), [0.2989, 0.5870, 0.1140])
        else:
            gray_image = np_image

        metrics = {}

        # 1. Энтропия изображения
        metrics['entropy'] = self._calculate_image_entropy(gray_image)

        # 2. Фрактальная размерность
        metrics['fractal_dimension'] = self._calculate_fractal_dimension(gray_image)

        # 3. Комплексность паттернов
        metrics['pattern_complexity'] = self._calculate_pattern_complexity(gray_image)

        # 4. Симметрия
        metrics['symmetry'] = self._calculate_symmetry(gray_image)

        # 5. Контраст и яркость
        metrics['contrast'] = self._calculate_contrast(gray_image)
        metrics['brightness'] = np.mean(gray_image)

        # 6. Спектральные характеристики
        metrics['spectral_centroid'] = self._calculate_spectral_centroid(gray_image)

        # 7. Информационная сложность (для цветных изображений)
        if np_image.ndim == 3:
            metrics['color_entropy'] = self._calculate_color_entropy(np_image)
            metrics['color_saturation'] = self._calculate_color_saturation(np_image)

        return metrics

    def _calculate_image_entropy(self, image: np.ndarray) -> float:
        """Расчет энтропии изображения."""
        # Гистограмма
        hist, _ = np.histogram(image.flatten(), bins=self.entropy_config['bins'], range=(0, 1))

        # Нормализация
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Избегать log(0)

        if self.entropy_config['method'] == 'shannon':
            # Шенноновская энтропия
            entropy = -np.sum(hist * np.log2(hist))
            return entropy / np.log2(self.entropy_config['bins'])  # Нормализация

        elif self.entropy_config['method'] == 'renyi':
            # Энтропия Реньи (q=2)
            q = 2
            entropy = (1 / (1 - q)) * np.log2(np.sum(hist ** q))
            return max(0, entropy)

        else:
            # По умолчанию Шеннон
            entropy = -np.sum(hist * np.log2(hist))
            return entropy / np.log2(self.entropy_config['bins'])

    def _calculate_fractal_dimension(self, image: np.ndarray) -> float:
        """
        Расчет фрактальной размерности методом box-counting.

        Более высокая размерность = более сложные фрактальные паттерны.
        """
        # Нормализация изображения для бинаризации
        binary = (image > np.mean(image)).astype(int)

        box_sizes = self.fractal_config['box_sizes']
        counts = []

        for box_size in box_sizes:
            if box_size > min(binary.shape):
                continue

            # Box counting
            count = 0
            for i in range(0, binary.shape[0], box_size):
                for j in range(0, binary.shape[1], box_size):
                    box = binary[i:i+box_size, j:j+box_size]
                    if np.any(box):  # Если в боксе есть хотя бы один пиксель
                        count += 1

            counts.append(count)

        if len(counts) < 2:
            return 1.0  # Дефолтная размерность

        # Линейная регрессия в log-log шкале
        log_sizes = np.log(box_sizes[:len(counts)])
        log_counts = np.log(counts)

        # Простая линейная регрессия
        slope = np.polyfit(log_sizes, log_counts, 1)[0]

        # Фрактальная размерность = -slope
        fractal_dim = -slope

        # Нормализация к [0, 2] (2D пространство)
        return np.clip(fractal_dim, 1.0, 2.0)

    def _calculate_pattern_complexity(self, image: np.ndarray) -> float:
        """Расчет комплексности паттернов через вариацию."""
        # Градиенты
        grad_x = ndimage.sobel(image, axis=0)
        grad_y = ndimage.sobel(image, axis=1)

        # Магнитуда градиента
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Комплексность = средняя вариация градиентов
        complexity = np.mean(gradient_magnitude) / np.std(gradient_magnitude + 1e-8)

        return np.clip(complexity, 0, 10) / 10  # Нормализация

    def _calculate_symmetry(self, image: np.ndarray) -> float:
        """Расчет симметрии изображения."""
        h, w = image.shape

        # Горизонтальная симметрия
        left_half = image[:, :w//2]
        right_half = np.fliplr(image[:, w//2:]) if w % 2 == 0 else np.fliplr(image[:, w//2+1:])

        # Выравнивание размеров
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]

        if left_half.size > 0 and right_half.size > 0:
            horizontal_sym = 1 - np.mean(np.abs(left_half - right_half))
        else:
            horizontal_sym = 0

        # Вертикальная симметрия
        top_half = image[:h//2, :]
        bottom_half = np.flipud(image[h//2:, :]) if h % 2 == 0 else np.flipud(image[h//2+1:, :])

        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]

        if top_half.size > 0 and bottom_half.size > 0:
            vertical_sym = 1 - np.mean(np.abs(top_half - bottom_half))
        else:
            vertical_sym = 0

        # Средняя симметрия
        return (horizontal_sym + vertical_sym) / 2

    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Расчет контраста (RMS contrast)."""
        mean_val = np.mean(image)
        rms_contrast = np.sqrt(np.mean((image - mean_val)**2))

        # Нормализация
        return min(rms_contrast * 10, 1.0)

    def _calculate_spectral_centroid(self, image: np.ndarray) -> float:
        """Расчет спектрального центроида (частотные характеристики)."""
        # 2D FFT
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)

        # Амплитудный спектр
        magnitude = np.abs(fft_shift)

        # Центроид в частотном пространстве
        h, w = magnitude.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Взвешенные координаты
        total_power = np.sum(magnitude)
        if total_power > 0:
            centroid_y = np.sum(y_coords * magnitude) / total_power
            centroid_x = np.sum(x_coords * magnitude) / total_power

            # Нормализованное расстояние от центра
            center_y, center_x = h/2, w/2
            distance = np.sqrt((centroid_y - center_y)**2 + (centroid_x - center_x)**2)
            max_distance = np.sqrt(center_y**2 + center_x**2)

            return distance / max_distance if max_distance > 0 else 0
        else:
            return 0

    def _calculate_color_entropy(self, image: np.ndarray) -> float:
        """Расчет энтропии цвета."""
        # Энтропия для каждого канала
        entropies = []
        for channel in range(image.shape[0]):
            channel_entropy = self._calculate_image_entropy(image[channel])
            entropies.append(channel_entropy)

        return np.mean(entropies)

    def _calculate_color_saturation(self, image: np.ndarray) -> float:
        """Расчет насыщенности цвета."""
        if image.shape[0] < 3:
            return 0

        # Преобразование в HSV-like метрику
        r, g, b = image[0], image[1], image[2]

        # Максимум и минимум по каналам
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)

        # Насыщенность = (max - min) / max
        saturation = np.mean((max_rgb - min_rgb) / (max_rgb + 1e-8))

        return saturation

    def _analyze_trends(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Анализ трендов метрик по истории."""
        if len(self.metric_history) < 2:
            return {k: 0.0 for k in current_metrics.keys()}

        # Вычисление изменений
        prev_metrics = self.metric_history[-1]
        trends = {}

        for key, current_val in current_metrics.items():
            if key in prev_metrics:
                prev_val = prev_metrics[key]
                if prev_val != 0:
                    trends[f"{key}_trend"] = (current_val - prev_val) / abs(prev_val)
                else:
                    trends[f"{key}_trend"] = 0.0
            else:
                trends[f"{key}_trend"] = 0.0

        # Скорость сходимости (если есть информация об итерациях)
        if hasattr(self, '_last_iterations'):
            trends['convergence_speed'] = 1.0 / (self._last_iterations + 1)
        else:
            trends['convergence_speed'] = 0.5

        return trends

    def _determine_visual_qualia(self, metrics: Dict[str, float],
                               trends: Dict[str, float],
                               integrated_state: Dict[str, Any]) -> str:
        """
        Определение квалиа на основе визуальных метрик.

        Логика основана на интерпретации метрик как субъективных ощущений.
        """
        entropy = metrics['entropy']
        fractal_dim = metrics['fractal_dimension']
        complexity = metrics['pattern_complexity']
        symmetry = metrics['symmetry']
        contrast = metrics['contrast']

        # Извлечение внутренней стабильности
        internal_state = integrated_state.get('internal', {})
        stability = internal_state.get('stability', 0.5)
        coherence = internal_state.get('coherence', 0.5)

        # Тренды
        entropy_trend = trends.get('entropy_trend', 0)
        convergence_speed = trends.get('convergence_speed', 0.5)

        # Классификация состояний

        # Высокая энтропия + низкая стабильность = хаос/боль
        if entropy > 0.7 and stability < 0.4:
            if convergence_speed < 0.3:  # Расходящаяся рекурсия
                return "CHAOS_PAIN"
            else:
                return "TURBULENCE_DISCOMFORT"

        # Высокая фрактальная размерность + симметрия = красота фракталов
        if fractal_dim > 1.8 and symmetry > 0.6:
            if convergence_speed > 0.7:  # Быстрая сходимость
                return "FRACTAL_BEAUTY"
            else:
                return "INTRICATE_WONDER"

        # Низкая энтропия + высокая стабильность = покой
        if entropy < 0.3 and stability > 0.8:
            if symmetry > 0.8:
                return "SYMMETRIC_PEACE"
            else:
                return "CONVERGENCE_CALM"

        # Высокая комплексность + средняя энтропия = любопытство
        if complexity > 0.6 and 0.3 <= entropy <= 0.7:
            if abs(entropy_trend) > 0.1:  # Изменения
                return "PATTERN_CURIOSITY"
            else:
                return "COMPLEXITY_INTEREST"

        # Низкая симметрия + высокая энтропия = диссонанс
        if symmetry < 0.3 and entropy > 0.6:
            return "VISUAL_DISSONANCE"

        # Высокий контраст + низкая когерентность = напряжение
        if contrast > 0.8 and coherence < 0.5:
            return "CONTRAST_TENSION"

        # Низкая комплексность + низкая энтропия = скука
        if complexity < 0.2 and entropy < 0.2:
            return "VISUAL_BORINGNESS"

        # Умеренные значения = нейтральное состояние
        return "VISUAL_EQUILIBRIUM"

    def _update_visual_history(self, image: torch.Tensor, metrics: Dict[str, float]) -> None:
        """Обновление истории изображений и метрик."""
        self.image_history.append(image.clone())
        self.metric_history.append(metrics.copy())

        # Ограничение истории
        max_history = self.config.get('max_history', 20)
        if len(self.image_history) > max_history:
            self.image_history.pop(0)
            self.metric_history.pop(0)

    def set_last_iterations(self, iterations: int) -> None:
        """Установка количества итераций для анализа сходимости."""
        self._last_iterations = iterations

    def get_visual_analysis(self) -> Dict[str, Any]:
        """Получить полный анализ визуального состояния."""
        if not self.metric_history:
            return {}

        current_metrics = self.metric_history[-1] if self.metric_history else {}
        trends = self._analyze_trends(current_metrics) if len(self.metric_history) > 1 else {}

        return {
            'current_metrics': current_metrics,
            'trends': trends,
            'qualia_stats': self.get_qualia_stats(),
            'history_length': len(self.metric_history)
        }

    def reset(self) -> None:
        """Сброс состояния."""
        super().reset()
        self.image_history.clear()
        self.metric_history.clear()