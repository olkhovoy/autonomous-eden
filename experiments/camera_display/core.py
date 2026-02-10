"""
Camera-Display Core - рекурсивный движок для визуального эксперимента.

Реализует петлю обратной связи камера-дисплей, где изображение
с экрана снимается камерой, обрабатывается через fixed-point iteration,
и выводится обратно на экран.
"""

import torch
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional, Callable
from pathlib import Path
import cv2

try:
    from ..core.unitary_state import UnitaryState
    from ..core.fixed_point import ImageFixedPoint
    from .qualia_engine import VisualQualiaEngine
    from .controller import CameraParams, DisplayParams
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.unitary_state import UnitaryState
    from core.fixed_point import ImageFixedPoint
    from camera_display.qualia_engine import VisualQualiaEngine
    from camera_display.controller import CameraParams, DisplayParams

class CameraDisplayCore(UnitaryState):
    """
    Рекурсивный движок камера-дисплей.

    Реализует принцип УМС: камера снимает экран, на который выводится
    изображение с этой камеры. Через fixed-point iteration достигается
    рекурсивное замыкание, генерирующее квалиа из визуальных паттернов.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Camera_Display_Core", config)

        # Визуальные параметры
        self.image_size = self.config.get('image_size', (256, 256))
        self.channels = self.config.get('channels', 3)  # RGB

        # Текущее изображение в петле
        self.current_image = self._generate_initial_image()
        self.previous_image = None

        # Генератор квалиа
        self.qualia_engine = VisualQualiaEngine(config)

        # История изображений для анализа
        self.image_history = []
        self.max_history = self.config.get('max_history', 10)

    def _generate_initial_image(self) -> torch.Tensor:
        """Генерация начального изображения для петли."""
        if self.config.get('initial_pattern') == 'noise':
            # Случайный шум
            image = torch.randn(self.channels, *self.image_size)
        elif self.config.get('initial_pattern') == 'gradient':
            # Градиент
            x = torch.linspace(-1, 1, self.image_size[1])
            y = torch.linspace(-1, 1, self.image_size[0])
            X, Y = torch.meshgrid(x, y, indexing='ij')
            image = torch.stack([X, Y, (X + Y) / 2])
        else:
            # По умолчанию - серый шум
            image = torch.randn(self.channels, *self.image_size) * 0.1 + 0.5

        # Нормализация к [0, 1]
        image = torch.clamp(image, 0, 1)
        return image

    def _fixed_point_iteration(self, integrated_field: Dict[str, Any]) -> Tuple[Any, int]:
        """
        Fixed-point iteration для изображений.

        Функция итерации: берет текущее изображение, применяет трансформации
        (камера + эффекты), и возвращает новое изображение.
        """
        def image_iteration(current_img: torch.Tensor) -> torch.Tensor:
            """
            Одна итерация петли камера-дисплей.

            Симулирует процесс: изображение → камера → обработка → экран → следующая итерация
            """
            # 1. Применение эффектов камеры (из контроллера)
            camera_transformed = self._apply_camera_effects(current_img)

            # 2. Симуляция дисплея (некоторые эффекты отображения)
            display_output = self._apply_display_effects(camera_transformed)

            # 3. Добавление шума/искажений для реализма
            with_noise = self._add_realism_noise(display_output)

            # 4. Интеграция с внутренним состоянием (квалиа влияет на изображение)
            qualia_influenced = self._apply_qualia_influence(with_noise)

            return qualia_influenced

        # Запуск fixed-point iteration
        final_image, iterations = ImageFixedPoint.iterate(
            image_iteration,
            self.current_image.clone(),
            max_iterations=self.config['max_iterations'],
            convergence_threshold=self.config['convergence_threshold'],
            min_iterations=self.config['min_iterations'],
            training=False,  # Для экспериментов не обучаем
            device=self.config.get('device', 'cpu')
        )

        # Обновление истории изображений
        self._update_image_history(final_image, iterations)

        return final_image, iterations

    def _apply_camera_effects(self, image: torch.Tensor) -> torch.Tensor:
        """
        Применение эффектов камеры: позиция, угол, масштаб, искажения.
        """
        # Получение параметров камеры из контроллера
        camera_params = getattr(self, 'camera_params', None)
        if camera_params is None:
            # Используем значения по умолчанию если параметры не установлены
            scale_val = 1.0
            angle_val = 0.0
            distortion_val = 0.0
        else:
            # camera_params - это объект dataclass, получаем атрибуты
            scale_val = getattr(camera_params, 'scale', 1.0)
            angle_val = getattr(camera_params, 'angle', 0.0)
            distortion_val = getattr(camera_params, 'distortion', 0.0)

        transformed = image.clone()

        # Масштабирование
        if scale_val != 1.0:
            transformed = torch.nn.functional.interpolate(
                transformed.unsqueeze(0),
                scale_factor=scale_val,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # Обрезка/паддинг до исходного размера
            h, w = transformed.shape[1:]
            target_h, target_w = self.image_size

            if h > target_h:
                start_h = (h - target_h) // 2
                transformed = transformed[:, start_h:start_h+target_h, :]
            if w > target_w:
                start_w = (w - target_w) // 2
                transformed = transformed[:, :, start_w:start_w+target_w]

        # Поворот (простая аппроксимация)
        if angle_val != 0:
            angle_rad = torch.tensor(angle_val * np.pi / 180)
            cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)

            # Простой поворот через affine transform
            rotation_matrix = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0]
            ]).float()

            grid = torch.nn.functional.affine_grid(
                rotation_matrix.unsqueeze(0),
                transformed.unsqueeze(0).shape,
                align_corners=False
            )
            transformed = torch.nn.functional.grid_sample(
                transformed.unsqueeze(0),
                grid,
                align_corners=False
            ).squeeze(0)

        # Искажения линзы
        if distortion_val > 0:
            transformed = self._apply_lens_distortion(transformed, distortion_val)

        return transformed

    def _apply_lens_distortion(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """Применение искажений линзы."""
        h, w = image.shape[1:]

        # Создание сетки координат
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij'
        )

        # Радиус от центра
        r = torch.sqrt(x_coords**2 + y_coords**2)

        # Баррелет distortion
        distortion = 1 + strength * (r**2)

        # Новые координаты
        x_distorted = x_coords * distortion
        y_distorted = y_coords * distortion

        # Нормализация к [-1, 1]
        x_distorted = torch.clamp(x_distorted, -1, 1)
        y_distorted = torch.clamp(y_distorted, -1, 1)

        # Интерполяция
        grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)

        distorted = torch.nn.functional.grid_sample(
            image.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        ).squeeze(0)

        return distorted

    def _apply_display_effects(self, image: torch.Tensor) -> torch.Tensor:
        """Применение эффектов дисплея: гамма, контраст, пикселизация."""
        display_params = getattr(self, 'display_params', None)
        if display_params is None:
            # Используем значения по умолчанию если параметры не установлены
            gamma_val = 1.0
            contrast_val = 1.0
            brightness_val = 0.0
            pixelation_val = 0.0
        else:
            # display_params - это объект dataclass, получаем атрибуты
            gamma_val = getattr(display_params, 'gamma', 1.0)
            contrast_val = getattr(display_params, 'contrast', 1.0)
            brightness_val = getattr(display_params, 'brightness', 0.0)
            pixelation_val = getattr(display_params, 'pixelation', 0.0)

        transformed = image.clone()

        # Гамма-коррекция
        if gamma_val != 1.0:
            transformed = torch.pow(transformed.clamp(0, 1), 1/gamma_val)

        # Контраст и яркость
        if contrast_val != 1.0 or brightness_val != 0.0:
            transformed = transformed * contrast_val + brightness_val

        # Пикселизация
        if pixelation_val > 0:
            block_size = max(1, int(pixelation_val * min(self.image_size)))
            h, w = transformed.shape[1:]

            # Downsample
            small_h, small_w = h // block_size, w // block_size
            downsampled = torch.nn.functional.interpolate(
                transformed.unsqueeze(0),
                size=(small_h, small_w),
                mode='nearest'
            )

            # Upsample back
            transformed = torch.nn.functional.interpolate(
                downsampled,
                size=(h, w),
                mode='nearest'
            ).squeeze(0)

        return torch.clamp(transformed, 0, 1)

    def _add_realism_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Добавление реалистичных шумов и артефактов."""
        noise_config = self.config.get('noise', {
            'gaussian': 0.01,
            'salt_pepper': 0.001,
            'chromatic_aberration': 0.0
        })

        noisy = image.clone()

        # Гауссов шум
        if noise_config['gaussian'] > 0:
            noise = torch.randn_like(noisy) * noise_config['gaussian']
            noisy = noisy + noise

        # Salt & pepper noise
        if noise_config['salt_pepper'] > 0:
            prob = noise_config['salt_pepper']
            salt_pepper = torch.rand_like(noisy) < prob
            noisy = torch.where(salt_pepper, torch.rand_like(noisy), noisy)

        # Хроматическая аберрация
        if noise_config['chromatic_aberration'] > 0:
            shift = int(noise_config['chromatic_aberration'] * self.image_size[1])
            if self.channels >= 3:
                # Красный канал смещается вправо
                noisy[0] = torch.roll(noisy[0], shift, dims=1)
                # Синий канал смещается влево
                noisy[2] = torch.roll(noisy[2], -shift, dims=1)

        return torch.clamp(noisy, 0, 1)

    def _apply_qualia_influence(self, image: torch.Tensor) -> torch.Tensor:
        """Влияние квалиа на изображение (нисходящая причинность)."""
        qualia = self.current_qualia

        influenced = image.clone()

        # Влияние разных квалиа на визуальные паттерны
        if 'CHAOS' in qualia:
            # Хаос добавляет турбулентность
            turbulence = torch.randn_like(influenced) * 0.1
            influenced = influenced + turbulence

        elif 'FRACTAL' in qualia:
            # Фракталы усиливают паттерны
            influenced = self._enhance_fractal_patterns(influenced)

        elif 'PEACE' in qualia:
            # Мир добавляет плавность
            influenced = self._smooth_image(influenced, sigma=0.5)

        elif 'PAIN' in qualia:
            # Боль добавляет резкие контрасты
            influenced = self._increase_contrast(influenced, factor=1.5)

        return torch.clamp(influenced, 0, 1)

    def _enhance_fractal_patterns(self, image: torch.Tensor) -> torch.Tensor:
        """Усиление фрактальных паттернов."""
        # Простое усиление через частотный фильтр
        # (в реальности можно использовать более сложные методы)
        return image * 1.2

    def _smooth_image(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """Сглаживание изображения."""
        # Простое гауссово размытие через свертку
        kernel_size = int(sigma * 6) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Создание гауссового ядра
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        # Применение к каждому каналу
        smoothed = []
        for c in range(image.shape[0]):
            channel = image[c:c+1]  # (1, H, W)
            smoothed_c = torch.nn.functional.conv2d(
                channel.unsqueeze(0),
                g.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * g.unsqueeze(0).unsqueeze(-1).unsqueeze(0),
                padding=kernel_size//2
            ).squeeze(0)
            smoothed.append(smoothed_c)

        return torch.cat(smoothed, dim=0)

    def _increase_contrast(self, image: torch.Tensor, factor: float) -> torch.Tensor:
        """Увеличение контраста."""
        # Простая контрастная коррекция
        mean_val = image.mean()
        return torch.clamp((image - mean_val) * factor + mean_val, 0, 1)

    def _update_image_history(self, image: torch.Tensor, iterations: int) -> None:
        """Обновление истории изображений."""
        self.image_history.append({
            'image': image.clone(),
            'iterations': iterations,
            'qualia': self.current_qualia,
            'timestamp': torch.tensor(time.time())
        })

        if len(self.image_history) > self.max_history:
            self.image_history.pop(0)

    def generate_qualia(self, integrated_state: Dict[str, Any]) -> str:
        """Генерация квалиа на основе визуального состояния."""
        return self.qualia_engine.generate(integrated_state)

    def set_camera_params(self, position: Tuple[float, float] = None,
                         angle: float = None, scale: float = None,
                         distortion: float = None) -> None:
        """Установка параметров камеры."""
        if not hasattr(self, 'camera_params'):
            self.camera_params = CameraParams()

        # Создаем новый объект с обновленными параметрами
        current_position = getattr(self.camera_params, 'position', (0.0, 0.0))
        current_angle = getattr(self.camera_params, 'angle', 0.0)
        current_scale = getattr(self.camera_params, 'scale', 1.0)
        current_distortion = getattr(self.camera_params, 'distortion', 0.0)

        self.camera_params = CameraParams(
            position=position if position is not None else current_position,
            angle=angle if angle is not None else current_angle,
            scale=scale if scale is not None else current_scale,
            distortion=distortion if distortion is not None else current_distortion
        )

    def set_display_params(self, gamma: float = None, contrast: float = None,
                          brightness: float = None, pixelation: float = None) -> None:
        """Установка параметров дисплея."""
        if not hasattr(self, 'display_params'):
            self.display_params = DisplayParams()

        # Создаем новый объект с обновленными параметрами
        current_gamma = getattr(self.display_params, 'gamma', 1.0)
        current_contrast = getattr(self.display_params, 'contrast', 1.0)
        current_brightness = getattr(self.display_params, 'brightness', 0.0)
        current_pixelation = getattr(self.display_params, 'pixelation', 0.0)

        self.display_params = DisplayParams(
            gamma=gamma if gamma is not None else current_gamma,
            contrast=contrast if contrast is not None else current_contrast,
            brightness=brightness if brightness is not None else current_brightness,
            pixelation=pixelation if pixelation is not None else current_pixelation
        )

    def get_current_image(self) -> torch.Tensor:
        """Получить текущее изображение из петли."""
        return self.current_image.clone()

    def get_image_history(self) -> list:
        """Получить историю изображений."""
        return self.image_history.copy()

    def save_image(self, path: str, image: torch.Tensor = None) -> None:
        """Сохранить изображение."""
        if image is None:
            image = self.current_image

        # Конвертация в numpy и сохранение
        np_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Для grayscale изображений
        if np_image.shape[2] == 1:
            np_image = np_image.squeeze(2)

        cv2.imwrite(path, cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR) if np_image.ndim == 3 else np_image)

    def reset(self) -> None:
        """Сброс состояния."""
        super().reset()
        self.current_image = self._generate_initial_image()
        self.previous_image = None
        self.image_history.clear()
        self.qualia_engine.reset()