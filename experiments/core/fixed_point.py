"""
Fixed-Point Iteration utilities for UMC experiments.

Адаптированные утилиты для fixed-point iteration из benchmark/models/recursive_gpt2.py.
Поддерживают разные типы данных: тензоры, изображения, текстовые векторы.
"""

import torch
import numpy as np
from typing import Any, Tuple, Callable, Optional
import math

# Попытка импорта Triton для ускорения
try:
    from benchmark.cuda.recursive_kernel_triton import (
        FixedPointIterationTriton,
        check_convergence,
        anderson_acceleration_step,
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

class FixedPointIteration:
    """
    Утилиты для fixed-point iteration в экспериментах УМС.

    Поддерживает:
    - Чистый PyTorch
    - Triton-ускорение (если доступно)
    - Разные типы данных
    - Anderson acceleration
    """

    @staticmethod
    def iterate_pytorch(
        f: Callable[[Any], Any],
        initial_state: Any,
        max_iterations: int = 24,
        convergence_threshold: float = 1e-4,
        min_iterations: int = 4,
        training: bool = True,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu'
    ) -> Tuple[Any, int]:
        """
        Fixed-point iteration на чистом PyTorch.

        Args:
            f: Функция итерации f(z) -> z_next
            initial_state: Начальное состояние
            max_iterations: Максимум итераций
            convergence_threshold: Порог сходимости
            min_iterations: Минимум итераций перед проверкой
            training: Режим обучения (влияет на градиенты)
            dtype: Тип данных
            device: Устройство

        Returns:
            (fixed_point, iterations): Сошедшееся состояние и число итераций
        """
        z = initial_state
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).to(dtype=dtype, device=device)

        num_grad_iters = 4  # Только последние N итераций сохраняют градиенты

        for i in range(max_iterations):
            # Отсоединение ранних итераций для экономии памяти
            if training and i < max_iterations - num_grad_iters:
                z = z.detach()

            z_new = f(z)

            # Проверка сходимости после минимума итераций
            if i >= min_iterations:
                with torch.no_grad():
                    if isinstance(z_new, torch.Tensor) and isinstance(z, torch.Tensor):
                        diff = torch.norm(z_new - z) / (torch.norm(z) + 1e-8)
                        if diff < convergence_threshold:
                            break
                    else:
                        # Для не-тензоров используем простую разницу
                        diff = torch.norm(torch.tensor(z_new, dtype=torch.float32) -
                                        torch.tensor(z, dtype=torch.float32))
                        if diff < convergence_threshold:
                            break

            z = z_new

        iterations = i + 1
        return z, iterations

    @staticmethod
    def iterate_triton(
        f: Callable[[torch.Tensor], torch.Tensor],
        initial_state: torch.Tensor,
        max_iterations: int = 24,
        convergence_threshold: float = 1e-4,
        min_iterations: int = 4,
        use_anderson: bool = True,
        training: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """
        Fixed-point iteration с Triton-ускорением.

        Args:
            f: Функция итерации (должна работать с torch.Tensor)
            initial_state: Начальное состояние (torch.Tensor)
            max_iterations: Максимум итераций
            convergence_threshold: Порог сходимости
            min_iterations: Минимум итераций
            use_anderson: Использовать Anderson acceleration
            training: Режим обучения

        Returns:
            (fixed_point, iterations): Сошедшееся состояние и число итераций
        """
        if not TRITON_AVAILABLE:
            return FixedPointIteration.iterate_pytorch(
                f, initial_state, max_iterations, convergence_threshold,
                min_iterations, training
            )

        if not isinstance(initial_state, torch.Tensor):
            raise ValueError("Triton iteration requires torch.Tensor input")

        if not initial_state.is_cuda:
            # Fallback to PyTorch если не на GPU
            return FixedPointIteration.iterate_pytorch(
                f, initial_state, max_iterations, convergence_threshold,
                min_iterations, training
            )

        # Использование Triton-ускорения
        triton_iter = FixedPointIterationTriton(
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            min_iterations=min_iterations,
            use_anderson=use_anderson
        )

        return triton_iter.iterate(f, initial_state)

    @staticmethod
    def iterate(
        f: Callable[[Any], Any],
        initial_state: Any,
        max_iterations: int = 24,
        convergence_threshold: float = 1e-4,
        min_iterations: int = 4,
        training: bool = True,
        use_triton: bool = True,
        use_anderson: bool = True,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu'
    ) -> Tuple[Any, int]:
        """
        Универсальный диспетчер fixed-point iteration.

        Автоматически выбирает между Triton и PyTorch реализациями.

        Args:
            f: Функция итерации
            initial_state: Начальное состояние
            max_iterations: Максимум итераций
            convergence_threshold: Порог сходимости
            min_iterations: Минимум итераций
            training: Режим обучения
            use_triton: Разрешить Triton-ускорение
            use_anderson: Использовать Anderson acceleration
            dtype: Тип данных
            device: Устройство

        Returns:
            (fixed_point, iterations): Сошедшееся состояние и число итераций
        """
        # Проверяем возможность использования Triton
        if (use_triton and TRITON_AVAILABLE and
            isinstance(initial_state, torch.Tensor) and initial_state.is_cuda):

            try:
                return FixedPointIteration.iterate_triton(
                    f, initial_state, max_iterations, convergence_threshold,
                    min_iterations, use_anderson, training
                )
            except Exception as e:
                print(f"[WARNING] Triton iteration failed: {e}, falling back to PyTorch")
                # Fallback to PyTorch

        # PyTorch реализация
        return FixedPointIteration.iterate_pytorch(
            f, initial_state, max_iterations, convergence_threshold,
            min_iterations, training, dtype, device
        )

class ImageFixedPoint:
    """
    Специализированная fixed-point iteration для изображений.

    Используется в эксперименте с камерой-дисплеем.
    """

    @staticmethod
    def iterate(
        image_transform: Callable[[torch.Tensor], torch.Tensor],
        initial_image: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """
        Fixed-point iteration для изображений.

        Args:
            image_transform: Функция преобразования изображения
            initial_image: Начальное изображение (C, H, W) или (H, W)
            **kwargs: Параметры для FixedPointIteration.iterate

        Returns:
            (final_image, iterations): Итоговое изображение и число итераций
        """
        # Нормализация формы
        if initial_image.dim() == 2:  # (H, W)
            initial_image = initial_image.unsqueeze(0)  # (1, H, W)

        # Преобразование в float для вычислений
        if initial_image.dtype != torch.float32:
            initial_image = initial_image.float()

        # Масштабирование к [0, 1] если необходимо
        if initial_image.max() > 1.0:
            initial_image = initial_image / 255.0

        return FixedPointIteration.iterate(
            image_transform,
            initial_image,
            **kwargs
        )

class TextFixedPoint:
    """
    Специализированная fixed-point iteration для текстовых данных.

    Используется в эксперименте с новостями.
    """

    @staticmethod
    def iterate(
        text_processor: Callable[[torch.Tensor], torch.Tensor],
        initial_embedding: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """
        Fixed-point iteration для текстовых эмбеддингов.

        Args:
            text_processor: Функция обработки текста
            initial_embedding: Начальный эмбеддинг текста
            **kwargs: Параметры для FixedPointIteration.iterate

        Returns:
            (final_embedding, iterations): Итоговый эмбеддинг и число итераций
        """
        return FixedPointIteration.iterate(
            text_processor,
            initial_embedding,
            **kwargs
        )

class NumericFixedPoint:
    """
    Специализированная fixed-point iteration для числовых данных.

    Используется в эксперименте с трейдингом.
    """

    @staticmethod
    def iterate(
        numeric_processor: Callable[[torch.Tensor], torch.Tensor],
        initial_values: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """
        Fixed-point iteration для числовых значений.

        Args:
            numeric_processor: Функция обработки чисел
            initial_values: Начальные числовые значения
            **kwargs: Параметры для FixedPointIteration.iterate

        Returns:
            (final_values, iterations): Итоговые значения и число итераций
        """
        return FixedPointIteration.iterate(
            numeric_processor,
            initial_values,
            **kwargs
        )