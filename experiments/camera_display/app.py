"""
Camera-Display Experiment App - веб-интерфейс с Gradio.

Интерактивное веб-приложение для эксперимента с петлей камера-дисплей.
Позволяет управлять параметрами камеры и дисплея в реальном времени
и наблюдать за генерацией квалиа.
"""

import gradio as gr
import torch
import numpy as np
import time
import threading
from pathlib import Path
import asyncio
from typing import Dict, Any, Optional, Tuple

from .core import CameraDisplayCore
from .controller import CameraController, ControlMode
from .visualizer import Visualizer

class CameraDisplayApp:
    """
    Gradio приложение для эксперимента камера-дисплей.

    Предоставляет интерактивный интерфейс для:
    - Управления параметрами камеры и дисплея
    - Наблюдения за петлей в реальном времени
    - Анализа квалиа и состояния системы
    - Сохранения результатов
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # Компоненты эксперимента
        self.core = None
        self.controller = None
        self.visualizer = None

        # Состояние приложения
        self.running = False
        self.update_thread = None
        self.current_image = None
        self.system_status = {}

        # История для графиков
        self.history_data = {
            'qualia': [],
            'stability': [],
            'iterations': [],
            'timestamps': []
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию."""
        return {
            'image_size': (256, 256),
            'channels': 3,
            'update_interval': 0.1,  # секунды
            'max_history_points': 100,
            'gradio_theme': 'default',
            'server_port': 7860,
            'server_name': '0.0.0.0'
        }

    def initialize_experiment(self) -> bool:
        """
        Инициализация эксперимента.

        Returns:
            True если инициализация успешна
        """
        try:
            # Конфигурация для компонентов
            core_config = {
                'image_size': self.config['image_size'],
                'channels': self.config['channels'],
                'max_iterations': 24,
                'convergence_threshold': 1e-4,
                'min_iterations': 4,
                'device': 'cpu',  # Для веб-интерфейса используем CPU
                'initial_pattern': 'gradient'
            }

            # Инициализация компонентов
            self.core = CameraDisplayCore(core_config)
            self.controller = CameraController()
            self.visualizer = Visualizer({
                'window_name': 'Camera-Display Web',
                'display_size': (400, 400),
                'show_info': False,  # Информация будет в интерфейсе
                'save_path': 'experiments/camera_display/web_output'
            })

            # Связывание контроллера с ядром
            self.core.camera_params = self.controller.camera_params
            self.core.display_params = self.controller.display_params

            return True

        except Exception as e:
            print(f"[App] Initialization failed: {e}")
            return False

    def start_experiment(self) -> str:
        """Запуск эксперимента."""
        if self.running:
            return "Эксперимент уже запущен"

        if not self.initialize_experiment():
            return "Ошибка инициализации эксперимента"

        self.running = True

        # Запуск потока обновлений
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        return "Эксперимент запущен"

    def stop_experiment(self) -> str:
        """Остановка эксперимента."""
        if not self.running:
            return "Эксперимент не запущен"

        self.running = False

        # Ожидание завершения потока
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)

        # Очистка
        if self.visualizer:
            self.visualizer.cleanup()

        return "Эксперимент остановлен"

    def _update_loop(self) -> None:
        """Основной цикл обновлений."""
        while self.running:
            try:
                if self.core:
                    # Выполнение цикла
                    external_data = {'current_image': self.core.current_image}
                    qualia = self.core.process_cycle(external_data)

                    # Применение влияния квалиа через контроллер
                    self.controller.apply_qualia_influence(qualia)

                    # Получение текущего изображения
                    self.current_image = self.core.get_current_image()

                    # Обновление статуса
                    self.system_status = self.core.get_status()
                    self.system_status['iterations'] = self.core.internal_state.get('iteration_count', 0)

                    # Добавление визуального анализа
                    visual_analysis = self.core.qualia_engine.get_visual_analysis()
                    self.system_status['visual_analysis'] = visual_analysis

                    # Обновление истории
                    self._update_history()

                time.sleep(self.config['update_interval'])

            except Exception as e:
                print(f"[App] Update loop error: {e}")
                time.sleep(1.0)

    def _update_history(self) -> None:
        """Обновление истории для графиков."""
        if not self.system_status:
            return

        timestamp = time.time()

        self.history_data['qualia'].append(self.system_status.get('current_qualia', 'UNKNOWN'))
        self.history_data['stability'].append(self.system_status.get('internal_state', {}).get('stability', 0.0))
        self.history_data['iterations'].append(self.system_status.get('iterations', 0))
        self.history_data['timestamps'].append(timestamp)

        # Ограничение истории
        max_points = self.config['max_history_points']
        for key in self.history_data:
            if len(self.history_data[key]) > max_points:
                self.history_data[key] = self.history_data[key][-max_points:]

    def get_current_image_display(self) -> np.ndarray:
        """
        Получение текущего изображения для отображения в интерфейсе.

        Returns:
            numpy array для Gradio
        """
        if self.current_image is None:
            # Возвращаем пустое изображение
            return np.zeros((256, 256, 3), dtype=np.uint8)

        # Конвертация torch tensor в numpy
        image_np = self.current_image.permute(1, 2, 0).cpu().numpy()

        # Масштабирование к [0, 255]
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        return image_np

    def update_camera_params(self, position_x: float, position_y: float,
                           angle: float, scale: float, distortion: float) -> str:
        """
        Обновление параметров камеры.

        Args:
            position_x, position_y: Позиция (-1, 1)
            angle: Угол поворота (0-360)
            scale: Масштаб (0.1-3.0)
            distortion: Искажения (0-1)

        Returns:
            Статус обновления
        """
        if not self.controller:
            return "Контроллер не инициализирован"

        try:
            self.controller.set_camera_params(
                position=(position_x, position_y),
                angle=angle,
                scale=scale,
                distortion=distortion
            )
            return "Параметры камеры обновлены"
        except Exception as e:
            return f"Ошибка обновления: {e}"

    def update_display_params(self, gamma: float, contrast: float,
                            brightness: float, pixelation: float) -> str:
        """
        Обновление параметров дисплея.

        Args:
            gamma: Гамма-коррекция (0.1-3.0)
            contrast: Контраст (0.1-3.0)
            brightness: Яркость (-1, 1)
            pixelation: Пикселизация (0-1)

        Returns:
            Статус обновления
        """
        if not self.controller:
            return "Контроллер не инициализирован"

        try:
            self.controller.set_display_params(
                gamma=gamma,
                contrast=contrast,
                brightness=brightness,
                pixelation=pixelation
            )
            return "Параметры дисплея обновлены"
        except Exception as e:
            return f"Ошибка обновления: {e}"

    def set_control_mode(self, mode: str) -> str:
        """
        Установка режима управления.

        Args:
            mode: Режим ('manual', 'automatic', 'scripted', 'random')

        Returns:
            Статус установки
        """
        if not self.controller:
            return "Контроллер не инициализирован"

        try:
            mode_enum = ControlMode(mode)
            self.controller.set_control_mode(mode_enum)
            return f"Режим управления установлен: {mode}"
        except Exception as e:
            return f"Ошибка установки режима: {e}"

    def randomize_params(self, intensity: float) -> str:
        """
        Случайная настройка параметров.

        Args:
            intensity: Интенсивность случайных изменений (0-1)

        Returns:
            Статус операции
        """
        if not self.controller:
            return "Контроллер не инициализирован"

        try:
            self.controller.randomize_params(intensity)
            return "Параметры случайным образом изменены"
        except Exception as e:
            return f"Ошибка рандомизации: {e}"

    def save_current_frame(self) -> str:
        """Сохранение текущего кадра."""
        if not self.visualizer or self.current_image is None:
            return "Визуализатор не инициализирован или нет изображения"

        try:
            filepath = self.visualizer.save_frame(self.current_image)
            return f"Кадр сохранен: {filepath}"
        except Exception as e:
            return f"Ошибка сохранения: {e}"

    def start_stop_recording(self) -> str:
        """Переключение записи видео."""
        if not self.visualizer:
            return "Визуализатор не инициализирован"

        try:
            if self.visualizer.recording:
                filepath = self.visualizer.stop_recording()
                return f"Запись остановлена. Файл: {filepath}"
            else:
                success = self.visualizer.start_recording()
                if success:
                    return "Запись начата"
                else:
                    return "Ошибка начала записи"
        except Exception as e:
            return f"Ошибка записи: {e}"

    def get_system_status_text(self) -> str:
        """Получение текстового статуса системы."""
        if not self.system_status:
            return "Система не инициализирована"

        status = self.system_status
        internal = status.get('internal_state', {})

        text = f"""
**Текущая квалиа:** {status.get('current_qualia', 'UNKNOWN')}

**Внутреннее состояние:**
- Стабильность: {internal.get('stability', 0.0):.3f}
- Когерентность: {internal.get('coherence', 0.0):.3f}
- Итерации: {status.get('iterations', 0)}

**Статистика:**
- Всего циклов: {status.get('stats', {}).get('total_cycles', 0)}
- Среднее сходимости: {status.get('stats', {}).get('avg_convergence_time', 0.0):.2f}
"""
        return text.strip()

    def get_qualia_history_plot(self) -> Optional[np.ndarray]:
        """Получение графика истории квалиа."""
        if len(self.history_data['qualia']) < 2:
            return None

        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

            # График стабильности
            ax1.plot(self.history_data['stability'], 'b-', linewidth=2)
            ax1.set_title('Стабильность системы')
            ax1.set_ylabel('Стабильность')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)

            # График итераций
            ax2.plot(self.history_data['iterations'], 'r-', linewidth=2)
            ax2.set_title('Итерации сходимости')
            ax2.set_xlabel('Шаги')
            ax2.set_ylabel('Итерации')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Конвертация в изображение для Gradio
            fig.canvas.draw()
            plot_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

            # Убираем альфа-канал для RGB
            plot_image = plot_image[:, :, :3]

            plt.close(fig)
            return plot_image

        except Exception as e:
            print(f"[App] Plot generation error: {e}")
            return None

    def create_gradio_interface(self):
        """
        Создание Gradio интерфейса.

        Returns:
            Gradio Blocks интерфейс
        """
        with gr.Blocks(title="Camera-Display Experiment", theme=self.config['gradio_theme']) as interface:

            gr.Markdown("# Камера-Дисплей: Рекурсивная Петля Сознания")
            gr.Markdown("*Эксперимент по проверке теории Унитарной Модели Сознания*")

            with gr.Row():
                # Левая колонка - изображение и статус
                with gr.Column(scale=2):
                    image_display = gr.Image(
                        label="Рекурсивная петля",
                        height=400,
                        width=400
                    )

                    status_text = gr.Textbox(
                        label="Статус системы",
                        lines=8,
                        interactive=False
                    )

                # Правая колонка - управление
                with gr.Column(scale=1):
                    gr.Markdown("### Управление экспериментом")

                    # Кнопки запуска/остановки
                    with gr.Row():
                        start_btn = gr.Button("▶️ Запустить", variant="primary")
                        stop_btn = gr.Button("⏹️ Остановить", variant="secondary")

                    start_status = gr.Textbox(label="Статус запуска", interactive=False)

                    # Режим управления
                    control_mode = gr.Radio(
                        choices=["manual", "automatic", "scripted", "random"],
                        value="manual",
                        label="Режим управления"
                    )

                    mode_status = gr.Textbox(label="Статус режима", interactive=False)

                    gr.Markdown("### Параметры камеры")

                    # Параметры камеры
                    with gr.Row():
                        pos_x = gr.Slider(-1, 1, 0, step=0.1, label="Позиция X")
                        pos_y = gr.Slider(-1, 1, 0, step=0.1, label="Позиция Y")

                    with gr.Row():
                        angle = gr.Slider(0, 360, 0, step=5, label="Угол (°)")
                        scale = gr.Slider(0.1, 3.0, 1.0, step=0.1, label="Масштаб")

                    distortion = gr.Slider(0, 1, 0, step=0.05, label="Искажения")

                    camera_status = gr.Textbox(label="Статус камеры", interactive=False)

                    gr.Markdown("### Параметры дисплея")

                    # Параметры дисплея
                    gamma = gr.Slider(0.1, 3.0, 1.0, step=0.1, label="Гамма")
                    contrast = gr.Slider(0.1, 3.0, 1.0, step=0.1, label="Контраст")
                    brightness = gr.Slider(-1, 1, 0, step=0.1, label="Яркость")
                    pixelation = gr.Slider(0, 1, 0, step=0.05, label="Пикселизация")

                    display_status = gr.Textbox(label="Статус дисплея", interactive=False)

                    # Дополнительные функции
                    with gr.Row():
                        randomize_btn = gr.Button("🎲 Рандом")
                        save_btn = gr.Button("💾 Сохранить кадр")
                        record_btn = gr.Button("🎥 Запись")

                    randomize_intensity = gr.Slider(0, 1, 0.5, step=0.1, label="Интенсивность рандома")

                    action_status = gr.Textbox(label="Статус действий", interactive=False)

            # График истории
            with gr.Row():
                history_plot = gr.Image(label="История состояний", height=300)

            # Функции обратного вызова

            # Запуск/остановка
            start_btn.click(
                fn=self.start_experiment,
                outputs=start_status
            )

            stop_btn.click(
                fn=self.stop_experiment,
                outputs=start_status
            )

            # Режим управления
            control_mode.change(
                fn=self.set_control_mode,
                inputs=control_mode,
                outputs=mode_status
            )

            # Параметры камеры
            camera_inputs = [pos_x, pos_y, angle, scale, distortion]
            for input_component in camera_inputs:
                input_component.change(
                    fn=self.update_camera_params,
                    inputs=camera_inputs,
                    outputs=camera_status
                )

            # Параметры дисплея
            display_inputs = [gamma, contrast, brightness, pixelation]
            for input_component in display_inputs:
                input_component.change(
                    fn=self.update_display_params,
                    inputs=display_inputs,
                    outputs=display_status
                )

            # Дополнительные действия
            randomize_btn.click(
                fn=self.randomize_params,
                inputs=randomize_intensity,
                outputs=action_status
            )

            save_btn.click(
                fn=self.save_current_frame,
                outputs=action_status
            )

            record_btn.click(
                fn=self.start_stop_recording,
                outputs=action_status
            )

            # Периодическое обновление интерфейса
            def update_interface():
                current_img = self.get_current_image_display()
                status = self.get_system_status_text()
                plot = self.get_qualia_history_plot()

                return current_img, status, plot

            # Таймер для обновлений
            timer = gr.Timer(0.5)  # Обновление каждые 0.5 секунды
            timer.tick(
                fn=update_interface,
                outputs=[image_display, status_text, history_plot]
            )

        return interface

    def launch(self, **kwargs):
        """
        Запуск приложения.

        Args:
            **kwargs: Параметры для Gradio launch()
        """
        interface = self.create_gradio_interface()

        # Параметры по умолчанию
        launch_kwargs = {
            'server_port': self.config['server_port'],
            'server_name': self.config['server_name'],
            'share': False,
            'show_error': True
        }
        launch_kwargs.update(kwargs)

        interface.launch(**launch_kwargs)

def main():
    """Запуск приложения."""
    app = CameraDisplayApp()
    app.launch()

if __name__ == "__main__":
    main()