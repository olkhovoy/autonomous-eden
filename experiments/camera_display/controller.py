"""
Camera Controller - управление параметрами камеры и дисплея.

Позволяет интерактивно изменять параметры камеры (позиция, угол, масштаб)
и дисплея (гамма, контраст), влияя на рекурсивную петлю камера-дисплей.
"""

import torch
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class ControlMode(Enum):
    """Режимы управления."""
    MANUAL = "manual"        # Ручное управление
    AUTOMATIC = "automatic"  # Автоматическое (на основе квалиа)
    SCRIPTED = "scripted"    # Сценарии
    RANDOM = "random"        # Случайные изменения

@dataclass
class CameraParams:
    """Параметры камеры."""
    position: Tuple[float, float] = (0.0, 0.0)  # Смещение по x,y (-1, 1)
    angle: float = 0.0                          # Угол поворота (градусы)
    scale: float = 1.0                           # Масштаб (0.1, 3.0)
    distortion: float = 0.0                      # Искажения линзы (0, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'position': self.position,
            'angle': self.angle,
            'scale': self.scale,
            'distortion': self.distortion
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraParams':
        return cls(
            position=data.get('position', (0.0, 0.0)),
            angle=data.get('angle', 0.0),
            scale=data.get('scale', 1.0),
            distortion=data.get('distortion', 0.0)
        )

@dataclass
class DisplayParams:
    """Параметры дисплея."""
    gamma: float = 1.0           # Гамма-коррекция (0.1, 3.0)
    contrast: float = 1.0        # Контраст (0.1, 3.0)
    brightness: float = 0.0      # Яркость (-1, 1)
    pixelation: float = 0.0      # Пикселизация (0, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'gamma': self.gamma,
            'contrast': self.contrast,
            'brightness': self.brightness,
            'pixelation': self.pixelation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DisplayParams':
        return cls(
            gamma=data.get('gamma', 1.0),
            contrast=data.get('contrast', 1.0),
            brightness=data.get('brightness', 0.0),
            pixelation=data.get('pixelation', 0.0)
        )

class CameraController:
    """
    Контроллер камеры и дисплея для эксперимента камера-дисплей.

    Управляет параметрами, которые влияют на рекурсивную петлю,
    создавая обратную связь между субъективным состоянием (квалиа)
    и объективными параметрами системы.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # Текущие параметры
        self.camera_params = CameraParams()
        self.display_params = DisplayParams()

        # История параметров
        self.param_history: list = []
        self.max_history = self.config.get('max_history', 100)

        # Режим управления
        self.control_mode = ControlMode.MANUAL

        # Автоматическое управление
        self.auto_rules: Dict[str, Callable] = {}
        self._setup_auto_rules()

        # Сценарии
        self.scripts: Dict[str, list] = {}
        self.current_script: Optional[str] = None
        self.script_step = 0

    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию."""
        return {
            'max_history': 100,
            'auto_sensitivity': 0.5,
            'script_speed': 1.0,
            'smooth_transitions': True,
            'transition_steps': 10
        }

    def _setup_auto_rules(self) -> None:
        """Настройка правил автоматического управления на основе квалиа."""
        self.auto_rules = {
            'CHAOS_PAIN': self._rule_chaos_pain,
            'FRACTAL_BEAUTY': self._rule_fractal_beauty,
            'CONVERGENCE_PEACE': self._rule_convergence_peace,
            'VISUAL_DISSONANCE': self._rule_visual_dissonance,
            'TURBULENCE_DISCOMFORT': self._rule_turbulence_discomfort,
            'SYMMETRIC_PEACE': self._rule_symmetric_peace,
            'PATTERN_CURIOSITY': self._rule_pattern_curiosity,
            'CONTRAST_TENSION': self._rule_contrast_tension
        }

    # Правила автоматического управления

    def _rule_chaos_pain(self) -> None:
        """Правило для состояния CHAOS_PAIN: стабилизация."""
        # Уменьшение искажений, нормализация масштаба
        self.camera_params.distortion = max(0, self.camera_params.distortion - 0.1)
        self.camera_params.scale = 0.8 + 0.4 * torch.rand(1).item()  # Случайный масштаб 0.8-1.2
        self.camera_params.angle = 0  # Выравнивание

        # Нормализация дисплея
        self.display_params.gamma = 1.0
        self.display_params.contrast = 1.0
        self.display_params.brightness = 0.0

    def _rule_fractal_beauty(self) -> None:
        """Правило для состояния FRACTAL_BEAUTY: усиление паттернов."""
        # Небольшие искажения для создания интересных паттернов
        self.camera_params.distortion = 0.3 + 0.2 * torch.rand(1).item()
        self.camera_params.scale = 1.2 + 0.3 * torch.rand(1).item()  # Увеличение 1.2-1.5

        # Усиление контраста для выделения паттернов
        self.display_params.contrast = 1.5
        self.display_params.gamma = 0.8

    def _rule_convergence_peace(self) -> None:
        """Правило для состояния CONVERGENCE_PEACE: поддержание стабильности."""
        # Минимальные изменения, поддержание текущих параметров
        perturbation = 0.05 * (torch.rand(4) - 0.5)  # Маленькие случайные изменения

        self.camera_params.scale = np.clip(
            self.camera_params.scale + perturbation[0].item(), 0.5, 2.0
        )
        self.camera_params.angle = (self.camera_params.angle + perturbation[1].item() * 10) % 360
        self.camera_params.distortion = np.clip(
            self.camera_params.distortion + perturbation[2].item(), 0, 0.5
        )

        # Плавные изменения дисплея
        self.display_params.gamma = np.clip(
            self.display_params.gamma + perturbation[3].item() * 0.2, 0.5, 1.5
        )

    def _rule_visual_dissonance(self) -> None:
        """Правило для состояния VISUAL_DISSONANCE: поиск гармонии."""
        # Случайные изменения для выхода из диссонанса
        self.camera_params.angle = torch.rand(1).item() * 360
        self.camera_params.scale = 0.7 + 0.6 * torch.rand(1).item()
        self.camera_params.distortion = 0.2 * torch.rand(1).item()

        # Сброс дисплея к нейтральным значениям
        self.display_params.contrast = 1.0
        self.display_params.brightness = 0.0

    def _rule_turbulence_discomfort(self) -> None:
        """Правило для состояния TURBULENCE_DISCOMFORT: успокоение."""
        # Постепенное уменьшение турбулентности
        self.camera_params.distortion *= 0.9
        self.camera_params.scale = 1.0 + 0.1 * (self.camera_params.scale - 1.0)  # К 1.0
        self.camera_params.angle *= 0.95  # Постепенное выравнивание

        # Успокаивающие настройки дисплея
        self.display_params.contrast = 1.0
        self.display_params.gamma = 1.0

    def _rule_symmetric_peace(self) -> None:
        """Правило для состояния SYMMETRIC_PEACE: поддержание симметрии."""
        # Поддержание симметричных настроек
        self.camera_params.angle = 0  # Выравнивание
        self.camera_params.position = (0.0, 0.0)  # Центрирование
        self.camera_params.scale = 1.0

        # Минимальные искажения
        self.camera_params.distortion = min(self.camera_params.distortion, 0.1)

    def _rule_pattern_curiosity(self) -> None:
        """Правило для состояния PATTERN_CURIOSITY: исследование."""
        # Случайные изменения для исследования новых паттернов
        changes = 0.1 * (torch.rand(4) - 0.5)

        self.camera_params.angle = (self.camera_params.angle + changes[0].item() * 30) % 360
        self.camera_params.scale = np.clip(self.camera_params.scale + changes[1].item(), 0.5, 2.0)
        self.camera_params.distortion = np.clip(self.camera_params.distortion + changes[2].item(), 0, 0.8)

        self.display_params.contrast = np.clip(self.display_params.contrast + changes[3].item(), 0.5, 2.0)

    def _rule_contrast_tension(self) -> None:
        """Правило для состояния CONTRAST_TENSION: смягчение."""
        # Снижение контраста и напряжения
        self.display_params.contrast = max(0.7, self.display_params.contrast * 0.95)
        self.camera_params.distortion *= 0.9

        # Добавление небольшого размытия для смягчения
        self.display_params.pixelation = min(0.3, self.display_params.pixelation + 0.05)

    # Основные методы управления

    def set_camera_params(self, **params) -> None:
        """Установка параметров камеры."""
        for key, value in params.items():
            if hasattr(self.camera_params, key):
                setattr(self.camera_params, key, value)

        self._update_history()

    def set_display_params(self, **params) -> None:
        """Установка параметров дисплея."""
        for key, value in params.items():
            if hasattr(self.display_params, key):
                setattr(self.display_params, key, value)

        self._update_history()

    def set_control_mode(self, mode: ControlMode) -> None:
        """Установка режима управления."""
        self.control_mode = mode

        if mode == ControlMode.SCRIPTED and not self.current_script:
            self.current_script = list(self.scripts.keys())[0] if self.scripts else None
            self.script_step = 0

    def apply_qualia_influence(self, qualia: str) -> None:
        """
        Применение влияния квалиа на параметры (автоматическое управление).

        Если включен автоматический режим, квалиа влияет на параметры системы.
        """
        if self.control_mode != ControlMode.AUTOMATIC:
            return

        if qualia in self.auto_rules:
            # Применение правила с чувствительностью
            sensitivity = self.config['auto_sensitivity']
            if torch.rand(1).item() < sensitivity:
                self.auto_rules[qualia]()

                # Плавные переходы
                if self.config['smooth_transitions']:
                    self._apply_smooth_transitions()

                self._update_history()

    def _apply_smooth_transitions(self) -> None:
        """Применение плавных переходов к изменениям."""
        # Простая интерполяция к новым значениям
        steps = self.config['transition_steps']

        # Для демонстрации - мгновенные изменения, но в реальности
        # можно реализовать постепенные переходы
        pass

    def create_script(self, name: str, script_data: list) -> None:
        """
        Создание сценария управления.

        Args:
            name: Имя сценария
            script_data: Список шагов сценария
        """
        self.scripts[name] = script_data

    def play_script_step(self) -> bool:
        """
        Выполнение следующего шага сценария.

        Returns:
            True если сценарий продолжается, False если закончен
        """
        if not self.current_script or self.current_script not in self.scripts:
            return False

        script = self.scripts[self.current_script]
        if self.script_step >= len(script):
            return False

        step = script[self.script_step]

        # Применение шага
        if 'camera' in step:
            self.set_camera_params(**step['camera'])
        if 'display' in step:
            self.set_display_params(**step['display'])
        if 'delay' in step:
            # В реальном приложении здесь будет asyncio.sleep или time.sleep
            pass

        self.script_step += 1
        return self.script_step < len(script)

    def randomize_params(self, intensity: float = 0.5) -> None:
        """
        Случайная настройка параметров.

        Args:
            intensity: Интенсивность случайных изменений (0-1)
        """
        # Камера
        self.camera_params.position = (
            intensity * 2 * (torch.rand(1).item() - 0.5),
            intensity * 2 * (torch.rand(1).item() - 0.5)
        )
        self.camera_params.angle = intensity * 360 * torch.rand(1).item()
        self.camera_params.scale = 1.0 + intensity * (torch.rand(1).item() - 0.5) * 2
        self.camera_params.distortion = intensity * torch.rand(1).item()

        # Дисплей
        self.display_params.gamma = 1.0 + intensity * (torch.rand(1).item() - 0.5)
        self.display_params.contrast = 1.0 + intensity * (torch.rand(1).item() - 0.5)
        self.display_params.brightness = intensity * (torch.rand(1).item() - 0.5) * 0.5
        self.display_params.pixelation = intensity * torch.rand(1).item()

        # Нормализация
        self.camera_params.scale = np.clip(self.camera_params.scale, 0.1, 3.0)
        self.camera_params.distortion = np.clip(self.camera_params.distortion, 0, 1)
        self.display_params.gamma = np.clip(self.display_params.gamma, 0.1, 3.0)
        self.display_params.contrast = np.clip(self.display_params.contrast, 0.1, 3.0)
        self.display_params.brightness = np.clip(self.display_params.brightness, -1, 1)
        self.display_params.pixelation = np.clip(self.display_params.pixelation, 0, 1)

        self._update_history()

    def get_current_params(self) -> Dict[str, Any]:
        """Получение текущих параметров."""
        return {
            'camera': self.camera_params.to_dict(),
            'display': self.display_params.to_dict(),
            'mode': self.control_mode.value,
            'script': self.current_script,
            'script_step': self.script_step
        }

    def _update_history(self) -> None:
        """Обновление истории параметров."""
        self.param_history.append({
            'timestamp': time.time(),
            'camera': self.camera_params.to_dict(),
            'display': self.display_params.to_dict()
        })

        if len(self.param_history) > self.max_history:
            self.param_history.pop(0)

    def get_param_history(self) -> list:
        """Получение истории параметров."""
        return self.param_history.copy()

    def save_preset(self, name: str) -> None:
        """Сохранение текущих параметров как пресет."""
        if not hasattr(self, 'presets'):
            self.presets = {}

        self.presets[name] = self.get_current_params()

    def load_preset(self, name: str) -> bool:
        """Загрузка пресета."""
        if not hasattr(self, 'presets') or name not in self.presets:
            return False

        preset = self.presets[name]
        self.camera_params = CameraParams.from_dict(preset['camera'])
        self.display_params = DisplayParams.from_dict(preset['display'])
        self.control_mode = ControlMode(preset.get('mode', 'manual'))

        return True

    def reset(self) -> None:
        """Сброс контроллера."""
        self.camera_params = CameraParams()
        self.display_params = DisplayParams()
        self.param_history.clear()
        self.control_mode = ControlMode.MANUAL
        self.current_script = None
        self.script_step = 0

    # Вспомогательные методы для создания типичных сценариев

    def create_exploration_script(self, name: str = "exploration") -> None:
        """Создание сценария исследования (плавное изменение параметров)."""
        script = []
        for i in range(20):
            t = i / 19.0  # 0 to 1
            script.append({
                'camera': {
                    'angle': 360 * t,
                    'scale': 1.0 + 0.5 * np.sin(t * 2 * np.pi),
                    'distortion': 0.3 * np.sin(t * 4 * np.pi)
                },
                'display': {
                    'gamma': 1.0 + 0.3 * np.sin(t * 3 * np.pi),
                    'contrast': 1.0 + 0.4 * np.cos(t * 2 * np.pi)
                },
                'delay': 0.5
            })

        self.create_script(name, script)

    def create_chaos_to_order_script(self, name: str = "chaos_to_order") -> None:
        """Создание сценария от хаоса к порядку."""
        script = []

        # Фаза хаоса
        for i in range(10):
            script.append({
                'camera': {
                    'angle': torch.rand(1).item() * 360,
                    'scale': 0.5 + torch.rand(1).item() * 2,
                    'distortion': torch.rand(1).item() * 0.8
                },
                'display': {
                    'gamma': 0.5 + torch.rand(1).item(),
                    'contrast': 1.0 + torch.rand(1).item(),
                    'brightness': (torch.rand(1).item() - 0.5) * 0.4
                },
                'delay': 0.3
            })

        # Переход к порядку
        for i in range(15):
            t = i / 14.0
            damping = 1.0 - t  # От 1 к 0

            script.append({
                'camera': {
                    'angle': damping * torch.rand(1).item() * 180,  # Уменьшение хаоса
                    'scale': 1.0 + damping * (torch.rand(1).item() - 0.5),
                    'distortion': damping * torch.rand(1).item() * 0.4
                },
                'display': {
                    'gamma': 1.0 + damping * (torch.rand(1).item() - 0.5) * 0.4,
                    'contrast': 1.0 + damping * (torch.rand(1).item() - 0.5) * 0.4,
                    'brightness': damping * (torch.rand(1).item() - 0.5) * 0.2
                },
                'delay': 0.4
            })

        self.create_script(name, script)