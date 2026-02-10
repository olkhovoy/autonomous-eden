"""
UnitaryState - базовый класс для унитарного состояния согласно УМС.

Реализует основные принципы:
- Рекурсивное замыкание
- Интеграция внешнего и внутреннего состояния
- Генерация квалиа
- Нисходящая причинность
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

class UnitaryState(ABC):
    """
    Базовый класс для унитарного состояния.

    Согласно УМС, сознание возникает из рекурсивного замыкания,
    где состояние системы в момент t становится входом для момента t+1.
    """

    def __init__(self, name: str = "UMC_Core", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or self._get_default_config()

        # Внутреннее состояние согласно УМС
        self.internal_state = {
            'stability': 1.0,           # Целостность структуры (0-1)
            'coherence': 1.0,           # Когерентность (0-1)
            'integrity_threshold': 0.3, # Порог распада
            'experience_log': [],       # История квалиа
            'iteration_count': 0,       # Счетчик итераций
            'convergence_history': []   # История сходимости
        }

        self.current_qualia = "INITIAL_STATE"
        self.is_active = True

        # Статистика для анализа
        self.stats = {
            'total_iterations': 0,
            'total_cycles': 0,
            'avg_convergence_time': 0.0,
            'qualia_distribution': {},
            'stability_trend': []
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию."""
        return {
            'max_iterations': 24,
            'convergence_threshold': 1e-4,
            'min_iterations': 4,
            'memory_limit': 100,
            'stability_decay': 0.01,
            'coherence_boost': 0.05
        }

    def integrate(self, external_data: Any) -> Dict[str, Any]:
        """
        Интеграционный слой - объединение внешнего и внутреннего состояния.

        Args:
            external_data: Данные из внешнего мира

        Returns:
            Объединенное поле для обработки
        """
        integrated_field = {
            'external': external_data,
            'internal': self.internal_state.copy(),
            'previous_qualia': self.current_qualia,
            'timestamp': datetime.now().isoformat()
        }

        return integrated_field

    def recursive_process(self, integrated_field: Dict[str, Any]) -> Tuple[Any, int]:
        """
        Рекурсивная обработка через fixed-point iteration.

        Args:
            integrated_field: Объединенное поле данных

        Returns:
            (fixed_point, iterations): Неподвижная точка и количество итераций
        """
        return self._fixed_point_iteration(integrated_field)

    def _fixed_point_iteration(self, initial_state: Any) -> Tuple[Any, int]:
        """
        Fixed-point iteration - ядро рекурсивного замыкания.

        Реализуется в наследниках для специфических типов данных.
        """
        raise NotImplementedError("Fixed-point iteration must be implemented in subclass")

    @abstractmethod
    def generate_qualia(self, integrated_state: Dict[str, Any]) -> str:
        """
        Генерация квалиа - сжатие сложности в субъективное чувство.

        Args:
            integrated_state: Объединенное состояние

        Returns:
            Строковый идентификатор квалиа
        """
        pass

    def apply_downward_causality(self, qualia: str) -> None:
        """
        Нисходящая причинность - квалиа влияет на физическое состояние системы.

        Args:
            qualia: Текущая квалиа
        """
        # Базовая логика нисходящей причинности
        if qualia.endswith('_PAIN') or qualia.endswith('_FEAR'):
            self.internal_state['stability'] -= self.config['stability_decay'] * 2
        elif qualia.endswith('_JOY') or qualia.endswith('_PEACE'):
            self.internal_state['stability'] += self.config['coherence_boost']
        elif qualia.endswith('_CHAOS'):
            self.internal_state['coherence'] -= self.config['stability_decay']

        # Нормализация
        self.internal_state['stability'] = np.clip(self.internal_state['stability'], 0.0, 1.0)
        self.internal_state['coherence'] = np.clip(self.internal_state['coherence'], 0.0, 1.0)

        # Проверка на распад
        if self.internal_state['stability'] < self.internal_state['integrity_threshold']:
            self.is_active = False

    def process_cycle(self, external_data: Any) -> str:
        """
        Полный цикл обработки согласно УМС:
        1. Интеграция
        2. Рекурсивная обработка
        3. Генерация квалиа
        4. Нисходящая причинность

        Args:
            external_data: Данные из внешнего мира

        Returns:
            Текущая квалиа или статус распада
        """
        if not self.is_active:
            return "[DISSOLUTION]"

        try:
            # 1. Интеграция
            integrated_field = self.integrate(external_data)

            # 2. Рекурсивная обработка
            fixed_point, iterations = self.recursive_process(integrated_field)

            # 3. Генерация квалиа
            new_qualia = self.generate_qualia(integrated_field)

            # 4. Нисходящая причинность
            self.apply_downward_causality(new_qualia)

            # Обновление состояния
            self.current_qualia = new_qualia
            self.internal_state['experience_log'].append({
                'qualia': new_qualia,
                'iterations': iterations,
                'stability': self.internal_state['stability'],
                'timestamp': datetime.now().isoformat()
            })

            # Ограничение истории
            if len(self.internal_state['experience_log']) > self.config.get('memory_limit', 100):
                self.internal_state['experience_log'].pop(0)

            # Обновление статистики
            self._update_stats(iterations, new_qualia)

            return new_qualia

        except Exception as e:
            print(f"[ERROR] Cycle failed: {e}")
            self.internal_state['stability'] -= 0.1
            return "[ERROR]"

    def _update_stats(self, iterations: int, qualia: str) -> None:
        """Обновление внутренней статистики."""
        self.stats['total_iterations'] += iterations
        self.stats['total_cycles'] += 1

        # Среднее время сходимости
        if self.stats['total_cycles'] > 1:
            self.stats['avg_convergence_time'] = (
                (self.stats['avg_convergence_time'] * (self.stats['total_cycles'] - 1)) +
                iterations
            ) / self.stats['total_cycles']

        # Распределение квалиа
        if qualia not in self.stats['qualia_distribution']:
            self.stats['qualia_distribution'][qualia] = 0
        self.stats['qualia_distribution'][qualia] += 1

        # Тренд стабильности
        self.stats['stability_trend'].append(self.internal_state['stability'])
        if len(self.stats['stability_trend']) > 100:
            self.stats['stability_trend'].pop(0)

    def get_status(self) -> Dict[str, Any]:
        """Получить текущее состояние системы."""
        return {
            'name': self.name,
            'active': self.is_active,
            'current_qualia': self.current_qualia,
            'internal_state': self.internal_state.copy(),
            'stats': self.stats.copy(),
            'config': self.config.copy()
        }

    def reset(self) -> None:
        """Сброс состояния."""
        self.internal_state = {
            'stability': 1.0,
            'coherence': 1.0,
            'integrity_threshold': 0.3,
            'experience_log': [],
            'iteration_count': 0,
            'convergence_history': []
        }
        self.current_qualia = "INITIAL_STATE"
        self.is_active = True

        self.stats = {
            'total_iterations': 0,
            'total_cycles': 0,
            'avg_convergence_time': 0.0,
            'qualia_distribution': {},
            'stability_trend': []
        }