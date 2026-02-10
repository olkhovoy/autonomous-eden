"""
QualiaBase - базовый генератор квалиа для экспериментов УМС.

Квалиа - это субъективные чувства, возникающие из сжатия сложности
в дискретные семантические токены согласно теории УМС.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum

class QualiaType(Enum):
    """Базовые типы квалиа."""
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CHAOTIC = "chaotic"
    ORDERLY = "orderly"

class QualiaEngine(ABC):
    """
    Базовый генератор квалиа.

    Квалиа возникают из взаимодействия внешнего мира и внутреннего состояния системы.
    Они сжимают сложность в субъективные чувства и влияют на поведение через
    нисходящую причинность.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.qualia_history: List[str] = []
        self.qualia_stats: Dict[str, int] = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию."""
        return {
            'entropy_threshold_high': 0.7,
            'entropy_threshold_low': 0.3,
            'stability_threshold_high': 0.8,
            'stability_threshold_low': 0.4,
            'coherence_threshold': 0.6,
            'memory_limit': 50
        }

    @abstractmethod
    def generate(self, integrated_state: Dict[str, Any]) -> str:
        """
        Генерация квалиа из объединенного состояния.

        Args:
            integrated_state: {
                'external': данные из внешнего мира,
                'internal': внутреннее состояние системы,
                'previous_qualia': предыдущая квалиа,
                'timestamp': время
            }

        Returns:
            Строковый идентификатор квалиа
        """
        pass

    def _extract_metrics(self, integrated_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Извлечение базовых метрик из объединенного состояния.

        Returns:
            {
                'entropy': энтропия внешнего мира,
                'stability': стабильность системы,
                'coherence': когерентность системы,
                'convergence_speed': скорость сходимости,
                'complexity': сложность состояния
            }
        """
        external = integrated_state.get('external', {})
        internal = integrated_state.get('internal', {})

        # Энтропия внешнего мира (специфично для каждого эксперимента)
        entropy = self._calculate_entropy(external)

        # Стабильность и когерентность из внутреннего состояния
        stability = internal.get('stability', 0.5)
        coherence = internal.get('coherence', 0.5)

        # Скорость сходимости из истории
        convergence_history = internal.get('convergence_history', [])
        convergence_speed = self._calculate_convergence_speed(convergence_history)

        # Сложность состояния
        complexity = self._calculate_complexity(integrated_state)

        return {
            'entropy': entropy,
            'stability': stability,
            'coherence': coherence,
            'convergence_speed': convergence_speed,
            'complexity': complexity
        }

    def _calculate_entropy(self, external_data: Any) -> float:
        """
        Расчет энтропии внешних данных.

        Переопределяется в наследниках для специфических типов данных.
        """
        if isinstance(external_data, torch.Tensor):
            # Для тензоров - стандартная энтропия
            if external_data.numel() == 0:
                return 0.5

            # Нормализация
            data = external_data.flatten().float()
            if data.max() > 1.0:
                data = data / data.max()

            # Энтропия распределения
            hist = torch.histc(data, bins=10, min=0, max=1)
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # Избегать log(0)
            entropy = -torch.sum(hist * torch.log(hist))
            return entropy.item() / math.log(10)  # Нормализация

        elif isinstance(external_data, (list, dict)):
            # Для структурных данных - оценка сложности
            if isinstance(external_data, dict):
                size = len(external_data)
            else:
                size = len(external_data)

            # Простая эвристика: больше данных = выше энтропия
            return min(1.0, size / 100.0)

        else:
            # По умолчанию - средняя энтропия
            return 0.5

    def _calculate_convergence_speed(self, convergence_history: List[float]) -> float:
        """Расчет скорости сходимости из истории."""
        if len(convergence_history) < 2:
            return 0.5

        # Средняя скорость изменения
        diffs = np.diff(convergence_history)
        avg_speed = np.mean(np.abs(diffs))

        # Нормализация: быстрая сходимость = низкая скорость изменения
        return 1.0 - min(1.0, avg_speed * 10)

    def _calculate_complexity(self, integrated_state: Dict[str, Any]) -> float:
        """Расчет сложности объединенного состояния."""
        # Комбинация энтропии, стабильности и когерентности
        metrics = self._extract_metrics(integrated_state)

        # Сложность растет с энтропией и падает со стабильностью/когерентностью
        complexity = (
            0.4 * metrics['entropy'] +
            0.3 * (1.0 - metrics['stability']) +
            0.3 * (1.0 - metrics['coherence'])
        )

        return min(1.0, max(0.0, complexity))

    def _get_qualia_type(self, metrics: Dict[str, float]) -> QualiaType:
        """
        Определение типа квалиа на основе метрик.

        Returns:
            QualiaType: базовый тип квалиа
        """
        entropy = metrics['entropy']
        stability = metrics['stability']
        coherence = metrics['coherence']

        # Хаотичное состояние
        if entropy > self.config['entropy_threshold_high'] and stability < self.config['stability_threshold_low']:
            return QualiaType.CHAOTIC

        # Упорядоченное состояние
        elif entropy < self.config['entropy_threshold_low'] and stability > self.config['stability_threshold_high']:
            return QualiaType.ORDERLY

        # Положительное состояние
        elif stability > self.config['stability_threshold_high'] and coherence > self.config['coherence_threshold']:
            return QualiaType.POSITIVE

        # Отрицательное состояние
        elif stability < self.config['stability_threshold_low'] or coherence < self.config['coherence_threshold']:
            return QualiaType.NEGATIVE

        # Нейтральное состояние
        else:
            return QualiaType.NEUTRAL

    def update_history(self, qualia: str) -> None:
        """Обновление истории квалиа."""
        self.qualia_history.append(qualia)

        # Ограничение истории
        if len(self.qualia_history) > self.config.get('memory_limit', 50):
            self.qualia_history.pop(0)

        # Статистика
        if qualia not in self.qualia_stats:
            self.qualia_stats[qualia] = 0
        self.qualia_stats[qualia] += 1

    def get_qualia_stats(self) -> Dict[str, Any]:
        """Получить статистику квалиа."""
        return {
            'history': self.qualia_history.copy(),
            'distribution': self.qualia_stats.copy(),
            'most_frequent': max(self.qualia_stats, key=self.qualia_stats.get) if self.qualia_stats else None,
            'total_count': len(self.qualia_history)
        }

    def reset(self) -> None:
        """Сброс состояния генератора квалиа."""
        self.qualia_history.clear()
        self.qualia_stats.clear()

class SimpleQualiaEngine(QualiaEngine):
    """
    Простой генератор квалиа на основе порогов.

    Используется для базового тестирования.
    """

    def generate(self, integrated_state: Dict[str, Any]) -> str:
        """Простая генерация квалиа на основе порогов."""
        metrics = self._extract_metrics(integrated_state)

        qualia_type = self._get_qualia_type(metrics)

        # Преобразование типа в конкретную квалиа
        qualia_map = {
            QualiaType.CHAOTIC: "CHAOS_DISORDER",
            QualiaType.ORDERLY: "ORDER_HARMONY",
            QualiaType.POSITIVE: "CONTENTMENT",
            QualiaType.NEGATIVE: "UNEASE",
            QualiaType.NEUTRAL: "EQUILIBRIUM"
        }

        qualia = qualia_map[qualia_type]

        # Обновление истории
        self.update_history(qualia)

        return qualia