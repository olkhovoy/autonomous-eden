"""
Core components for UMC experiments.

Базовые компоненты для всех экспериментов УМС:
- UnitaryState: базовый класс унитарного состояния
- FixedPoint: утилиты для рекурсивной итерации
- QualiaBase: базовый генератор квалиа
"""

from .unitary_state import UnitaryState
from .fixed_point import FixedPointIteration
from .qualia_base import QualiaEngine

__all__ = ['UnitaryState', 'FixedPointIteration', 'QualiaEngine']