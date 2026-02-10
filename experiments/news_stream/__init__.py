"""
News Stream Experiment

Эксперимент с рекурсивным чтением новостей согласно УМС.
Система адаптирует поток новостей на основе внутреннего состояния и квалиа.
"""

from .unitary_reader import UnitaryNewsReader
from .source_manager import NewsSourceManager
from .mood_filter import MoodBasedFilter

__all__ = ['UnitaryNewsReader', 'NewsSourceManager', 'MoodBasedFilter']