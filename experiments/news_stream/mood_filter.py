"""
Mood-Based Filter - фильтр новостей на основе настроения (квалиа).

Реализует адаптивную фильтрацию новостей согласно УМС:
система выбирает источники и фильтрует новости на основе текущего
эмоционального состояния, создавая петлю обратной связи.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

class MoodBasedFilter:
    """
    Фильтр новостей на основе настроения согласно УМС.

    Принцип работы:
    1. Анализ текущей квалиа системы
    2. Выбор подходящих источников (мягкая/жесткая/позитивная информация)
    3. Фильтрация индивидуальных новостей
    4. Адаптация стратегии на основе эффективности

    Это создает "эмоциональный иммунитет" - система учится выбирать
    информацию, которая поддерживает ее стабильность.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # Профили настроений для квалиа
        self.mood_profiles = self._initialize_mood_profiles()

        # Статистика эффективности фильтрации
        self.filter_stats = {
            'total_filtered': 0,
            'qualia_distribution': defaultdict(int),
            'source_preferences': defaultdict(lambda: defaultdict(float)),
            'filtering_accuracy': 0.0,
            'adaptation_cycles': 0
        }

        # Адаптивные веса
        self.adaptive_weights = {
            'stability_threshold': 0.6,  # Порог для жесткой фильтрации
            'diversity_factor': 0.3,     # Вес разнообразия источников
            'recency_factor': 0.2,       # Вес свежести новостей
            'sentiment_alignment': 0.8   # Вес соответствия настроению
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию."""
        return {
            'filtering_enabled': True,
            'adaptation_enabled': True,
            'max_sources_per_qualia': 3,
            'filtering_threshold': 0.5,
            'adaptation_rate': 0.1,
            'memory_window': 50  # Размер окна для анализа трендов
        }

    def _initialize_mood_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Инициализация профилей настроений для разных квалиа.

        Каждый профиль определяет:
        - Предпочитаемые категории источников
        - Уровень фильтрации
        - Тип желаемой информации
        """
        return {
            'HOPE': {
                'preferred_categories': ['world', 'technology', 'science'],
                'avoid_categories': ['politics_negative'],
                'filtering_level': 'soft',  # Мягкая фильтрация
                'sentiment_preference': 'positive',  # Предпочтение позитивным новостям
                'diversity_bonus': 0.2,
                'description': 'Открытость к позитивным изменениям'
            },

            'DREAD': {
                'preferred_categories': ['security', 'defense', 'local'],
                'avoid_categories': ['conflict', 'crisis'],
                'filtering_level': 'hard',  # Жесткая фильтрация
                'sentiment_preference': 'neutral',  # Нейтральная информация
                'diversity_bonus': -0.1,  # Меньше разнообразия при страхе
                'description': 'Фокус на безопасности и стабильности'
            },

            'CURIOSITY': {
                'preferred_categories': ['technology', 'science', 'culture'],
                'avoid_categories': [],  # Открытость ко всему
                'filtering_level': 'minimal',  # Минимальная фильтрация
                'sentiment_preference': 'any',  # Любая эмоциональная окраска
                'diversity_bonus': 0.4,
                'description': 'Исследовательское настроение'
            },

            'SATURATION': {
                'preferred_categories': ['local', 'entertainment'],
                'avoid_categories': ['world', 'politics'],
                'filtering_level': 'selective',  # Выборочная фильтрация
                'sentiment_preference': 'light',  # Легкие новости
                'diversity_bonus': 0.1,
                'description': 'Насыщенность - поиск развлечений'
            },

            'DREAD_INTENSE': {
                'preferred_categories': ['local', 'sports'],
                'avoid_categories': ['world', 'politics', 'conflict', 'crisis', 'economy'],
                'filtering_level': 'maximum',  # Максимальная фильтрация
                'sentiment_preference': 'positive_only',  # Только позитив
                'diversity_bonus': -0.3,
                'description': 'Интенсивный страх - изоляция от угроз'
            },

            'HOPE_RELIEF': {
                'preferred_categories': ['world', 'economy', 'technology'],
                'avoid_categories': ['conflict'],
                'filtering_level': 'moderate',  # Умеренная фильтрация
                'sentiment_preference': 'positive',  # Позитив для поддержания надежды
                'diversity_bonus': 0.3,
                'description': 'Надежда после трудностей'
            },

            'DREAD_MANAGED': {
                'preferred_categories': ['world', 'politics', 'defense'],
                'avoid_categories': ['sensationalism'],
                'filtering_level': 'moderate',  # Управляемая фильтрация
                'sentiment_preference': 'factual',  # Факты без эмоций
                'diversity_bonus': 0.0,
                'description': 'Управляемый страх - анализ угроз'
            },

            'NEUTRAL_NEWS': {
                'preferred_categories': ['general', 'local'],
                'avoid_categories': ['sensationalism', 'celebrity'],
                'filtering_level': 'minimal',  # Минимальная фильтрация
                'sentiment_preference': 'neutral',  # Нейтральные новости
                'diversity_bonus': 0.1,
                'description': 'Нейтральное состояние'
            }
        }

    def filter_sources(self, available_sources: Dict[str, Dict[str, Any]],
                      current_qualia: str, system_stability: float) -> List[str]:
        """
        Фильтрация источников на основе текущего настроения.

        Args:
            available_sources: Доступные источники
            current_qualia: Текущая квалиа
            system_stability: Стабильность системы (0-1)

        Returns:
            Список рекомендованных источников
        """
        if not self.config['filtering_enabled']:
            return list(available_sources.keys())

        # Получение профиля настроения
        mood_profile = self.mood_profiles.get(current_qualia, self.mood_profiles['NEUTRAL_NEWS'])

        # Расчет весов для источников
        source_weights = {}

        for source_id, source_info in available_sources.items():
            if not source_info.get('active', True):
                continue

            weight = self._calculate_source_weight(source_id, source_info, mood_profile,
                                                 system_stability)
            source_weights[source_id] = weight

        # Сортировка по весу
        sorted_sources = sorted(source_weights.items(), key=lambda x: x[1], reverse=True)

        # Выбор топ источников
        max_sources = self.config['max_sources_per_qualia']
        selected_sources = [source_id for source_id, _ in sorted_sources[:max_sources]]

        # Применение адаптивных модификаций
        if self.config['adaptation_enabled']:
            selected_sources = self._apply_adaptive_modifications(
                selected_sources, current_qualia, system_stability
            )

        return selected_sources

    def _calculate_source_weight(self, source_id: str, source_info: Dict[str, Any],
                               mood_profile: Dict[str, Any], system_stability: float) -> float:
        """
        Расчет веса источника для данного настроения.

        Args:
            source_id: ID источника
            source_info: Информация об источнике
            mood_profile: Профиль настроения
            system_stability: Стабильность системы

        Returns:
            Вес источника (0-1)
        """
        base_weight = 0.5  # Базовый вес

        # Категория источника
        source_category = source_info.get('category', 'general')

        # Бонус за предпочтительные категории
        if source_category in mood_profile['preferred_categories']:
            base_weight += 0.3

        # Штраф за нежелательные категории
        if source_category in mood_profile['avoid_categories']:
            base_weight -= 0.4

        # Приоритет источника
        priority = source_info.get('priority', 5) / 10.0  # Нормализация 0-1
        base_weight += priority * 0.2

        # Историческая эффективность для данного настроения
        historical_weight = self.filter_stats['source_preferences'][mood_profile['description']].get(source_id, 0.5)
        base_weight = base_weight * 0.7 + historical_weight * 0.3

        # Модификация на основе стабильности системы
        if system_stability < 0.4 and mood_profile['filtering_level'] in ['hard', 'maximum']:
            # При низкой стабильности предпочитаем надежные источники
            reliability = source_info.get('priority', 5) / 10.0
            base_weight += reliability * 0.2

        # Ограничение диапазона
        return np.clip(base_weight, 0.0, 1.0)

    def _apply_adaptive_modifications(self, selected_sources: List[str],
                                    current_qualia: str, system_stability: float) -> List[str]:
        """
        Применение адаптивных модификаций к выбору источников.

        Args:
            selected_sources: Выбранные источники
            current_qualia: Текущая квалиа
            system_stability: Стабильность системы

        Returns:
            Модифицированный список источников
        """
        # Анализ недавней эффективности
        recent_qualia = list(self.filter_stats['qualia_distribution'].keys())[-10:]

        # Если часто повторяется одно и то же настроение - добавить разнообразие
        if len(set(recent_qualia)) <= 2 and len(recent_qualia) >= 5:
            diversity_bonus = self.adaptive_weights['diversity_factor']
            # Можно добавить случайный источник для разнообразия
            pass

        # При очень низкой стабильности - консервативный подход
        if system_stability < 0.3:
            # Оставить только самые надежные источники
            selected_sources = selected_sources[:2]

        # При высокой стабильности - эксперименты с новыми источниками
        elif system_stability > 0.8:
            # Можно добавить источник с низким весом для тестирования
            pass

        return selected_sources

    def filter_news(self, news_batch: List[Dict[str, Any]], current_qualia: str,
                   system_stability: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Фильтрация индивидуальных новостей.

        Args:
            news_batch: Партия новостей
            current_qualia: Текущая квалиа
            system_stability: Стабильность системы

        Returns:
            (accepted_news, rejected_news): Принятые и отклоненные новости
        """
        if not self.config['filtering_enabled']:
            return news_batch, []

        accepted_news = []
        rejected_news = []

        mood_profile = self.mood_profiles.get(current_qualia, self.mood_profiles['NEUTRAL_NEWS'])

        for news in news_batch:
            if self._should_accept_news(news, mood_profile, system_stability):
                accepted_news.append(news)
            else:
                rejected_news.append(news)

        # Обновление статистики
        self.filter_stats['total_filtered'] += len(news_batch)
        self.filter_stats['qualia_distribution'][current_qualia] += len(accepted_news)

        return accepted_news, rejected_news

    def _should_accept_news(self, news: Dict[str, Any], mood_profile: Dict[str, Any],
                          system_stability: float) -> bool:
        """
        Определение, следует ли принять новость.

        Args:
            news: Новость
            mood_profile: Профиль настроения
            system_stability: Стабильность системы

        Returns:
            True если новость принимается
        """
        metadata = news.get('metadata', {})
        text = news.get('text', '')

        # Быстрые проверки
        if mood_profile['filtering_level'] == 'minimal':
            return True

        # Анализ эмоциональной окраски
        sentiment_score = self._analyze_news_sentiment(text)
        sentiment_preference = mood_profile['sentiment_preference']

        # Проверка соответствия настроению
        if not self._check_sentiment_alignment(sentiment_score, sentiment_preference):
            return False

        # Проверка категории
        category = metadata.get('category', 'general')
        if category in mood_profile['avoid_categories']:
            return False

        # При низкой стабильности - дополнительная фильтрация
        if system_stability < self.adaptive_weights['stability_threshold']:
            if sentiment_score < -0.3:  # Слишком негативно
                return False

            # Проверка на "триггерные" слова
            if self._contains_trigger_words(text):
                return False

        # Применение порога фильтрации
        acceptance_probability = self._calculate_acceptance_probability(
            news, mood_profile, system_stability
        )

        return torch.rand(1).item() < acceptance_probability

    def _analyze_news_sentiment(self, text: str) -> float:
        """
        Анализ эмоциональной окраски новости.

        Returns:
            Сентимент (-1.0 до 1.0)
        """
        text_lower = text.lower()

        positive_words = [
            'success', 'growth', 'achievement', 'positive', 'good', 'great',
            'excellent', 'amazing', 'wonderful', 'happy', 'joy', 'hope',
            'progress', 'breakthrough', 'win', 'victory', 'peace', 'agreement'
        ]

        negative_words = [
            'failure', 'decline', 'crisis', 'negative', 'bad', 'terrible',
            'awful', 'horrible', 'sad', 'fear', 'anger', 'disaster',
            'conflict', 'war', 'death', 'loss', 'crisis', 'threat',
            'attack', 'violence', 'destruction'
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total_indicators = positive_count + negative_count

        if total_indicators == 0:
            return 0.0

        return (positive_count - negative_count) / total_indicators

    def _check_sentiment_alignment(self, sentiment_score: float, preference: str) -> bool:
        """
        Проверка соответствия сентимента предпочтениям.

        Args:
            sentiment_score: Сентимент новости (-1 до 1)
            preference: Предпочтение ('positive', 'negative', 'neutral', etc.)

        Returns:
            True если соответствует
        """
        if preference == 'any':
            return True
        elif preference == 'positive' and sentiment_score > 0.1:
            return True
        elif preference == 'positive_only' and sentiment_score >= 0:
            return True
        elif preference == 'negative' and sentiment_score < -0.1:
            return True
        elif preference == 'neutral' and abs(sentiment_score) < 0.3:
            return True
        elif preference == 'light' and abs(sentiment_score) < 0.5:
            return True
        elif preference == 'factual' and abs(sentiment_score) < 0.2:
            return True

        return False

    def _contains_trigger_words(self, text: str) -> bool:
        """
        Проверка на наличие триггерных слов.

        Returns:
            True если содержит триггерные слова
        """
        trigger_words = [
            'panic', 'terror', 'catastrophe', 'apocalypse', 'doomsday',
            'crisis', 'emergency', 'disaster', 'tragedy', 'nightmare'
        ]

        text_lower = text.lower()
        return any(word in text_lower for word in trigger_words)

    def _calculate_acceptance_probability(self, news: Dict[str, Any],
                                       mood_profile: Dict[str, Any],
                                       system_stability: float) -> float:
        """
        Расчет вероятности принятия новости.

        Args:
            news: Новость
            mood_profile: Профиль настроения
            system_stability: Стабильность системы

        Returns:
            Вероятность принятия (0-1)
        """
        base_probability = 0.7  # Базовая вероятность

        # Модификатор уровня фильтрации
        filtering_multipliers = {
            'minimal': 0.9,
            'soft': 0.8,
            'moderate': 0.6,
            'hard': 0.4,
            'maximum': 0.2,
            'selective': 0.5
        }

        filter_level = mood_profile['filtering_level']
        base_probability *= filtering_multipliers.get(filter_level, 0.7)

        # Модификатор стабильности
        if system_stability < 0.4:
            base_probability *= 0.7  # Более строгая фильтрация
        elif system_stability > 0.8:
            base_probability *= 1.2  # Более открытая фильтрация

        # Модификатор свежести новости
        metadata = news.get('metadata', {})
        published_str = metadata.get('published', '')
        if published_str:
            try:
                published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                age_hours = (datetime.now() - published).total_seconds() / 3600

                # Штраф за старые новости
                if age_hours > 24:
                    age_penalty = min(age_hours / 24 * 0.1, 0.5)
                    base_probability *= (1 - age_penalty)
            except:
                pass  # Игнорируем ошибки парсинга даты

        return np.clip(base_probability, 0.0, 1.0)

    def adapt_filtering_strategy(self, feedback_qualia: str, system_stability: float) -> None:
        """
        Адаптация стратегии фильтрации на основе обратной связи.

        Args:
            feedback_qualia: Квалиа после обработки новостей
            system_stability: Текущая стабильность системы
        """
        if not self.config['adaptation_enabled']:
            return

        # Анализ эффективности
        recent_qualia = list(self.filter_stats['qualia_distribution'].keys())[-10:]

        # Если система часто в негативном состоянии - усилить фильтрацию
        negative_qualia = ['DREAD', 'FEAR', 'PAIN', 'DREAD_INTENSE']
        negative_count = sum(1 for q in recent_qualia if any(nq in q for nq in negative_qualia))

        if negative_count > 5:
            # Увеличить порог стабильности для жесткой фильтрации
            self.adaptive_weights['stability_threshold'] = min(
                self.adaptive_weights['stability_threshold'] + self.config['adaptation_rate'], 0.8
            )

        # Если система стабильна - можно ослабить фильтрацию
        elif system_stability > 0.7 and len(recent_qualia) >= 5:
            self.adaptive_weights['stability_threshold'] = max(
                self.adaptive_weights['stability_threshold'] - self.config['adaptation_rate'] * 0.5, 0.3
            )

        # Обновление счетчика адаптаций
        self.filter_stats['adaptation_cycles'] += 1

    def update_source_preferences(self, source_id: str, qualia: str,
                                effectiveness_score: float) -> None:
        """
        Обновление предпочтений для источника.

        Args:
            source_id: ID источника
            qualia: Полученная квалиа
            effectiveness_score: Эффективность (0-1)
        """
        current_pref = self.filter_stats['source_preferences'][qualia].get(source_id, 0.5)

        # Экспоненциальное сглаживание
        alpha = self.config['adaptation_rate']
        new_pref = current_pref * (1 - alpha) + effectiveness_score * alpha

        self.filter_stats['source_preferences'][qualia][source_id] = new_pref

    def get_filtering_stats(self) -> Dict[str, Any]:
        """Получение статистики фильтрации."""
        stats = self.filter_stats.copy()
        stats['adaptive_weights'] = self.adaptive_weights.copy()
        stats['mood_profiles_count'] = len(self.mood_profiles)

        # Расчет эффективности
        total_qualia = sum(stats['qualia_distribution'].values())
        if total_qualia > 0:
            positive_qualia = sum(count for qualia, count in stats['qualia_distribution'].items()
                                if any(pos in qualia for pos in ['HOPE', 'JOY', 'PEACE', 'CURIOSITY']))
            stats['positive_qualia_ratio'] = positive_qualia / total_qualia

        return stats

    def reset_adaptation(self) -> None:
        """Сброс адаптивных параметров."""
        self.adaptive_weights = {
            'stability_threshold': 0.6,
            'diversity_factor': 0.3,
            'recency_factor': 0.2,
            'sentiment_alignment': 0.8
        }

        # Сброс предпочтений источников
        self.filter_stats['source_preferences'] = defaultdict(lambda: defaultdict(float))
        self.filter_stats['adaptation_cycles'] = 0