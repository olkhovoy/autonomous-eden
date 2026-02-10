"""
News Stream Experiment App - Streamlit интерфейс.

Интерактивное веб-приложение для эксперимента с новостным потоком УМС.
Отображает поток новостей, управляет источниками и показывает
адаптацию системы на основе квалиа.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import asyncio
import threading
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px

from .unitary_reader import UnitaryNewsReader
from .source_manager import NewsSourceManager
from .mood_filter import MoodBasedFilter

class NewsStreamApp:
    """
    Streamlit приложение для новостного эксперимента УМС.

    Функциональность:
    - Отображение потока новостей в реальном времени
    - Управление источниками (активация/деактивация)
    - Графики стабильности и квалиа
    - Статистика фильтрации
    - Ручное добавление тестовых новостей
    """

    def __init__(self):
        self.reader = None
        self.source_manager = None
        self.mood_filter = None

        # Состояние приложения
        self.is_running = False
        self.news_queue = []
        self.stats_history = []
        self.max_history_points = 100

        # Параметры обновления
        self.update_interval = 5  # секунды
        self.batch_size = 3

    def initialize_systems(self):
        """Инициализация компонентов системы."""
        if self.reader is None:
            self.reader = UnitaryNewsReader()
        if self.source_manager is None:
            self.source_manager = NewsSourceManager()
        if self.mood_filter is None:
            self.mood_filter = MoodBasedFilter()

    def run_news_cycle(self):
        """Выполнение цикла обработки новостей."""
        if not self.is_running:
            return

        try:
            # Получение текущего состояния системы
            current_qualia = self.reader.current_qualia
            system_stability = self.reader.internal_state.get('stability', 0.5)

            # Фильтрация источников на основе настроения
            available_sources = self.source_manager.sources
            selected_sources = self.mood_filter.filter_sources(
                available_sources, current_qualia, system_stability
            )

            # Получение новостей из выбранных источников
            news_batch = []
            for source_id in selected_sources:
                try:
                    news = self.source_manager.fetch_news_from_source(source_id, force=False)
                    news_batch.extend(news[:2])  # Максимум 2 новости от источника
                except Exception as e:
                    st.error(f"Ошибка загрузки из {source_id}: {e}")

            if not news_batch:
                # Если нет свежих новостей, берем из кэша
                news_batch = self.source_manager.get_next_news_batch(
                    batch_size=self.batch_size
                )

            # Фильтрация новостей через mood filter
            accepted_news, rejected_news = self.mood_filter.filter_news(
                news_batch, current_qualia, system_stability
            )

            # Обработка принятых новостей
            for news in accepted_news:
                qualia = self.reader.process_news(news['text'], news['metadata'])

                # Добавление в очередь отображения
                news_entry = {
                    'title': news['title'],
                    'text': news['text'][:200] + '...' if len(news['text']) > 200 else news['text'],
                    'source': news['metadata']['source_name'],
                    'qualia': qualia,
                    'timestamp': datetime.now(),
                    'sentiment': self.analyze_sentiment_quick(news['text'])
                }
                self.news_queue.append(news_entry)

                # Ограничение очереди
                if len(self.news_queue) > 20:
                    self.news_queue.pop(0)

            # Адаптация стратегии фильтрации
            if accepted_news:
                self.mood_filter.adapt_filtering_strategy(current_qualia, system_stability)

            # Сбор статистики
            self._collect_stats()

        except Exception as e:
            st.error(f"Ошибка в цикле новостей: {e}")

    def _collect_stats(self):
        """Сбор статистики для графиков."""
        stats_entry = {
            'timestamp': datetime.now(),
            'qualia': self.reader.current_qualia,
            'stability': self.reader.internal_state.get('stability', 0.0),
            'coherence': self.reader.internal_state.get('coherence', 0.0),
            'iterations': self.reader.internal_state.get('iteration_count', 0),
            'total_processed': self.reader.news_stats['total_processed'],
            'filtering_stats': self.mood_filter.get_filtering_stats()
        }

        self.stats_history.append(stats_entry)

        # Ограничение истории
        if len(self.stats_history) > self.max_history_points:
            self.stats_history.pop(0)

    def analyze_sentiment_quick(self, text: str) -> float:
        """Быстрый анализ сентимента для отображения."""
        return self.reader._analyze_sentiment(text)

    def create_streamlit_app(self):
        """Создание Streamlit интерфейса."""
        st.set_page_config(
            page_title="News Stream Experiment",
            page_icon="📰",
            layout="wide"
        )

        st.title("📰 News Stream Experiment")
        st.markdown("*Рекурсивное чтение новостей согласно Унитарной Модели Сознания*")

        # Инициализация систем
        self.initialize_systems()

        # Боковая панель управления
        self._create_sidebar()

        # Основная область
        col1, col2 = st.columns([2, 1])

        with col1:
            self._create_main_area()

        with col2:
            self._create_stats_area()

        # Панель отладки (свернутая)
        with st.expander("🔧 Отладка и статистика"):
            self._create_debug_area()

    def _create_sidebar(self):
        """Создание боковой панели управления."""
        st.sidebar.header("🎛️ Управление")

        # Кнопки запуска/остановки
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("▶️ Запустить", type="primary"):
                self.is_running = True
                st.rerun()
        with col2:
            if st.button("⏹️ Остановить"):
                self.is_running = False
                st.rerun()

        st.sidebar.markdown("---")

        # Настройки обновления
        st.sidebar.subheader("⚙️ Параметры")
        self.update_interval = st.sidebar.slider(
            "Интервал обновления (сек)",
            min_value=1, max_value=30, value=self.update_interval
        )
        self.batch_size = st.sidebar.slider(
            "Размер партии новостей",
            min_value=1, max_value=10, value=self.batch_size
        )

        st.sidebar.markdown("---")

        # Управление источниками
        st.sidebar.subheader("📡 Источники")

        sources_info = self.source_manager.get_sources_info()
        active_count = len(sources_info['active_sources'])

        st.sidebar.metric("Активных источников", active_count)

        # Переключатели для источников
        for source_id, source_info in sources_info['sources'].items():
            is_active = source_id in sources_info['active_list']

            if st.sidebar.checkbox(
                f"{source_info['name']} ({source_info['category']})",
                value=is_active,
                key=f"source_{source_id}"
            ):
                if not is_active:
                    self.source_manager.activate_source(source_id, True)
            else:
                if is_active:
                    self.source_manager.activate_source(source_id, False)

        st.sidebar.markdown("---")

        # Тестовые новости
        st.sidebar.subheader("🧪 Тест")
        test_news = st.sidebar.text_area(
            "Тестовая новость",
            placeholder="Введите текст новости для тестирования...",
            height=100
        )

        if st.sidebar.button("Отправить тестовую новость") and test_news.strip():
            test_metadata = {
                'source': 'manual_test',
                'source_name': 'Тестовый ввод',
                'category': 'test'
            }

            qualia = self.reader.process_news(test_news, test_metadata)

            st.sidebar.success(f"Квалиа: {qualia}")

            # Добавление в очередь отображения
            news_entry = {
                'title': 'Тестовая новость',
                'text': test_news[:200] + '...' if len(test_news) > 200 else test_news,
                'source': 'Тестовый ввод',
                'qualia': qualia,
                'timestamp': datetime.now(),
                'sentiment': self.analyze_sentiment_quick(test_news)
            }
            self.news_queue.append(news_entry)

    def _create_main_area(self):
        """Создание основной области с потоком новостей."""
        st.subheader("📰 Поток новостей")

        # Статус системы
        status_col1, status_col2, status_col3 = st.columns(3)

        with status_col1:
            qualia = self.reader.current_qualia if self.reader else "Не инициализировано"
            st.metric("Текущая квалиа", qualia)

        with status_col2:
            stability = self.reader.internal_state.get('stability', 0.0) if self.reader else 0.0
            st.metric("Стабильность", f"{stability:.3f}")

        with status_col3:
            total_processed = self.reader.news_stats['total_processed'] if self.reader else 0
            st.metric("Обработано новостей", total_processed)

        # Поток новостей
        st.markdown("---")

        if not self.news_queue:
            st.info("Новости появятся после запуска эксперимента...")
        else:
            # Отображение последних новостей
            for news in reversed(self.news_queue[-10:]):  # Последние 10
                self._display_news_item(news)

    def _display_news_item(self, news: Dict[str, Any]):
        """Отображение элемента новости."""
        # Цвет рамки в зависимости от квалиа
        qualia_colors = {
            'HOPE': '🟢',
            'DREAD': '🔴',
            'CURIOSITY': '🟡',
            'SATURATION': '⚪',
            'NEUTRAL_NEWS': '⚫'
        }

        color_emoji = qualia_colors.get(news['qualia'], '⚫')

        # Сентимент
        sentiment = news.get('sentiment', 0)
        if sentiment > 0.1:
            sentiment_emoji = "😊"
        elif sentiment < -0.1:
            sentiment_emoji = "😔"
        else:
            sentiment_emoji = "😐"

        # Время
        time_str = news['timestamp'].strftime("%H:%M:%S")

        # Отображение
        st.markdown(f"""
        <div style="border-left: 4px solid {'green' if 'HOPE' in news['qualia'] else 'red' if 'DREAD' in news['qualia'] else 'gray'}; padding: 10px; margin: 5px 0; background-color: #f8f9fa; border-radius: 5px;">
            <strong>{news['title']}</strong><br>
            <small>{color_emoji} Квалиа: {news['qualia']} | {sentiment_emoji} Сентимент: {sentiment:.2f} | 📡 {news['source']} | 🕐 {time_str}</small><br>
            {news['text']}
        </div>
        """, unsafe_allow_html=True)

    def _create_stats_area(self):
        """Создание области статистики."""
        st.subheader("📊 Статистика")

        if not self.stats_history:
            st.info("Статистика появится после обработки новостей...")
            return

        # График стабильности
        if len(self.stats_history) > 1:
            recent_stats = self.stats_history[-20:]  # Последние 20 точек

            # График стабильности
            stability_data = [s['stability'] for s in recent_stats]
            fig_stability = go.Figure()
            fig_stability.add_trace(go.Scatter(
                y=stability_data,
                mode='lines+markers',
                name='Стабильность',
                line=dict(color='blue', width=2)
            ))
            fig_stability.update_layout(
                title="Стабильность системы",
                height=200,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_stability, use_container_width=True)

            # Распределение квалиа
            qualia_counts = {}
            for stat in recent_stats:
                qualia = stat['qualia']
                qualia_counts[qualia] = qualia_counts.get(qualia, 0) + 1

            if qualia_counts:
                fig_qualia = px.pie(
                    values=list(qualia_counts.values()),
                    names=list(qualia_counts.keys()),
                    title="Распределение квалиа"
                )
                fig_qualia.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_qualia, use_container_width=True)

    def _create_debug_area(self):
        """Создание области отладки."""
        # Системная информация
        st.subheader("🔍 Системная информация")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Внутреннее состояние:**")
            if self.reader:
                internal = self.reader.internal_state
                st.json({
                    'stability': internal.get('stability', 0.0),
                    'coherence': internal.get('coherence', 0.0),
                    'experience_log_length': len(internal.get('experience_log', []))
                })

        with col2:
            st.write("**Статистика фильтрации:**")
            if self.mood_filter:
                filter_stats = self.mood_filter.get_filtering_stats()
                st.json({
                    'total_filtered': filter_stats.get('total_filtered', 0),
                    'adaptation_cycles': filter_stats.get('adaptation_cycles', 0),
                    'mood_profiles': filter_stats.get('mood_profiles_count', 0)
                })

        # Статистика источников
        st.subheader("📡 Статистика источников")
        sources_stats = self.source_manager.get_stats()
        st.json({
            'total_fetched': sources_stats.get('total_fetched', 0),
            'cache_size': sources_stats.get('cache_size', 0),
            'duplicates_filtered': sources_stats.get('duplicates_filtered', 0)
        })

        # История квалиа
        if self.reader and self.reader.internal_state.get('experience_log'):
            st.subheader("📝 История квалиа")
            recent_qualia = self.reader.internal_state['experience_log'][-10:]
            qualia_df = pd.DataFrame([
                {'Квалиа': entry.get('qualia', 'unknown')} for entry in recent_qualia
            ])
            st.dataframe(qualia_df, use_container_width=True)

def main():
    """Запуск приложения."""
    app = NewsStreamApp()
    app.create_streamlit_app()

    # Автоматическое обновление
    if app.is_running:
        time.sleep(app.update_interval)
        app.run_news_cycle()
        st.rerun()

if __name__ == "__main__":
    main()