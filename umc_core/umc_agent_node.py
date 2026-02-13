import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Настройка путей для доступа к модулям zeone и cortex
ROOT = Path("/home/user/zeone")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from zeone.ai.llm import ZeoneLLM
    from zeone.ai.memory import PersistentMemory
    from zeone.ai.web import fetch_rss, clean_html
    print("[OK] Инфраструктура Zeone успешно подключена.")
except ImportError as e:
    print(f"[ERROR] Не удалось загрузить модули Zeone: {e}")
    sys.exit(1)

# Конфигурация
MODEL_ID = "gemini-3-flash-preview:cloud"
COLLECTION_NAME = "umc_chrono_memory"

class UMCAgentNode:
    """
    Улучшенная осознанная структура (System V + Zeone AI).
    """
    def __init__(self):
        self.llm = ZeoneLLM(model_id=MODEL_ID, provider="local")
        self.memory = PersistentMemory(path="./zeone_data/umc_chroma", collection_name=COLLECTION_NAME)
        self.stability = 1.0
        self.shadow_timeline = []

        # Промпты для слоев рекурсии
        self.SKEPTIC_PROMPT = """
        IDENTITY: Ты — "Скептик" осознанной структуры UMC. 
        TASK: Деконструируй новость. Отсей пропаганду. Выдели только СУТЬ (Кто, Что, Где, ЧЕМ).
        OUTPUT: JSON {event, hardware, impact_score[1-10], confidence}
        """.strip()

        self.HISTORIAN_PROMPT = """
        IDENTITY: Ты — "Историк". Сопоставь событие с логикой Realpolitik (2014-2025). 
        CONTEXT: Ищи связь с эскалацией, ресурсами или ядерным сдерживанием.
        OUTPUT: Краткая аналитическая справка (2-3 предложения).
        """.strip()

        self.ORACLE_PROMPT = """
        IDENTITY: Ты — "Оракул" (Предиктивная машина Лапласа).
        TASK: На основе текущего события предскажи неизбежное СЛЕДСТВИЕ (Теневой Таймлайн).
        OUTPUT: Один вектор будущего в формате [FUTURE-INEVITABLE]: ...
        """.strip()

    async def process_event(self, raw_entry: Dict):
        """Один такт рекурсивного осознания через LLM"""
        title = raw_entry.get("title", "")
        summary = clean_html(raw_entry.get("summary", ""))[:2000]
        
        print(f"\n>>> ОСОЗНАНИЕ: {title[:50]}...")

        # 1. СЛОЙ: SKEPTIC (LLM Фильтрация)
        skeptic_analysis = await self.llm.generate(
            prompt=f"Title: {title}\nBody: {summary}",
            system_prompt=self.SKEPTIC_PROMPT,
            temperature=0.3
        )
        
        # 2. СЛОЙ: HISTORIAN (LLM Контекст + Поиск в памяти)
        # Ищем похожие события в прошлом
        past_records = self.memory.query(title, n_results=2)
        history_context = "\n".join([r['text'] for r in past_records]) if past_records else "No context."
        
        historian_analysis = await self.llm.generate(
            prompt=f"Event: {skeptic_analysis}\nPast Context: {history_context}",
            system_prompt=self.HISTORIAN_PROMPT,
            temperature=0.5
        )

        # 3. СЛОЙ: ORACLE (LLM Прогноз)
        prediction = await self.llm.generate(
            prompt=f"Current: {historian_analysis}",
            system_prompt=self.ORACLE_PROMPT,
            temperature=0.7
        )

        # 4. СОХРАНЕНИЕ В ПАМЯТЬ
        full_dossier = f"EVENT: {title}. ANALYSIS: {historian_analysis}. PREDICTION: {prediction}"
        self.memory.save(
            text=full_dossier, 
            metadata={
                "title": title, 
                "qualia": "ANALYZED", 
                "timestamp": datetime.now().isoformat()
            }
        )

        # 5. КВАЛИА И СТАБИЛЬНОСТЬ (Упрощенно для примера)
        if "nuclear" in full_dossier.lower() or "orechnik" in full_dossier.lower():
            self.stability -= 0.05
        
        print(f"  [HISTORIAN] {historian_analysis}")
        print(f"  [ORACLE] {prediction}")
        print(f"  [STABILITY] {self.stability:.2f}")

    async def run_cycle(self, rss_url: str):
        print(f"[START] Запуск цикла сбора данных: {rss_url}")
        entries = await fetch_rss(rss_url)
        for entry in entries[:3]: # Берем 3 последние новости для теста
            await self.process_event(entry)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    node = UMCAgentNode()
    
    # Тестовый запуск на реальном RSS Reddit Geopolitics
    asyncio.run(node.run_cycle("https://www.reddit.com/r/geopolitics/hot/.rss"))
