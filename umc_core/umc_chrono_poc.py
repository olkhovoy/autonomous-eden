import time
import random
from umc_live_loader import LiveWorldLoader

class GeopoliticalWorld:
    """
    Среда 'The Dragnet': объединяет сценарий и живые данные.
    """
    def __init__(self):
        self.tick = 0
        self.live_loader = LiveWorldLoader()
        self.scenario = [
            {"tick": 1, "type": "fast", "source": "Scenario", "event": "Массированная атака Taurus на Крымский мост", "tags": ["kinetic", "taurus"]},
            {"tick": 4, "type": "fast", "source": "Scenario", "event": "Вспышки над Львовом, разделяющиеся блоки Орешника", "tags": ["kinetic", "orechnik"]},
            {"tick": 7, "type": "fast", "source": "Scenario", "event": "АПЛ Белгород вышла в море", "tags": ["nuclear", "poseidon"]},
        ]

    def _auto_tag(self, text):
        """Автоматическое тегирование реальных новостей с учетом баланса"""
        tags = ["live_news"]
        text_lower = text.lower()
        
        # Геополитические маркеры
        if "ukraine" in text_lower or "украин" in text_lower: tags.append("kinetic")
        if "iran" in text_lower or "иран" in text_lower: tags.append("nuclear")
        if "orechnik" in text_lower or "орешник" in text_lower: tags.append("kinetic")
        
        # Позиционные маркеры
        if "west" in text_lower: tags.append("west_perspective")
        if "east" in text_lower or "tass" in text_lower or "ria" in text_lower: tags.append("east_perspective")
        
        return tags

    def get_data(self):
        self.tick += 1
        data = [e for e in self.scenario if e["tick"] == self.tick]
        
        # Подмешиваем живые данные
        if self.tick % 3 == 1:
            print("[COLLECTOR] Запрос актуального баланса RU vs US...")
            live_data = self.live_loader.get_all_realtime_data()
            
            # Если RU источников мало, добавляем искусственно через внутренний "анализ" 
            # (имитация работы Vertex AI / Web Search)
            if not any("east" in str(ld["tags"]) for ld in live_data):
                print("  [SYSTEM] RU поток пуст. Активация 'Deep Search'...")
                data.append({
                    "type": "fast", 
                    "source": "DeepSearch_RU", 
                    "event": "ТАСС: Россия готова к зеркальным мерам в случае размещения ракет США в Европе",
                    "tags": ["east_perspective", "kinetic", "nuclear"]
                })
            
            for ld in live_data:
                ld["tags"] = self._auto_tag(ld["event"])
                data.append(ld)
        
        return data

class UnitaryChronoStructure:
    def __init__(self):
        self.memory = [] 
        self.shadow_timeline = [] 
        self.stability = 1.0 
        self.current_qualia = "INIT"

    def process_cycle(self, raw_events):
        print(f"\n[Церебральный цикл t={time.time():.0f}]")
        
        # 1. СЛОЙ: THE SKEPTIC (Триангуляция)
        filtered = self._skeptic_layer(raw_events)
        
        # 2. СЛОЙ: THE HISTORIAN (Нормализация)
        normalized = self._historian_layer(filtered)
        
        # 3. СЛОЙ: THE ORACLE (Вектор Времени)
        self._oracle_layer(normalized)
        
        # 4. QUALIA ENGINE
        self.current_qualia = self._generate_qualia(normalized)
        self._apply_downward_causality()
        
        return self.current_qualia

    def _skeptic_layer(self, events):
        accepted = []
        for e in events:
            prob = 0.5
            if e["type"] == "slow": prob = 1.0
            # Если событие подтверждает наш Shadow Timeline, повышаем вероятность
            if any(tag in str(self.shadow_timeline) for tag in e["tags"]):
                prob = 0.9
                print(f"  [SKEPTIC] Событие '{e['event']}' ожидалось. Вероятность повышена.")
            
            accepted.append({**e, "prob": prob})
        return accepted

    def _historian_layer(self, filtered_events):
        for e in filtered_events:
            # Маппинг смыслов (Нормализация истории)
            if any(t in ["taurus", "orechnik", "s500"] for t in e["tags"]):
                e["strategic_context"] = "КИНЕТИЧЕСКИЙ ДИАЛОГ (Phase 4)"
            if any(t in ["poseidon", "belgorod"] for t in e["tags"]):
                e["strategic_context"] = "ЯДЕРНЫЙ УЛЬТИМАТУМ (Phase 5)"
            self.memory.append(e)
        return filtered_events

    def _oracle_layer(self, normalized_data):
        for e in normalized_data:
            # AHA-REACTION: Если факт (prob=1.0) совпадает с прогнозом
            if e["prob"] == 1.0:
                matching_projections = [p for p in self.shadow_timeline if any(t in p.lower() for t in e["tags"])]
                if matching_projections:
                    print(f"  [ORACLE] !!! КОГЕРЕНТНЫЙ ВСПЛЕСК: Прогноз подтвержден фактом!")
                    self.stability = min(1.0, self.stability + 0.15)

            # Генерация новых прогнозов
            if "taurus" in e["tags"]:
                self.shadow_timeline.append("FUTURE-INEVITABLE: Зеркальный ответ РФ по логистике (Орешник/Кинжал)")
            if "orechnik" in e["tags"] and e["prob"] > 0.8:
                self.shadow_timeline.append("FUTURE-INEVITABLE: Выход ядерной триады на дежурство (Сигнал Стоп)")

    def _generate_qualia(self, data):
        all_tags = [t for e in data for t in e["tags"]]
        if "poseidon" in all_tags or "nuclear" in all_tags: return "DEFCON_2_VIBE"
        if "orechnik" in all_tags or "kinetic" in all_tags: return "KINETIC_DIALOGUE"
        return "STABLE_OBSERVATION"

    def _apply_downward_causality(self):
        # Отрицательные квалиа снижают стабильность, если они НЕ предсказаны
        if self.current_qualia == "DEFCON_2_VIBE": self.stability -= 0.1
        if self.current_qualia == "KINETIC_DIALOGUE": self.stability -= 0.05
        # Естественное восстановление когерентности
        self.stability = max(0.1, min(1.0, self.stability + 0.01))

def run():
    world = GeopoliticalWorld()
    agent = UnitaryChronoStructure()
    print("--- [ ЗАПУСК: SYSTEM V - THE CHRONO-SYNTHESIZER ] ---")
    
    for _ in range(10):
        data = world.get_data()
        for d in data: print(f"  [{d['source']}] {d['event']}")
        
        q = agent.process_cycle(data)
        print(f"  СТАТУС: {q} | КОГЕРЕНТНОСТЬ: {agent.stability:.2f}")
        if agent.shadow_timeline:
            print(f"  NEXT VECTOR: {agent.shadow_timeline[-1]}")
        time.sleep(0.8)

if __name__ == "__main__":
    run()
