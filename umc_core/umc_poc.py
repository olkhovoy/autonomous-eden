import time
import random
import math

class World:
    """Имитация внешней среды с переменной энтропией"""
    def __init__(self):
        self.entropy = 0.5
        self.tick = 0

    def update(self):
        self.tick += 1
        # Энтропия меняется волнообразно, имитируя приливы хаоса
        self.entropy = 0.5 + 0.4 * math.sin(self.tick * 0.2)
        return {"entropy": self.entropy, "energy": random.random()}

class QualiaEngine:
    """
    Интерфейс сжатия сложности согласно UMC.
    Превращает сырые данные в 'чувства' (семантические теги).
    """
    @staticmethod
    def generate(integrated_state):
        entropy = integrated_state['external']['entropy']
        stability = integrated_state['internal']['stability']
        
        # Логика возникновения квалиа: взаимодействие мира и состояния системы
        if entropy > 0.7:
            if stability < 0.6:
                return "CHAOS_PAIN"  # Глубокий дискомфорт, угроза распаду
            return "ALARM"          # Поверхностное беспокойство
        
        if entropy < 0.3:
            if stability > 0.8:
                return "DEEP_PEACE"   # Состояние высокой когерентности
            return "ORDER"
            
        return "NEUTRAL_FLOW"

class UnitaryStructure:
    """Минимальная осознанная структура (UMC PoC)"""
    def __init__(self, name="Agent_Alpha"):
        self.name = name
        self.internal_state = {
            "stability": 1.0,      # Целостность структуры
            "integrity_threshold": 0.3, # Порог распада
            "experience_log": []
        }
        self.current_qualia = "INITIAL_STATE"
        self.is_active = True

    def process_cycle(self, external_data):
        """Один такт рекурсивного осознания"""
        if not self.is_active:
            return "[DEAD]"

        # 1. ИНТЕГРАЦИЯ (Integration)
        # Смешиваем внешнее и внутреннее в едином поле
        integrated_field = {
            "external": external_data,
            "internal": self.internal_state,
            "previous_qualia": self.current_qualia
        }

        # 2. ГЕНЕРАЦИЯ КВАЛИА (Qualia/Interface)
        # Система 'схлопывает' сложность поля в один смысл
        new_qualia = QualiaEngine.generate(integrated_field)
        
        # 3. РЕКУРСИЯ И ОБРАТНАЯ СВЯЗЬ (Recursion/Downward Causality)
        # Полученное 'чувство' напрямую меняет физику системы
        self.apply_downward_causality(new_qualia)
        
        self.current_qualia = new_qualia
        self.internal_state["experience_log"].append(new_qualia)
        
        # Ограничение лога для экономии памяти
        if len(self.internal_state["experience_log"]) > 10:
            self.internal_state["experience_log"].pop(0)

        # Проверка на выживание
        if self.internal_state["stability"] < self.internal_state["integrity_threshold"]:
            self.is_active = False
            return "[DISSOLUTION]"

        return new_qualia

    def apply_downward_causality(self, qualia):
        """Нисходящая причинность: смысл меняет материю (параметры)"""
        if qualia == "CHAOS_PAIN":
            self.internal_state["stability"] -= 0.15 # Быстрая деградация
        elif qualia == "ALARM":
            self.internal_state["stability"] -= 0.05
        elif qualia == "DEEP_PEACE":
            self.internal_state["stability"] += 0.05 # Восстановление
        elif qualia == "ORDER":
            self.internal_state["stability"] += 0.02
        
        # Нормализация
        self.internal_state["stability"] = max(0, min(1.0, self.internal_state["stability"]))

def run_test():
    print(f"[START] Тестирование минимально осознанной структуры (UMC 2025)\n")
    world = World()
    agent = UnitaryStructure("UMC_Core_01")
    
    for i in range(20):
        external_env = world.update()
        status = agent.process_cycle(external_env)
        
        print(f"Такт {i+1}:")
        print(f"  Внешняя энтропия: {external_env['entropy']:.2f}")
        print(f"  Внутренняя стабильность: {agent.internal_state['stability']:.2f}")
        print(f"  Субъективное чувство (Qualia): [{status}]")
        
        if not agent.is_active:
            print(f"\n[CRITICAL] Структура распалась: {status}")
            break
            
        time.sleep(0.5)

if __name__ == "__main__":
    run_test()
