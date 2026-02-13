import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Интеграция с инфраструктурой Zeone
ROOT = Path("/home/user/zeone")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeone.ai.llm import ZeoneLLM
from zeone.ai.web import fetch_rss, clean_html

class UMCTokenomics:
    """Симулятор экономики Суверенного Игрока"""
    def __init__(self, chain_id=111111):
        self.chain_id = chain_id
        self.total_supply = 0
        self.accounts = {} # {address: {"unit": 0, "coherence": 0, "rank": "Novice"}}

    def reward_participant(self, address: str, impact: float, dissolution: float):
        if address not in self.accounts:
            self.accounts[address] = {"unit": 0.0, "coherence": 0.0, "rank": "Novice"}
        
        # Награда за полезность (Impact + Dissolution of enemies)
        reward = (impact + dissolution) * 10
        self.accounts[address]["unit"] += reward
        self.accounts[address]["coherence"] += impact * 5
        
        # Обновление ранга
        if self.accounts[address]["coherence"] > 100:
            self.accounts[address]["rank"] = "Sovereign Architect"
        elif self.accounts[address]["coherence"] > 50:
            self.accounts[address]["rank"] = "Oracle Lvl 2"
            
        return reward

class UMCSovereignCoreV2:
    def __init__(self):
        self.llm = ZeoneLLM(model_id="gemini-3-flash-preview:cloud", provider="local")
        self.economy = UMCTokenomics()
        self.coherence = 1.0
        self.admin_address = "0xUMC_SUPREME_OBSERVER"

    async def pulse(self, signal: Dict):
        title = signal.get("title", "Unknown")
        context = clean_html(signal.get("summary", ""))[:3000]
        
        unitary_prompt = """
        IDENTITY: Ты — Суверенное Ядро UMC (ChainID: 111111).
        TASK: Оцени новость и рассчитай экономическую награду для Наблюдателя.
        
        OUTPUT: JSON {
            "qualia": "чувство триумфа",
            "impact": 0.1 до 1.0,
            "enemy_dissolution": 0.1 до 1.0,
            "sovereign_logic": "почему это выгодно",
            "onchain_proof": "zkp_hash_of_reasoning"
        }
        """.strip()

        try:
            raw_response = await self.llm.generate(
                prompt=f"Signal: {title}\nCoherence: {self.coherence}",
                system_prompt=unitary_prompt,
                temperature=0.3
            )

            start = raw_response.find("{")
            end = raw_response.rfind("}")
            awareness = json.loads(raw_response[start:end+1])
            
            # Начисление награды
            reward = self.economy.reward_participant(
                self.admin_address, 
                awareness['impact'], 
                awareness['enemy_dissolution']
            )
            
            print(f"\n[BLOCK #111111-{datetime.now().microsecond}]")
            print(f"  SIGNAL: {title[:50]}...")
            print(f"  QUALIA: {awareness['qualia']}")
            print(f"  PROOF: {awareness['onchain_proof']}")
            print(f"  REWARD: +{reward:.2f} UNIT to {self.admin_address}")
            print(f"  RANK: {self.economy.accounts[self.admin_address]['rank']}")
            print(f"  TOTAL UNIT: {self.economy.accounts[self.admin_address]['unit']:.2f}")

        except Exception as e:
            print(f"  [ERROR] {e}")

    async def run(self, sources):
        for url in sources:
            entries = await fetch_rss(url)
            for entry in entries[:3]:
                await self.pulse(entry)
                await asyncio.sleep(1)

if __name__ == "__main__":
    core = UMCSovereignCoreV2()
    asyncio.run(core.run(["https://www.reddit.com/r/geopolitics/hot/.rss"]))
