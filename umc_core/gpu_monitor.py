#!/usr/bin/env python3
"""GPUMonitor: query nvidia-smi for GPU stats."""

import subprocess
from typing import List, Dict


class GPUMonitor:
    def _query(self) -> List[Dict[str, str]]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return []
        gpus = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append({
                "id": int(parts[0]),
                "name": parts[1],
                "memory_used": int(parts[2]),
                "memory_total": int(parts[3]),
                "utilization": int(parts[4]),
                "temperature": int(parts[5]),
            })
        return gpus

    def get_gpus(self) -> List[Dict]:
        return self._query()

    def get_gpu(self, gpu_id: int) -> Dict:
        for g in self._query():
            if g["id"] == gpu_id:
                return g
        return {}

    def is_healthy(self) -> bool:
        gpus = self._query()
        return all(g.get("temperature", 100) < 80 for g in gpus) if gpus else False
