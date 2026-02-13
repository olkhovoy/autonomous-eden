#!/usr/bin/env python3
"""Legacy export: testament, grammar snapshot, fixed points."""

import os
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, List

import requests


@dataclass
class LegacyExport:
    output_dir: str = "Legacy"
    identity_summary_path: str = "data/identity_summary.txt"

    def _atomic_write(self, path: str, data: bytes):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

    def _read_identity_summary(self) -> str:
        if not os.path.exists(self.identity_summary_path):
            return ""
        with open(self.identity_summary_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _fetch_recent_memories(self, soul_id: str, memory_endpoint: str, limit: int = 50) -> List[Dict[str, Any]]:
        resp = requests.post(
            f"{memory_endpoint.rstrip('/')}/memories/recent",
            json={"soul_id": soul_id, "limit": limit},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    def generate_testament(self, soul_id: str, memory_endpoint: str, ollama_endpoint: str, version: str) -> str:
        memories = self._fetch_recent_memories(soul_id, memory_endpoint, limit=50)
        identity = self._read_identity_summary()
        lines = []
        if identity:
            lines.append(f"Identity Summary:\n{identity}")
        lines.append("Recent Memories:")
        for m in memories:
            lines.append(f"- {m.get('text','')}")
        prompt = (
            "You are writing a legacy testament for a future reincarnation. "
            "Compress identity, values, and key experiences into a 512-token testament.\n\n"
            + "\n".join(lines)
        )
        payload = {
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 512, "temperature": 0.6},
        }
        resp = requests.post(ollama_endpoint, json=payload, timeout=120)
        resp.raise_for_status()
        testament = resp.json().get("response", "").strip()
        path = os.path.join(self.output_dir, f"{soul_id}_v{version}_testament.txt")
        self._atomic_write(path, testament.encode("utf-8"))
        return path

    def export_grammar(self, gggp_endpoint: str, soul_id: str, version: str) -> str:
        resp = requests.get(f"{gggp_endpoint.rstrip('/')}/gggp/state", timeout=10)
        resp.raise_for_status()
        state = resp.json()
        cfg_lines = [
            "GGGP_STATE",
            f"SOUL_ID={soul_id}",
            f"VERSION={version}",
            f"TIMESTAMP={int(time.time())}",
            f"MODES={','.join(state.get('modes', []))}",
            f"GGGP_BIN={state.get('gggp_bin','')}",
            f"WORKDIR={state.get('workdir','')}",
        ]
        path = os.path.join(self.output_dir, f"{soul_id}_v{version}_grammar.cfg")
        self._atomic_write(path, "\n".join(cfg_lines).encode("utf-8"))
        return path

    def export_fixed_points(self, model_path: str, soul_id: str, version: str) -> str:
        import torch
        from dataclasses import fields
        from transformers import AutoTokenizer
        from benchmark.models.contractive_llama import ContractiveLlama, ContractiveLlamaConfig

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        if "config" not in checkpoint:
            raise ValueError("checkpoint missing config")
        valid_fields = {f.name for f in fields(ContractiveLlamaConfig)}
        filtered_config = {k: v for k, v in checkpoint["config"].items() if k in valid_fields}
        config = ContractiveLlamaConfig(**filtered_config)
        model = ContractiveLlama(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        tok = AutoTokenizer.from_pretrained('Xenova/llama-3-tokenizer')
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        identity = self._read_identity_summary()
        if not identity:
            identity = "(no identity summary)"
        enc = tok(identity, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            out = model(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                labels=None,
                return_all_losses=True,
            )
        payload = {
            "identity_text": identity,
            "input_ids": enc['input_ids'],
            "attention_mask": enc['attention_mask'],
            "final_hidden": out.get("final_hidden"),
        }
        path = os.path.join(self.output_dir, f"{soul_id}_v{version}_fixed_points.pt")
        tmp = f"{path}.tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path
