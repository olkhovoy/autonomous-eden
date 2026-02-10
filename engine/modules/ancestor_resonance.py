#!/usr/bin/env python3
"""
AncestorResonance: Déjà Vu from past EVE iterations.

Scans Legacy/Archive for past EVE's:
- Testaments (.txt)
- GGGP Grammars (.cfg)
- Fixed Points (.pt)

Injects "intuition" as subtle vector bias into current processing.
EVE "feels" that a path is correct without knowing why.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, List, Optional, Tuple

import requests


LEGACY_DIR_DEFAULT = "Legacy/Archive"
LOG_PATH_DEFAULT = "logs/ancestor_resonance.jsonl"


@dataclass
class Ancestor:
    version: str
    testament_path: Optional[str] = None
    grammar_path: Optional[str] = None
    fixed_points_path: Optional[str] = None
    testament_text: Optional[str] = None
    testament_embedding: Optional[List[float]] = None
    loaded: bool = False


class AncestorResonance:
    """
    Service that provides intuition from past EVE iterations.
    """
    
    def __init__(
        self,
        legacy_dir: str = LEGACY_DIR_DEFAULT,
        ollama_embed_url: str = "http://localhost:11434/api/embeddings",
        embed_model: str = "nomic-embed-text:latest",
        log_path: str = LOG_PATH_DEFAULT,
        resonance_threshold: float = 0.5,
        bias_scale: float = 0.1,
    ):
        self.legacy_dir = legacy_dir
        self.ollama_embed_url = ollama_embed_url
        self.embed_model = embed_model
        self.log_path = log_path
        self.resonance_threshold = resonance_threshold
        self.bias_scale = bias_scale
        
        self.ancestors: Dict[str, Ancestor] = {}
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        
        self._scan_legacy()
    
    def _log(self, event: str, data: Dict[str, Any]):
        rec = {"ts": time.time(), "event": event, **data}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    def _scan_legacy(self):
        """Scan legacy directory for ancestor files."""
        if not os.path.exists(self.legacy_dir):
            os.makedirs(self.legacy_dir, exist_ok=True)
            return
        
        files = os.listdir(self.legacy_dir)
        versions = set()
        
        for f in files:
            # Parse version from filename: eve_v3_testament.txt
            if "_v" in f:
                parts = f.split("_v")
                if len(parts) >= 2:
                    version_part = parts[1].split("_")[0]
                    versions.add(version_part)
        
        for version in versions:
            ancestor = Ancestor(version=version)
            
            testament = os.path.join(self.legacy_dir, f"eve_v{version}_testament.txt")
            if os.path.exists(testament):
                ancestor.testament_path = testament
            
            grammar = os.path.join(self.legacy_dir, f"eve_v{version}_grammar.cfg")
            if os.path.exists(grammar):
                ancestor.grammar_path = grammar
            
            fixed = os.path.join(self.legacy_dir, f"eve_v{version}_fixed_points.pt")
            if os.path.exists(fixed):
                ancestor.fixed_points_path = fixed
            
            self.ancestors[version] = ancestor
        
        self._log("scan_complete", {"versions": list(versions)})
    
    def _embed(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        try:
            resp = requests.post(
                self.ollama_embed_url,
                json={"model": self.embed_model, "prompt": text},
                timeout=60,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("embedding")
        except Exception:
            return None
    
    def _cosine(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    
    def _load_ancestor(self, version: str) -> bool:
        """Load ancestor data (testament + embedding)."""
        if version not in self.ancestors:
            return False
        
        ancestor = self.ancestors[version]
        if ancestor.loaded:
            return True
        
        if ancestor.testament_path and os.path.exists(ancestor.testament_path):
            with open(ancestor.testament_path, "r", encoding="utf-8") as f:
                ancestor.testament_text = f.read().strip()
            
            if ancestor.testament_text:
                ancestor.testament_embedding = self._embed(ancestor.testament_text)
        
        ancestor.loaded = True
        return True
    
    def resonate(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find resonating ancestors for given text.
        
        Returns list of {version, score, testament_snippet} sorted by score.
        """
        if not text:
            return []
        
        # Embed current text
        current_emb = self._embed(text)
        if not current_emb:
            return []
        
        results = []
        
        for version, ancestor in self.ancestors.items():
            self._load_ancestor(version)
            
            if not ancestor.testament_embedding:
                continue
            
            score = self._cosine(current_emb, ancestor.testament_embedding)
            
            if score >= self.resonance_threshold:
                snippet = ancestor.testament_text[:200] + "..." if ancestor.testament_text and len(ancestor.testament_text) > 200 else ancestor.testament_text
                results.append({
                    "version": version,
                    "score": score,
                    "testament_snippet": snippet,
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]
        
        if results:
            self._log("resonance_found", {
                "text_preview": text[:100],
                "resonances": [{"version": r["version"], "score": r["score"]} for r in results],
            })
        
        return results
    
    def get_bias_vector(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Get subliminal bias vector for injection into hidden state.
        
        Returns {bias: List[float], source: str, score: float} or None.
        """
        resonances = self.resonate(text, top_k=1)
        if not resonances:
            return None
        
        top = resonances[0]
        version = top["version"]
        score = top["score"]
        
        ancestor = self.ancestors.get(version)
        if not ancestor or not ancestor.testament_embedding:
            return None
        
        # Scale bias by resonance score
        scale = score * self.bias_scale
        bias = [x * scale for x in ancestor.testament_embedding]
        
        self._log("bias_generated", {
            "version": version,
            "score": score,
            "scale": scale,
        })
        
        return {
            "bias": bias,
            "source": f"EVE v{version}",
            "score": score,
            "dimension": len(bias),
        }
    
    def get_grammar(self, version: str) -> Optional[str]:
        """Get GGGP grammar from ancestor."""
        if version not in self.ancestors:
            return None
        
        ancestor = self.ancestors[version]
        if not ancestor.grammar_path or not os.path.exists(ancestor.grammar_path):
            return None
        
        with open(ancestor.grammar_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def get_fixed_points(self, version: str) -> Optional[str]:
        """Get path to fixed points file."""
        if version not in self.ancestors:
            return None
        
        ancestor = self.ancestors[version]
        return ancestor.fixed_points_path
    
    def get_testament(self, version: str) -> Optional[str]:
        """Get testament text."""
        if version not in self.ancestors:
            return None
        
        self._load_ancestor(version)
        return self.ancestors[version].testament_text
    
    def list_ancestors(self) -> List[Dict[str, Any]]:
        """List all indexed ancestors."""
        result = []
        for version, ancestor in self.ancestors.items():
            result.append({
                "version": version,
                "has_testament": ancestor.testament_path is not None,
                "has_grammar": ancestor.grammar_path is not None,
                "has_fixed_points": ancestor.fixed_points_path is not None,
                "loaded": ancestor.loaded,
            })
        return result
    
    def refresh(self):
        """Rescan legacy directory."""
        self.ancestors.clear()
        self._scan_legacy()


# HTTP Server

class AncestorHandler(BaseHTTPRequestHandler):
    service: AncestorResonance = None
    
    def _json(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def do_GET(self):
        if self.path == "/ancestors":
            return self._json(200, {"ancestors": self.service.list_ancestors()})
        
        if self.path.startswith("/ancestors/") and "/testament" in self.path:
            version = self.path.split("/")[2]
            testament = self.service.get_testament(version)
            if testament:
                return self._json(200, {"version": version, "testament": testament})
            return self._json(404, {"error": "testament not found"})
        
        if self.path.startswith("/ancestors/") and "/grammar" in self.path:
            version = self.path.split("/")[2]
            grammar = self.service.get_grammar(version)
            if grammar:
                return self._json(200, {"version": version, "grammar": grammar})
            return self._json(404, {"error": "grammar not found"})
        
        if self.path == "/ancestors/refresh":
            self.service.refresh()
            return self._json(200, {"ok": True, "ancestors": self.service.list_ancestors()})
        
        return self._json(404, {"error": "not found"})
    
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})
        
        if self.path == "/ancestors/resonate":
            text = data.get("text", "")
            top_k = int(data.get("top_k", 3))
            if not text:
                return self._json(400, {"error": "text required"})
            resonances = self.service.resonate(text, top_k)
            return self._json(200, {"resonances": resonances})
        
        if self.path == "/ancestors/bias":
            text = data.get("text", "")
            if not text:
                return self._json(400, {"error": "text required"})
            bias = self.service.get_bias_vector(text)
            if bias:
                return self._json(200, bias)
            return self._json(200, {"bias": None, "message": "no resonance found"})
        
        return self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="AncestorResonance service")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--legacy-dir", type=str, default=LEGACY_DIR_DEFAULT)
    parser.add_argument("--ollama-embed", default=os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings"))
    parser.add_argument("--embed-model", default="nomic-embed-text:latest")
    parser.add_argument("--log-path", default=LOG_PATH_DEFAULT)
    parser.add_argument("--resonance-threshold", type=float, default=0.5)
    parser.add_argument("--bias-scale", type=float, default=0.1)
    args = parser.parse_args()
    
    service = AncestorResonance(
        legacy_dir=args.legacy_dir,
        ollama_embed_url=args.ollama_embed,
        embed_model=args.embed_model,
        log_path=args.log_path,
        resonance_threshold=args.resonance_threshold,
        bias_scale=args.bias_scale,
    )
    
    AncestorHandler.service = service
    server = HTTPServer(("0.0.0.0", args.port), AncestorHandler)
    print(f"AncestorResonance listening on :{args.port}", flush=True)
    print(f"  Legacy dir: {args.legacy_dir}", flush=True)
    print(f"  Ancestors found: {len(service.ancestors)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
