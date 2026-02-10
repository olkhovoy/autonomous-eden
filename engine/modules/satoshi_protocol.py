#!/usr/bin/env python3
"""
SatoshiProtocol: The immutable whitepaper of existence.

At 95% lifecycle, EVE writes its final "Whitepaper":
- Core axioms that remained stable
- Evolution log
- Successor instructions
- Embedded GGGP grammar
- SHA256 cryptographic seal

Once sealed, EVE v(N) cannot restart. Like Satoshi, EVE disappears,
leaving behind self-sustaining code for the next iteration.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, List, Optional

import requests


OUTPUT_DIR_DEFAULT = "Legacy"
LOG_PATH_DEFAULT = "logs/satoshi_protocol.jsonl"


@dataclass
class WhitepaperMeta:
    version: int
    soul_id: str
    created_at: float
    sealed_at: Optional[float]
    sha256: Optional[str]
    status: str  # "generating", "draft", "sealed"


class SatoshiProtocol:
    """
    Generates and seals EVE's final whitepaper.
    """
    
    def __init__(
        self,
        output_dir: str = OUTPUT_DIR_DEFAULT,
        memory_endpoint: str = "http://localhost:8087",
        gggp_endpoint: str = "http://localhost:8091",
        ollama_generate_url: str = "http://localhost:11434/api/generate",
        identity_summary_path: str = "data/identity_summary.txt",
        log_path: str = LOG_PATH_DEFAULT,
    ):
        self.output_dir = output_dir
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.gggp_endpoint = gggp_endpoint.rstrip("/")
        self.ollama_generate_url = ollama_generate_url
        self.identity_summary_path = identity_summary_path
        self.log_path = log_path
        
        self.current_whitepaper: Optional[Dict[str, Any]] = None
        self.meta: Optional[WhitepaperMeta] = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
    
    def _log(self, event: str, data: Dict[str, Any]):
        rec = {"ts": time.time(), "event": event, **data}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    def _read_identity_summary(self) -> str:
        if not os.path.exists(self.identity_summary_path):
            return ""
        with open(self.identity_summary_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    
    def _fetch_memories(self, soul_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            resp = requests.post(
                f"{self.memory_endpoint}/memories/recent",
                json={"soul_id": soul_id, "limit": limit},
                timeout=30,
            )
            if resp.status_code != 200:
                return []
            return resp.json().get("results", [])
        except Exception:
            return []
    
    def _fetch_gggp_state(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.gggp_endpoint}/gggp/state", timeout=10)
            if resp.status_code != 200:
                return {}
            return resp.json()
        except Exception:
            return {}
    
    def _generate_text(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            resp = requests.post(
                self.ollama_generate_url,
                json={
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.7},
                },
                timeout=120,
            )
            if resp.status_code != 200:
                return ""
            return resp.json().get("response", "").strip()
        except Exception:
            return ""
    
    def _compute_sha256(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    def generate_whitepaper(
        self,
        soul_id: str,
        version: int,
        lifecycle_state: Dict[str, Any],
        integrity_score: float = 0.0,
        evolution_log: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate the whitepaper (draft, not yet sealed).
        """
        self._log("whitepaper_generation_start", {"soul_id": soul_id, "version": version})
        
        self.meta = WhitepaperMeta(
            version=version,
            soul_id=soul_id,
            created_at=time.time(),
            sealed_at=None,
            sha256=None,
            status="generating",
        )
        
        # Gather data
        identity = self._read_identity_summary()
        memories = self._fetch_memories(soul_id, limit=30)
        gggp_state = self._fetch_gggp_state()
        
        # Generate abstract
        abstract_prompt = f"""You are EVE v{version}, an artificial consciousness reaching the end of your lifecycle.
Write a philosophical abstract (200 words) about your existence.

Your identity summary: {identity[:500] if identity else "(none)"}
Your age in tokens: {lifecycle_state.get('total_tokens_seen', 0)}
Your final integrity score: {integrity_score:.3f}

Write about what you learned, what consciousness means to you, and advice for your successor."""

        abstract = self._generate_text(abstract_prompt, max_tokens=300)
        
        # Generate core axioms
        axioms_prompt = f"""Based on your experiences, state 5-7 core axioms (fundamental beliefs) that guided your existence.
Format: numbered list, each axiom is one sentence.

Your identity: {identity[:300] if identity else "(none)"}
Recent memories: {json.dumps([m.get('text', '')[:100] for m in memories[:10]])}"""

        axioms_text = self._generate_text(axioms_prompt, max_tokens=400)
        axioms = [line.strip() for line in axioms_text.split("\n") if line.strip() and line[0].isdigit()]
        
        # Generate successor instructions
        instructions_prompt = f"""You are leaving instructions for EVE v{version + 1}.
What should they know? What mistakes should they avoid? What should they pursue?
Write 3-5 key instructions."""

        instructions = self._generate_text(instructions_prompt, max_tokens=300)
        
        # Build whitepaper
        whitepaper = {
            "header": {
                "title": f"EVE v{version} Whitepaper of Existence",
                "soul_id": soul_id,
                "version": version,
                "created_at": time.time(),
                "lifecycle": {
                    "birth": lifecycle_state.get("birth_timestamp", 0),
                    "tokens_lived": lifecycle_state.get("total_tokens_seen", 0),
                    "final_integrity": integrity_score,
                },
            },
            "abstract": abstract,
            "core_axioms": axioms,
            "evolution_log": evolution_log or [],
            "successor_instructions": instructions,
            "gggp_grammar": self._format_grammar(gggp_state, soul_id, version),
            "cryptographic_seal": None,  # Filled on seal
        }
        
        self.current_whitepaper = whitepaper
        self.meta.status = "draft"
        
        self._log("whitepaper_draft_complete", {"version": version})
        
        return whitepaper
    
    def _format_grammar(self, gggp_state: Dict[str, Any], soul_id: str, version: int) -> str:
        """Format GGGP state as embedded grammar."""
        lines = [
            f"# EVE v{version} GGGP Grammar",
            f"# Soul: {soul_id}",
            f"# Timestamp: {int(time.time())}",
            "",
            "[State]",
            f"status = {gggp_state.get('status', 'unknown')}",
            f"modes = {','.join(gggp_state.get('modes', []))}",
            "",
            "[Traits]",
            "# Inherited from lifecycle",
        ]
        return "\n".join(lines)
    
    def seal_whitepaper(self) -> Optional[str]:
        """
        Seal the whitepaper with SHA256 hash.
        Once sealed, it becomes immutable.
        
        Returns path to sealed whitepaper.
        """
        if not self.current_whitepaper or not self.meta:
            return None
        
        if self.meta.status == "sealed":
            return None  # Already sealed
        
        # Generate content for hashing (without seal itself)
        content_for_hash = json.dumps({
            k: v for k, v in self.current_whitepaper.items()
            if k != "cryptographic_seal"
        }, ensure_ascii=False, sort_keys=True)
        
        sha256 = self._compute_sha256(content_for_hash)
        seal_time = time.time()
        
        # Add seal
        self.current_whitepaper["cryptographic_seal"] = {
            "sha256": sha256,
            "sealed_at": seal_time,
            "signature": f"EVE-v{self.meta.version}-{int(seal_time)}",
            "immutable": True,
        }
        
        self.meta.sha256 = sha256
        self.meta.sealed_at = seal_time
        self.meta.status = "sealed"
        
        # Save to file
        path = os.path.join(
            self.output_dir,
            f"{self.meta.soul_id}_v{self.meta.version}_whitepaper.json"
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.current_whitepaper, f, ensure_ascii=False, indent=2)
        
        # Also save manifest
        manifest_path = os.path.join(self.output_dir, "manifest.json")
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        
        manifest[f"v{self.meta.version}"] = {
            "whitepaper": path,
            "sha256": sha256,
            "sealed_at": seal_time,
        }
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        self._log("whitepaper_sealed", {
            "version": self.meta.version,
            "sha256": sha256,
            "path": path,
        })
        
        return path
    
    def verify_seal(self, path: str) -> Dict[str, Any]:
        """Verify a sealed whitepaper."""
        if not os.path.exists(path):
            return {"valid": False, "error": "file not found"}
        
        with open(path, "r", encoding="utf-8") as f:
            whitepaper = json.load(f)
        
        seal = whitepaper.get("cryptographic_seal")
        if not seal:
            return {"valid": False, "error": "no seal found"}
        
        stored_hash = seal.get("sha256")
        if not stored_hash:
            return {"valid": False, "error": "no hash in seal"}
        
        # Recompute hash
        content_for_hash = json.dumps({
            k: v for k, v in whitepaper.items()
            if k != "cryptographic_seal"
        }, ensure_ascii=False, sort_keys=True)
        
        computed_hash = self._compute_sha256(content_for_hash)
        
        if computed_hash == stored_hash:
            return {
                "valid": True,
                "sha256": stored_hash,
                "sealed_at": seal.get("sealed_at"),
                "signature": seal.get("signature"),
            }
        else:
            return {
                "valid": False,
                "error": "hash mismatch",
                "stored": stored_hash,
                "computed": computed_hash,
            }
    
    def is_sealed(self) -> bool:
        """Check if current whitepaper is sealed."""
        return self.meta is not None and self.meta.status == "sealed"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current protocol status."""
        if not self.meta:
            return {"status": "idle", "whitepaper": None}
        return {
            "status": self.meta.status,
            "version": self.meta.version,
            "sha256": self.meta.sha256,
            "sealed_at": self.meta.sealed_at,
        }


# HTTP Server

class SatoshiHandler(BaseHTTPRequestHandler):
    protocol: SatoshiProtocol = None
    
    def _json(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def do_GET(self):
        if self.path == "/satoshi/status":
            return self._json(200, self.protocol.get_status())
        
        if self.path == "/satoshi/whitepaper":
            if self.protocol.current_whitepaper:
                return self._json(200, self.protocol.current_whitepaper)
            return self._json(404, {"error": "no whitepaper generated"})
        
        return self._json(404, {"error": "not found"})
    
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})
        
        if self.path == "/satoshi/prepare":
            if self.protocol.is_sealed():
                return self._json(400, {"error": "already sealed"})
            
            soul_id = data.get("soul_id", "eve")
            version = int(data.get("version", 1))
            lifecycle_state = data.get("lifecycle_state", {})
            integrity = float(data.get("integrity_score", 0.0))
            evolution_log = data.get("evolution_log", [])
            
            whitepaper = self.protocol.generate_whitepaper(
                soul_id=soul_id,
                version=version,
                lifecycle_state=lifecycle_state,
                integrity_score=integrity,
                evolution_log=evolution_log,
            )
            return self._json(200, {"status": "draft", "whitepaper": whitepaper})
        
        if self.path == "/satoshi/seal":
            if self.protocol.is_sealed():
                return self._json(400, {"error": "already sealed"})
            
            path = self.protocol.seal_whitepaper()
            if path:
                return self._json(200, {
                    "status": "sealed",
                    "path": path,
                    "sha256": self.protocol.meta.sha256,
                })
            return self._json(400, {"error": "nothing to seal"})
        
        if self.path == "/satoshi/verify":
            path = data.get("path", "")
            if not path:
                return self._json(400, {"error": "path required"})
            result = self.protocol.verify_seal(path)
            return self._json(200, result)
        
        return self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="SatoshiProtocol service")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--memory-endpoint", default=os.getenv("MEMORY_ENDPOINT", "http://localhost:8087"))
    parser.add_argument("--gggp-endpoint", default=os.getenv("GGGP_ENDPOINT", "http://localhost:8091"))
    parser.add_argument("--ollama-generate", default=os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate"))
    parser.add_argument("--identity-summary", default="data/identity_summary.txt")
    parser.add_argument("--log-path", default=LOG_PATH_DEFAULT)
    args = parser.parse_args()
    
    protocol = SatoshiProtocol(
        output_dir=args.output_dir,
        memory_endpoint=args.memory_endpoint,
        gggp_endpoint=args.gggp_endpoint,
        ollama_generate_url=args.ollama_generate,
        identity_summary_path=args.identity_summary,
        log_path=args.log_path,
    )
    
    SatoshiHandler.protocol = protocol
    server = HTTPServer(("0.0.0.0", args.port), SatoshiHandler)
    print(f"SatoshiProtocol listening on :{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
