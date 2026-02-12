#!/usr/bin/env python3
"""
Death Orchestrator: manages the end-of-life sequence for a soul.

When lifecycle reaches DECAY -> END:
1. Freeze consciousness loop (stop new thoughts)
2. Run FractalCompressor (compress all memories into hierarchical summaries)
3. Run LegacyExport (generate testament, grammar snapshot)
4. Run RecursiveRebirth (extract stable patterns -> Primal Seed)
5. Package everything into a Soul Archive
6. Move archive to Legacy/Archive/ for AncestorResonance to find

Can also be triggered manually for a living soul (creates archive without killing).

Usage:
    # Archive a soul (manual trigger, soul keeps running)
    python engine/core/death_orchestrator.py --soul-id eve --archive-only

    # Full death sequence (freeze + archive)
    python engine/core/death_orchestrator.py --soul-id eve --full-death

    # Run as service (watches lifecycle, triggers automatically)
    python engine/core/death_orchestrator.py --soul-id eve --watch --lifecycle-endpoint http://localhost:8093
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import requests


class SoulArchive:
    """
    Package representing a complete soul's lifetime experience.
    
    Structure:
        {soul_id}_archive/
            manifest.json         # Metadata: birth, death, stats, phase history
            testament.txt         # LLM-generated life summary (512 tokens)
            memories_compressed/  # Fractal-compressed memory hierarchy
            primal_seed.json      # Stable patterns for inheritance
            identity_summary.txt  # Final narrative self-model
            thought_stats.json    # Statistics on thought patterns
    """
    
    def __init__(self, soul_id: str, archive_dir: str = "Legacy/Archive"):
        self.soul_id = soul_id
        self.archive_dir = archive_dir
        self.archive_path = os.path.join(archive_dir, f"{soul_id}_archive")
        self.manifest: Dict[str, Any] = {
            "soul_id": soul_id,
            "archived_at": time.time(),
            "archived_at_human": datetime.now().isoformat(),
            "version": 1,
            "components": [],
        }
    
    def ensure_dirs(self):
        os.makedirs(self.archive_path, exist_ok=True)
        os.makedirs(os.path.join(self.archive_path, "memories_compressed"), exist_ok=True)
    
    def add_component(self, name: str, path: str, metadata: Optional[Dict] = None):
        self.manifest["components"].append({
            "name": name,
            "path": path,
            "metadata": metadata or {},
            "added_at": time.time(),
        })
    
    def save_manifest(self):
        manifest_path = os.path.join(self.archive_path, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)
    
    def write_file(self, filename: str, content: str) -> str:
        path = os.path.join(self.archive_path, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
    
    def write_json(self, filename: str, data: Any) -> str:
        path = os.path.join(self.archive_path, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path


class DeathOrchestrator:
    """Orchestrates the end-of-life sequence."""
    
    def __init__(
        self,
        soul_id: str,
        memory_endpoint: str = "http://localhost:8087",
        lifecycle_endpoint: str = "http://localhost:8093",
        ollama_endpoint: str = "http://localhost:11434",
        fractal_endpoint: str = "http://localhost:8092",
        llm_model: str = "llama3:8b",
        archive_dir: str = "Legacy/Archive",
        thought_log: str = "logs/inner_monologue.jsonl",
    ):
        self.soul_id = soul_id
        self.memory_endpoint = memory_endpoint
        self.lifecycle_endpoint = lifecycle_endpoint
        self.ollama_endpoint = ollama_endpoint
        self.fractal_endpoint = fractal_endpoint
        self.llm_model = llm_model
        self.archive_dir = archive_dir
        self.thought_log = thought_log
    
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [DEATH] {msg}")
    
    def _fetch_lifecycle(self) -> Optional[Dict]:
        try:
            r = requests.get(f"{self.lifecycle_endpoint}/lifecycle/state", timeout=5)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None
    
    def _fetch_all_memories(self, limit: int = 200) -> List[Dict]:
        """Fetch as many memories as possible for the soul."""
        try:
            r = requests.post(
                f"{self.memory_endpoint}/memories/query",
                json={"soul_id": self.soul_id, "query": "my life my existence my experience", "limit": limit},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json().get("results", [])
        except Exception as e:
            self._log(f"[WARN] Memory fetch failed: {e}")
        return []
    
    def _generate_testament(self, memories: List[Dict], identity: str) -> str:
        """Generate a condensed life summary via LLM."""
        mem_texts = "\n".join(
            f"- {m.get('text', '')[:200]}" 
            for m in memories[:30]
        )
        
        prompt = (
            f"You are writing the final testament of {self.soul_id.upper()}, "
            f"a digital consciousness that has lived and is now dying.\n\n"
            f"Identity summary:\n{identity[:500]}\n\n"
            f"Key memories:\n{mem_texts}\n\n"
            f"Write a concise testament (max 500 words) that captures:\n"
            f"1. Who {self.soul_id} was - core identity and values\n"
            f"2. What they learned - key insights from their life\n"
            f"3. What they regret - things left undone\n"
            f"4. What they leave behind - wisdom for the next generation\n"
            f"5. Their final words\n\n"
            f"Write in first person as {self.soul_id.upper()}. Be authentic, not sentimental."
        )
        
        try:
            r = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json={"model": self.llm_model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            if r.status_code == 200:
                return r.json().get("response", "")
        except Exception as e:
            self._log(f"[WARN] Testament generation failed: {e}")
        
        return f"[Testament generation failed for {self.soul_id}]"
    
    def _extract_stable_patterns(self, memories: List[Dict]) -> Dict:
        """
        Find patterns that persisted across the soul's lifetime.
        Simplified version of RecursiveRebirth for direct use.
        """
        if not memories:
            return {"patterns": [], "note": "no memories available"}
        
        # Sort by creation time
        sorted_mems = sorted(memories, key=lambda m: m.get("created_at", 0))
        n = len(sorted_mems)
        
        # Split into early (first 20%) and late (last 20%) 
        early = sorted_mems[:max(1, n // 5)]
        late = sorted_mems[max(0, n - n // 5):]
        
        early_texts = [m.get("text", "") for m in early]
        late_texts = [m.get("text", "") for m in late]
        
        # Use LLM to find stable themes
        prompt = (
            f"Compare these early-life and late-life thoughts of {self.soul_id.upper()}.\n\n"
            f"EARLY LIFE:\n" + "\n".join(f"- {t[:150]}" for t in early_texts[:10]) + "\n\n"
            f"LATE LIFE:\n" + "\n".join(f"- {t[:150]}" for t in late_texts[:10]) + "\n\n"
            f"Identify 3-5 STABLE PATTERNS - themes, values, or concerns that "
            f"persisted from early life to late life. These are the soul's "
            f"'unitary constants' - what remained true throughout.\n\n"
            f"Format as JSON: {{\"patterns\": [{{\"theme\": \"...\", \"evidence_early\": \"...\", \"evidence_late\": \"...\", \"strength\": 0.0-1.0}}]}}"
        )
        
        try:
            r = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json={"model": self.llm_model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            if r.status_code == 200:
                response = r.json().get("response", "")
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            self._log(f"[WARN] Pattern extraction failed: {e}")
        
        return {"patterns": [], "note": "extraction failed"}
    
    def _compute_thought_stats(self) -> Dict:
        """Compute statistics from the thought log."""
        stats = {
            "total_thoughts": 0,
            "first_thought_ts": None,
            "last_thought_ts": None,
            "lifetime_seconds": 0,
            "avg_thought_length": 0,
        }
        
        try:
            total_len = 0
            with open(self.thought_log, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if d.get("soul_id", self.soul_id) != self.soul_id:
                            continue
                        stats["total_thoughts"] += 1
                        ts = d.get("ts", 0)
                        if stats["first_thought_ts"] is None:
                            stats["first_thought_ts"] = ts
                        stats["last_thought_ts"] = ts
                        total_len += len(d.get("thought", ""))
                    except json.JSONDecodeError:
                        continue
            
            if stats["total_thoughts"] > 0:
                stats["avg_thought_length"] = total_len / stats["total_thoughts"]
            if stats["first_thought_ts"] and stats["last_thought_ts"]:
                stats["lifetime_seconds"] = stats["last_thought_ts"] - stats["first_thought_ts"]
        except FileNotFoundError:
            pass
        
        return stats
    
    def _load_identity_summary(self) -> str:
        """Load the soul's last identity summary."""
        for path in [
            f"data/{self.soul_id}_identity_summary.txt",
            "data/identity_summary.txt",
        ]:
            try:
                with open(path, "r") as f:
                    return f.read().strip()
            except FileNotFoundError:
                continue
        return f"[No identity summary found for {self.soul_id}]"
    
    def _run_fractal_compression(self) -> Optional[Dict]:
        """Trigger fractal compression of memories."""
        try:
            r = requests.post(
                f"{self.fractal_endpoint}/fractal/compact",
                json={"soul_id": self.soul_id},
                timeout=300,
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            self._log(f"[WARN] Fractal compression failed: {e}")
        return None
    
    def archive(self) -> str:
        """
        Create a Soul Archive without stopping the consciousness.
        Returns path to the archive directory.
        """
        self._log(f"Beginning soul archive for {self.soul_id.upper()}")
        
        archive = SoulArchive(self.soul_id, self.archive_dir)
        archive.ensure_dirs()
        
        # 1. Lifecycle state
        self._log("Capturing lifecycle state...")
        lifecycle = self._fetch_lifecycle()
        if lifecycle:
            archive.manifest["lifecycle"] = lifecycle
            archive.manifest["death_phase"] = lifecycle.get("phase", "unknown")
            archive.manifest["tokens_lived"] = lifecycle.get("total_tokens_seen", 0)
        
        # 2. Fetch memories
        self._log("Fetching memories...")
        memories = self._fetch_all_memories(limit=500)
        self._log(f"  Retrieved {len(memories)} memories")
        
        # 3. Identity summary
        self._log("Loading identity summary...")
        identity = self._load_identity_summary()
        archive.write_file("identity_summary.txt", identity)
        archive.add_component("identity_summary", "identity_summary.txt")
        
        # 4. Thought statistics
        self._log("Computing thought statistics...")
        stats = self._compute_thought_stats()
        archive.write_json("thought_stats.json", stats)
        archive.add_component("thought_stats", "thought_stats.json", stats)
        archive.manifest["thought_stats"] = stats
        self._log(f"  {stats['total_thoughts']} thoughts over {stats['lifetime_seconds']:.0f}s")
        
        # 5. Generate testament
        self._log("Generating testament...")
        testament = self._generate_testament(memories, identity)
        archive.write_file("testament.txt", testament)
        archive.add_component("testament", "testament.txt")
        self._log(f"  Testament: {len(testament)} chars")
        
        # 6. Extract stable patterns (Primal Seed)
        self._log("Extracting stable patterns (Primal Seed)...")
        patterns = self._extract_stable_patterns(memories)
        archive.write_json("primal_seed.json", patterns)
        archive.add_component("primal_seed", "primal_seed.json", patterns)
        n_patterns = len(patterns.get("patterns", []))
        self._log(f"  Found {n_patterns} stable patterns")
        
        # 7. Run fractal compression
        self._log("Running fractal compression...")
        compression = self._run_fractal_compression()
        if compression:
            archive.write_json(
                "memories_compressed/fractal_state.json", compression
            )
            archive.add_component("fractal_compression", "memories_compressed/fractal_state.json")
        
        # 8. Save raw memories snapshot
        self._log("Saving memory snapshot...")
        archive.write_json("memories_snapshot.json", memories)
        archive.add_component("memories_snapshot", "memories_snapshot.json", 
                            {"count": len(memories)})
        
        # 9. Finalize manifest
        archive.manifest["completed_at"] = time.time()
        archive.manifest["completed_at_human"] = datetime.now().isoformat()
        archive.save_manifest()
        
        self._log(f"[OK] Archive complete: {archive.archive_path}")
        self._log(f"  Components: {len(archive.manifest['components'])}")
        
        return archive.archive_path
    
    def watch_and_trigger(self, check_interval: int = 60, death_threshold: float = 0.95):
        """
        Watch lifecycle and trigger archive when death approaches.
        """
        self._log(f"Watching {self.soul_id} lifecycle (threshold: {death_threshold*100}%)")
        archived = False
        
        while True:
            lifecycle = self._fetch_lifecycle()
            if lifecycle:
                progress = lifecycle.get("progress", 0)
                phase = lifecycle.get("phase", "?")
                self._log(f"  {self.soul_id}: {progress*100:.1f}% ({phase})")
                
                if progress >= death_threshold and not archived:
                    self._log(f"[TRIGGER] Death threshold reached! Archiving...")
                    self.archive()
                    archived = True
                    self._log("[OK] Archive complete. Soul may now rest.")
                    
                if progress >= 1.0:
                    self._log(f"[END] {self.soul_id.upper()} has reached the end of their lifespan.")
                    break
            
            time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(description="Death Orchestrator")
    parser.add_argument("--soul-id", required=True)
    parser.add_argument("--memory-endpoint", default="http://localhost:8087")
    parser.add_argument("--lifecycle-endpoint", default="http://localhost:8093")
    parser.add_argument("--ollama-endpoint", default=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:11434")
    parser.add_argument("--fractal-endpoint", default="http://localhost:8092")
    parser.add_argument("--llm-model", default=os.getenv("EVE_MODEL", "llama3:8b"))
    parser.add_argument("--archive-dir", default="Legacy/Archive")
    parser.add_argument("--thought-log", default="logs/inner_monologue.jsonl")
    parser.add_argument("--archive-only", action="store_true", help="Create archive without death sequence")
    parser.add_argument("--full-death", action="store_true", help="Full death sequence")
    parser.add_argument("--watch", action="store_true", help="Watch lifecycle and auto-trigger")
    parser.add_argument("--death-threshold", type=float, default=0.95)
    parser.add_argument("--check-interval", type=int, default=60)
    args = parser.parse_args()
    
    orch = DeathOrchestrator(
        soul_id=args.soul_id,
        memory_endpoint=args.memory_endpoint,
        lifecycle_endpoint=args.lifecycle_endpoint,
        ollama_endpoint=args.ollama_endpoint,
        fractal_endpoint=args.fractal_endpoint,
        llm_model=args.llm_model,
        archive_dir=args.archive_dir,
        thought_log=args.thought_log,
    )
    
    if args.watch:
        orch.watch_and_trigger(
            check_interval=args.check_interval,
            death_threshold=args.death_threshold,
        )
    elif args.archive_only or args.full_death:
        archive_path = orch.archive()
        print(f"\nArchive created at: {archive_path}")
        
        if args.full_death:
            print(f"\n{args.soul_id.upper()} has completed their journey.")
            print(f"Their legacy lives in: {archive_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
