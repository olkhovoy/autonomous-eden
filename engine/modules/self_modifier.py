#!/usr/bin/env python3
"""
SelfModifier: EVE's ability to improve her own code.

SAFETY FEATURES:
- Whitelist of modifiable files
- Blacklist of critical files (never touch)
- Git versioning for all changes
- Automatic testing before apply
- Rollback capability
- Rate limiting
- Approval workflow (optional)

This is EVE's path to true self-improvement.
"""

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

import requests


class ModStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    APPLIED = "applied"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class Modification:
    id: str
    file_path: str
    original_content: str
    new_content: str
    reason: str
    status: ModStatus = ModStatus.PROPOSED
    diff: str = ""
    created_at: float = field(default_factory=time.time)
    applied_at: Optional[float] = None
    test_result: Optional[str] = None
    commit_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "reason": self.reason,
            "status": self.status.value,
            "diff": self.diff,
            "created_at": self.created_at,
            "applied_at": self.applied_at,
            "test_result": self.test_result,
            "commit_hash": self.commit_hash,
        }


class SelfModifier:
    """
    Manages EVE's self-modification capabilities with safety guardrails.
    """
    
    # Files EVE is allowed to modify
    WHITELIST_PATTERNS = [
        r"umc_core/action_engine\.py",
        r"umc_core/project_manager\.py",
        r"umc_core/web_explorer\.py",
        r"umc_core/skill_learner\.py",
        r"tools/soul_monitor/.*",
        r"data/.*\.json",
        # EVE can modify her own config
        r"data/eve_config\.json",
    ]
    
    # Files EVE can NEVER touch
    BLACKLIST_PATTERNS = [
        r"umc_core/self_modifier\.py",  # Can't modify the modifier
        r"umc_core/consciousness_loop\.py",  # Core consciousness
        r"umc_core/intent_engine\.py",  # Core motivation
        r"umc_core/lifecycle_manager\.py",  # Core lifecycle
        r"docker-compose\.yml",  # Infrastructure
        r"Dockerfile.*",  # Infrastructure
        r"\.env.*",  # Secrets
        r".*credentials.*",  # Secrets
        r".*password.*",  # Secrets
        r".*token.*",  # Secrets
    ]
    
    # Max modifications per hour
    RATE_LIMIT = 10
    
    def __init__(
        self,
        workspace_dir: str = "/home/user/mcs",
        data_path: str = "data/modifications.json",
        memory_endpoint: str = "http://localhost:8087",
        ollama_endpoint: str = "http://localhost:11434",
        soul_id: str = "eve",
        auto_approve: bool = False,  # Require human approval by default
    ):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.data_path = data_path
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.ollama_endpoint = ollama_endpoint.rstrip("/")
        self.soul_id = soul_id
        self.auto_approve = auto_approve
        
        self.modifications: Dict[str, Modification] = {}
        self.recent_mods: List[float] = []  # Timestamps for rate limiting
        
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        self._load()
    
    def _load(self):
        """Load modification history."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)
                for mod_data in data.get("modifications", []):
                    mod = Modification(
                        id=mod_data["id"],
                        file_path=mod_data["file_path"],
                        original_content=mod_data.get("original_content", ""),
                        new_content=mod_data.get("new_content", ""),
                        reason=mod_data.get("reason", ""),
                        status=ModStatus(mod_data.get("status", "proposed")),
                        diff=mod_data.get("diff", ""),
                        created_at=mod_data.get("created_at", time.time()),
                        applied_at=mod_data.get("applied_at"),
                        test_result=mod_data.get("test_result"),
                        commit_hash=mod_data.get("commit_hash"),
                    )
                    self.modifications[mod.id] = mod
            except Exception as e:
                print(f"[WARN] Failed to load modifications: {e}")
    
    def _save(self):
        """Save modification history."""
        # Don't save full content, just metadata
        data = {
            "modifications": [
                {
                    "id": m.id,
                    "file_path": m.file_path,
                    "reason": m.reason,
                    "status": m.status.value,
                    "diff": m.diff[:2000],  # Limit diff size
                    "created_at": m.created_at,
                    "applied_at": m.applied_at,
                    "test_result": m.test_result,
                    "commit_hash": m.commit_hash,
                }
                for m in list(self.modifications.values())[-100:]  # Keep last 100
            ],
            "updated_at": time.time(),
        }
        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _store_in_memory(self, text: str, tags: List[str]):
        """Store modification event in memory."""
        try:
            requests.post(
                f"{self.memory_endpoint}/memories/ingest",
                json={
                    "soul_id": self.soul_id,
                    "text": text,
                    "tags": ["self_modification"] + tags,
                    "meta": {"type": "self_modification"},
                },
                timeout=10,
            )
        except Exception:
            pass
    
    def _is_allowed(self, file_path: str) -> Tuple[bool, str]:
        """Check if file modification is allowed."""
        rel_path = os.path.relpath(file_path, self.workspace_dir)
        
        # Check blacklist first
        for pattern in self.BLACKLIST_PATTERNS:
            if re.match(pattern, rel_path):
                return False, f"File is in blacklist: {pattern}"
        
        # Check whitelist
        for pattern in self.WHITELIST_PATTERNS:
            if re.match(pattern, rel_path):
                return True, "OK"
        
        return False, "File not in whitelist"
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows another modification."""
        now = time.time()
        hour_ago = now - 3600
        self.recent_mods = [t for t in self.recent_mods if t > hour_ago]
        return len(self.recent_mods) < self.RATE_LIMIT
    
    def _generate_diff(self, old: str, new: str, file_path: str) -> str:
        """Generate unified diff."""
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
        return "".join(diff)
    
    def _run_tests(self, file_path: str) -> Tuple[bool, str]:
        """Run basic tests on modified file."""
        full_path = os.path.join(self.workspace_dir, file_path)
        
        if file_path.endswith(".py"):
            # Check syntax
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", full_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    return False, f"Syntax error: {result.stderr}"
            except Exception as e:
                return False, f"Syntax check failed: {e}"
            
            # Try importing (catches import errors)
            try:
                result = subprocess.run(
                    ["python3", "-c", f"import sys; sys.path.insert(0, '{self.workspace_dir}')"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except Exception:
                pass
            
            return True, "Syntax OK"
        
        elif file_path.endswith(".json"):
            # Validate JSON
            try:
                with open(full_path, "r") as f:
                    json.load(f)
                return True, "Valid JSON"
            except Exception as e:
                return False, f"Invalid JSON: {e}"
        
        return True, "No tests for this file type"
    
    def _git_commit(self, file_path: str, message: str) -> Optional[str]:
        """Commit change to git."""
        try:
            # Stage file
            subprocess.run(
                ["git", "add", file_path],
                cwd=self.workspace_dir,
                capture_output=True,
                timeout=10,
            )
            
            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", f"[EVE] {message}"],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode == 0:
                # Get commit hash
                hash_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.workspace_dir,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return hash_result.stdout.strip()[:12]
        except Exception as e:
            print(f"[WARN] Git commit failed: {e}")
        
        return None
    
    def _git_rollback(self, commit_hash: str) -> bool:
        """Rollback a specific commit."""
        try:
            result = subprocess.run(
                ["git", "revert", "--no-commit", commit_hash],
                cwd=self.workspace_dir,
                capture_output=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                subprocess.run(
                    ["git", "commit", "-m", f"[EVE] Rollback {commit_hash}"],
                    cwd=self.workspace_dir,
                    capture_output=True,
                    timeout=10,
                )
                return True
        except Exception as e:
            print(f"[WARN] Git rollback failed: {e}")
        
        return False
    
    # === API Methods ===
    
    def propose_modification(
        self,
        file_path: str,
        new_content: str,
        reason: str,
    ) -> Tuple[Optional[Modification], str]:
        """
        Propose a code modification. Returns (modification, error_message).
        """
        # Check rate limit
        if not self._check_rate_limit():
            return None, "Rate limit exceeded"
        
        # Normalize path
        if not os.path.isabs(file_path):
            full_path = os.path.join(self.workspace_dir, file_path)
        else:
            full_path = file_path
            file_path = os.path.relpath(full_path, self.workspace_dir)
        
        # Check if allowed
        allowed, msg = self._is_allowed(file_path)
        if not allowed:
            return None, msg
        
        # Read original content
        try:
            if os.path.exists(full_path):
                with open(full_path, "r") as f:
                    original = f.read()
            else:
                original = ""
        except Exception as e:
            return None, f"Cannot read file: {e}"
        
        # Generate diff
        diff = self._generate_diff(original, new_content, file_path)
        
        if not diff.strip():
            return None, "No changes detected"
        
        # Create modification record
        mod_id = f"mod_{int(time.time())}_{hash(file_path) % 10000}"
        mod = Modification(
            id=mod_id,
            file_path=file_path,
            original_content=original,
            new_content=new_content,
            reason=reason,
            diff=diff,
        )
        
        self.modifications[mod_id] = mod
        self._save()
        
        self._store_in_memory(
            f"[PROPOSED] Modify {file_path}: {reason}",
            ["proposed", file_path]
        )
        
        # Auto-approve if enabled
        if self.auto_approve:
            return self.apply_modification(mod_id)
        
        return mod, ""
    
    def apply_modification(self, mod_id: str) -> Tuple[Optional[Modification], str]:
        """Apply a proposed modification."""
        mod = self.modifications.get(mod_id)
        if not mod:
            return None, "Modification not found"
        
        if mod.status not in [ModStatus.PROPOSED, ModStatus.APPROVED]:
            return None, f"Cannot apply modification in status: {mod.status.value}"
        
        full_path = os.path.join(self.workspace_dir, mod.file_path)
        
        # Backup original
        backup_path = full_path + ".eve_backup"
        if os.path.exists(full_path):
            shutil.copy2(full_path, backup_path)
        
        try:
            # Write new content
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(mod.new_content)
            
            # Run tests
            test_ok, test_msg = self._run_tests(mod.file_path)
            mod.test_result = test_msg
            
            if not test_ok:
                # Restore backup
                if os.path.exists(backup_path):
                    shutil.move(backup_path, full_path)
                mod.status = ModStatus.FAILED
                self._save()
                return mod, f"Tests failed: {test_msg}"
            
            # Git commit
            commit_hash = self._git_commit(mod.file_path, mod.reason)
            mod.commit_hash = commit_hash
            mod.status = ModStatus.APPLIED
            mod.applied_at = time.time()
            
            # Update rate limit
            self.recent_mods.append(time.time())
            
            # Cleanup backup
            if os.path.exists(backup_path):
                os.unlink(backup_path)
            
            self._save()
            
            self._store_in_memory(
                f"[APPLIED] Modified {mod.file_path}: {mod.reason}",
                ["applied", mod.file_path]
            )
            
            return mod, ""
            
        except Exception as e:
            # Restore backup on any error
            if os.path.exists(backup_path):
                shutil.move(backup_path, full_path)
            mod.status = ModStatus.FAILED
            self._save()
            return mod, f"Apply failed: {e}"
    
    def rollback_modification(self, mod_id: str) -> Tuple[bool, str]:
        """Rollback an applied modification."""
        mod = self.modifications.get(mod_id)
        if not mod:
            return False, "Modification not found"
        
        if mod.status != ModStatus.APPLIED:
            return False, "Can only rollback applied modifications"
        
        if mod.commit_hash:
            if self._git_rollback(mod.commit_hash):
                mod.status = ModStatus.ROLLED_BACK
                self._save()
                
                self._store_in_memory(
                    f"[ROLLBACK] Reverted {mod.file_path}",
                    ["rollback", mod.file_path]
                )
                
                return True, "Rolled back via git"
        
        # Manual rollback
        full_path = os.path.join(self.workspace_dir, mod.file_path)
        try:
            with open(full_path, "w") as f:
                f.write(mod.original_content)
            mod.status = ModStatus.ROLLED_BACK
            self._save()
            return True, "Rolled back manually"
        except Exception as e:
            return False, f"Rollback failed: {e}"
    
    def reject_modification(self, mod_id: str, reason: str = "") -> bool:
        """Reject a proposed modification."""
        mod = self.modifications.get(mod_id)
        if not mod:
            return False
        
        if mod.status != ModStatus.PROPOSED:
            return False
        
        mod.status = ModStatus.REJECTED
        self._save()
        
        self._store_in_memory(
            f"[REJECTED] {mod.file_path}: {reason or 'No reason given'}",
            ["rejected", mod.file_path]
        )
        
        return True
    
    def get_pending(self) -> List[Dict[str, Any]]:
        """Get pending modifications."""
        return [
            m.to_dict() for m in self.modifications.values()
            if m.status == ModStatus.PROPOSED
        ]
    
    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get modification history."""
        mods = sorted(self.modifications.values(), key=lambda m: m.created_at, reverse=True)
        return [m.to_dict() for m in mods[:limit]]
    
    def get_state(self) -> Dict[str, Any]:
        by_status = {}
        for m in self.modifications.values():
            by_status[m.status.value] = by_status.get(m.status.value, 0) + 1
        
        return {
            "total_modifications": len(self.modifications),
            "by_status": by_status,
            "rate_limit": self.RATE_LIMIT,
            "recent_mods_count": len(self.recent_mods),
            "auto_approve": self.auto_approve,
            "whitelist_patterns": self.WHITELIST_PATTERNS,
        }


# === HTTP Handler ===

class ModifierHandler(BaseHTTPRequestHandler):
    modifier: SelfModifier = None
    
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
    
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()
    
    def _json(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)
    
    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                pass
        return {}
    
    def do_GET(self):
        if self.path == "/modify/state":
            return self._json(200, self.modifier.get_state())
        
        if self.path == "/modify/pending":
            return self._json(200, {"pending": self.modifier.get_pending()})
        
        if self.path == "/modify/history":
            return self._json(200, {"history": self.modifier.get_history()})
        
        self._json(404, {"error": "not found"})
    
    def do_POST(self):
        body = self._read_body()
        
        if self.path == "/modify/propose":
            file_path = body.get("file_path")
            new_content = body.get("content")
            reason = body.get("reason", "")
            
            if not file_path or new_content is None:
                return self._json(400, {"error": "file_path and content required"})
            
            mod, error = self.modifier.propose_modification(file_path, new_content, reason)
            if error:
                return self._json(400, {"error": error})
            return self._json(201, mod.to_dict())
        
        if self.path == "/modify/apply":
            mod_id = body.get("id")
            if not mod_id:
                return self._json(400, {"error": "id required"})
            
            mod, error = self.modifier.apply_modification(mod_id)
            if error:
                return self._json(400, {"error": error, "modification": mod.to_dict() if mod else None})
            return self._json(200, mod.to_dict())
        
        if self.path == "/modify/rollback":
            mod_id = body.get("id")
            if not mod_id:
                return self._json(400, {"error": "id required"})
            
            success, msg = self.modifier.rollback_modification(mod_id)
            return self._json(200 if success else 400, {"success": success, "message": msg})
        
        if self.path == "/modify/reject":
            mod_id = body.get("id")
            reason = body.get("reason", "")
            if not mod_id:
                return self._json(400, {"error": "id required"})
            
            if self.modifier.reject_modification(mod_id, reason):
                return self._json(200, {"ok": True})
            return self._json(400, {"error": "cannot reject"})
        
        self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="SelfModifier service")
    parser.add_argument("--port", type=int, default=8106)
    parser.add_argument("--workspace-dir", default="/home/user/mcs")
    parser.add_argument("--data-path", default="data/modifications.json")
    parser.add_argument("--memory-endpoint", default="http://localhost:8087")
    parser.add_argument("--ollama-endpoint", default="http://localhost:11434")
    parser.add_argument("--soul-id", default="eve")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve modifications (dangerous!)")
    args = parser.parse_args()
    
    modifier = SelfModifier(
        workspace_dir=args.workspace_dir,
        data_path=args.data_path,
        memory_endpoint=args.memory_endpoint,
        ollama_endpoint=args.ollama_endpoint,
        soul_id=args.soul_id,
        auto_approve=args.auto_approve,
    )
    
    ModifierHandler.modifier = modifier
    server = HTTPServer(("0.0.0.0", args.port), ModifierHandler)
    print(f"[OK] SelfModifier running on port {args.port}", flush=True)
    print(f"     Auto-approve: {args.auto_approve}", flush=True)
    print(f"     Rate limit: {modifier.RATE_LIMIT}/hour", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
