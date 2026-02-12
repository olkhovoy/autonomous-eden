#!/usr/bin/env python3
"""
Telegram Eden Digest Bot
========================

Setup:
1. Create a bot via @BotFather and get TELEGRAM_BOT_TOKEN.
2. Add the bot to your channel and give it permission to post.
3. Export environment variables:
   - TELEGRAM_BOT_TOKEN (required)
   - TELEGRAM_CHANNEL_ID (required, e.g. @my_channel or -100...)
   - DIGEST_INTERVAL_HOURS (optional, default: 6)
   - THOUGHTS_LOG_ADAM (optional, default: logs/adam_thoughts.jsonl)
   - THOUGHTS_LOG_EVE (optional, default: logs/inner_monologue.jsonl)
   - EDEN_LOG (optional, default: logs/eden.jsonl)

Run:
    python tools/telegram_bot.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Make project-root imports reliable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.eden_digest import build_digest, load_jsonl
except Exception:
    from eden_digest import build_digest, load_jsonl


MAX_TELEGRAM_LEN = 4096
MARKDOWN_V2_RE = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [telegram_bot] {msg}", flush=True)


def escape_md_v2(text: str) -> str:
    return MARKDOWN_V2_RE.sub(r"\\\1", text)


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def chunk_message(text: str, limit: int = MAX_TELEGRAM_LEN) -> List[str]:
    if len(text) <= limit:
        return [text]

    parts: List[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(line) > limit:
            if current:
                parts.append(current.rstrip())
                current = ""
            for i in range(0, len(line), limit):
                parts.append(line[i : i + limit].rstrip())
            continue
        if len(current) + len(line) > limit:
            parts.append(current.rstrip())
            current = line
        else:
            current += line
    if current:
        parts.append(current.rstrip())
    return [p for p in parts if p]


def format_ts(ts: float) -> str:
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def count_jsonl(path: Path) -> Tuple[int, float]:
    total = 0
    last_ts = 0.0
    if not path.exists():
        return total, last_ts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            try:
                ts = float(rec.get("ts", 0))
                if ts > last_ts:
                    last_ts = ts
            except Exception:
                continue
    return total, last_ts


def tail_jsonl(path: Path, n: int = 5) -> List[Dict[str, Any]]:
    items: deque[Dict[str, Any]] = deque(maxlen=n)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(items)


def infer_phase_from_lifecycle(path: Path) -> str:
    data = load_json_file(path)
    if not data:
        return "UNKNOWN"
    total = int(data.get("total_tokens_seen", 0))
    max_tokens = int(data.get("max_lifespan_tokens", 10_000_000))
    if max_tokens <= 0:
        return "UNKNOWN"
    progress = total / max_tokens
    if progress < 0.2:
        return "GROWTH"
    if progress < 0.8:
        return "PEAK"
    return "DECAY"


class EdenTelegramBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.channel_id = os.getenv("TELEGRAM_CHANNEL_ID", "").strip()
        self.interval_hours = float(os.getenv("DIGEST_INTERVAL_HOURS", "6"))
        self.logs = {
            "adam": Path(os.getenv("THOUGHTS_LOG_ADAM", "logs/adam_thoughts.jsonl")),
            "eve": Path(os.getenv("THOUGHTS_LOG_EVE", "logs/inner_monologue.jsonl")),
        }
        self.eden_log = Path(os.getenv("EDEN_LOG", "logs/eden.jsonl"))
        self.archive_root = Path("Legacy/Archive")
        self.lifecycle_path = Path("data/lifecycle_state.json")
        self.max_digest_entries = 40

        if not self.token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
        if not self.channel_id:
            raise RuntimeError("TELEGRAM_CHANNEL_ID is required")

        self.api_base = f"https://api.telegram.org/bot{self.token}"
        self._update_offset = 0
        self._stop_event = threading.Event()
        self._timer_lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def _api_post(self, method: str, payload: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
        url = f"{self.api_base}/{method}"
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=timeout)
                if resp.status_code == 429:
                    retry_after = 1
                    try:
                        retry_after = int(resp.json().get("parameters", {}).get("retry_after", 1))
                    except Exception:
                        pass
                    wait = max(1, retry_after)
                    log(f"Rate limited on {method}; sleeping {wait}s")
                    time.sleep(wait)
                    continue
                if resp.status_code >= 400:
                    log(f"Telegram API error {resp.status_code} on {method}: {resp.text[:300]}")
                    return None
                data = resp.json()
                if not data.get("ok", False):
                    log(f"Telegram API returned not ok on {method}: {data}")
                    return None
                return data
            except requests.RequestException as exc:
                if attempt == 2:
                    log(f"Request failed on {method}: {exc}")
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    def _api_get(self, method: str, params: Dict[str, Any], timeout: int = 35) -> Optional[Dict[str, Any]]:
        url = f"{self.api_base}/{method}"
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=timeout)
                if resp.status_code >= 400:
                    log(f"Telegram API error {resp.status_code} on {method}: {resp.text[:300]}")
                    return None
                data = resp.json()
                if not data.get("ok", False):
                    log(f"Telegram API returned not ok on {method}: {data}")
                    return None
                return data
            except requests.RequestException as exc:
                if attempt == 2:
                    log(f"Request failed on {method}: {exc}")
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    def send_markdown(self, text: str, chat_id: Optional[str] = None) -> bool:
        target = chat_id or self.channel_id
        chunks = chunk_message(text, MAX_TELEGRAM_LEN)
        sent_any = False
        for chunk in chunks:
            payload = {
                "chat_id": target,
                "text": chunk,
                "parse_mode": "MarkdownV2",
                "disable_web_page_preview": True,
            }
            result = self._api_post("sendMessage", payload)
            if result:
                sent_any = True
            time.sleep(0.2)
        return sent_any

    def _build_digest(self, hours: Optional[float] = None) -> List[Dict[str, Any]]:
        h = self.interval_hours if hours is None else max(0.0, hours)
        since = time.time() - h * 3600

        thoughts: List[Dict[str, Any]] = []
        adam = load_jsonl(str(self.logs["adam"]), since)
        for item in adam:
            item.setdefault("soul_id", "adam")
        thoughts.extend(adam)

        eve = load_jsonl(str(self.logs["eve"]), since)
        for item in eve:
            item.setdefault("soul_id", "eve")
        thoughts.extend(eve)

        events = load_jsonl(str(self.eden_log), since)
        if not thoughts and not events:
            return []
        return build_digest(thoughts, events, max_entries=self.max_digest_entries)

    def _format_digest(self, digest: List[Dict[str, Any]], hours: float) -> str:
        if not digest:
            return "*Eden Digest*\n\n_No new thoughts or events in the selected window\\._"

        lines = [
            "*Eden Digest*",
            f"_Window: last {escape_md_v2(f'{hours:g}')} hours_",
            "",
        ]
        for entry in digest:
            ts = format_ts(float(entry.get("ts", 0)))
            soul = str(entry.get("soul_id", "unknown")).upper()
            text = normalize_text(entry.get("text", ""))
            if not text:
                continue
            safe_ts = escape_md_v2(ts)
            safe_soul = escape_md_v2(soul)
            safe_text = escape_md_v2(text)

            if entry.get("type") == "event":
                lines.append(f"{safe_ts} \\| *{safe_soul}*")
                lines.append(f"*{safe_text}*")
                lines.append("")
            else:
                lines.append(f"{safe_ts} \\| *{safe_soul}*")
                lines.append(f"> _{safe_text}_")
                lines.append("")
        return "\n".join(lines).strip()

    def publish_digest(self, reason: str = "manual", hours: Optional[float] = None) -> bool:
        h = self.interval_hours if hours is None else max(0.0, hours)
        digest = self._build_digest(hours=h)
        text = self._format_digest(digest, h)
        ok = self.send_markdown(text, chat_id=self.channel_id)
        if ok:
            log(f"Digest published ({reason}), entries={len(digest)}")
        return ok

    def _find_archive_dir(self, soul_id: str) -> Optional[Path]:
        exact = self.archive_root / f"{soul_id}_archive"
        if exact.exists():
            return exact
        if not self.archive_root.exists():
            return None
        for child in sorted(self.archive_root.iterdir()):
            if child.is_dir() and child.name.startswith(f"{soul_id}_"):
                return child
        return None

    def _agent_status(self, soul_id: str, thought_log: Path) -> Dict[str, Any]:
        total, last_ts = count_jsonl(thought_log)
        archive_dir = self._find_archive_dir(soul_id)
        status = "alive"
        phase = "UNKNOWN"

        if archive_dir and (archive_dir / "manifest.json").exists():
            status = "dead"
            manifest = load_json_file(archive_dir / "manifest.json") or {}
            lifecycle = manifest.get("lifecycle", {}) if isinstance(manifest, dict) else {}
            phase = str(
                manifest.get("death_phase")
                or lifecycle.get("phase")
                or "ARCHIVED"
            )
        elif soul_id == "adam":
            phase = "PARADISE"
        else:
            phase = infer_phase_from_lifecycle(self.lifecycle_path)

        return {
            "soul_id": soul_id,
            "status": status,
            "phase": phase,
            "thought_count": total,
            "last_ts": last_ts,
        }

    def render_status(self) -> str:
        lines = ["*Eden Agent Status*", ""]
        for soul_id, path in self.logs.items():
            st = self._agent_status(soul_id, path)
            lines.append(f"*{escape_md_v2(st['soul_id'].upper())}*")
            lines.append(f"status: {escape_md_v2(st['status'])}")
            lines.append(f"phase: {escape_md_v2(st['phase'])}")
            lines.append(f"thoughts: {escape_md_v2(str(st['thought_count']))}")
            lines.append(f"last\\_thought: {escape_md_v2(format_ts(st['last_ts']))}")
            lines.append("")
        return "\n".join(lines).strip()

    def render_latest(self) -> str:
        lines = ["*Latest Thoughts*", ""]
        for soul_id, path in self.logs.items():
            st = self._agent_status(soul_id, path)
            if st["status"] == "dead":
                continue
            lines.append(f"*{escape_md_v2(soul_id.upper())}*")
            rows = tail_jsonl(path, n=5)
            if not rows:
                lines.append("_No thoughts found\\._")
                lines.append("")
                continue
            for row in rows:
                ts = format_ts(float(row.get("ts", 0)))
                thought = normalize_text(str(row.get("thought", "")))[:500]
                lines.append(f"{escape_md_v2(ts)}")
                lines.append(f"> _{escape_md_v2(thought)}_")
                lines.append("")
        if len(lines) == 2:
            lines.append("_No active agents found\\._")
        return "\n".join(lines).strip()

    def render_testament(self, soul_id: str) -> str:
        archive_dir = self._find_archive_dir(soul_id)
        if not archive_dir:
            return f"*Testament*\n\n_Archive not found for {escape_md_v2(soul_id)}\\._"
        testament = archive_dir / "testament.txt"
        if not testament.exists():
            return f"*Testament*\n\n_Testament not found for {escape_md_v2(soul_id)}\\._"
        with testament.open("r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return f"*Testament*\n\n_Empty testament for {escape_md_v2(soul_id)}\\._"
        lines = [
            f"*Testament of {escape_md_v2(soul_id.upper())}*",
            f"_Archive: {escape_md_v2(str(archive_dir))}_",
            "",
        ]
        for para in text.splitlines():
            p = normalize_text(para)
            if p:
                lines.append(f"> _{escape_md_v2(p)}_")
                lines.append("")
        return "\n".join(lines).strip()

    def _ack(self, chat_id: str, text: str) -> None:
        self.send_markdown(escape_md_v2(text), chat_id=chat_id)

    def _handle_command(self, chat_id: str, text: str) -> None:
        parts = text.strip().split(maxsplit=1)
        command = parts[0].split("@", 1)[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command == "/digest":
            ok = self.publish_digest(reason="command", hours=self.interval_hours)
            self._ack(chat_id, "Digest posted to channel." if ok else "Digest publish failed.")
            return
        if command == "/status":
            self.send_markdown(self.render_status(), chat_id=chat_id)
            return
        if command == "/latest":
            self.send_markdown(self.render_latest(), chat_id=chat_id)
            return
        if command == "/testament":
            soul_id = arg.lower() if arg else "eve"
            self.send_markdown(self.render_testament(soul_id), chat_id=chat_id)
            return
        if command == "/start":
            self._ack(chat_id, "Commands: /digest, /status, /latest, /testament [soul_id]")
            return

    def _poll_once(self) -> None:
        params = {
            "timeout": 30,
            "offset": self._update_offset,
            "allowed_updates": json.dumps(["message"]),
        }
        data = self._api_get("getUpdates", params=params, timeout=35)
        if not data:
            return
        for upd in data.get("result", []):
            self._update_offset = max(self._update_offset, int(upd.get("update_id", 0)) + 1)
            msg = upd.get("message", {})
            text = msg.get("text", "").strip()
            if not text.startswith("/"):
                continue
            chat = msg.get("chat", {})
            chat_id = str(chat.get("id", ""))
            if not chat_id:
                continue
            try:
                self._handle_command(chat_id, text)
            except Exception as exc:
                log(f"Command handling failed ({text}): {exc}")
                self._ack(chat_id, "Command failed.")

    def _schedule_periodic(self) -> None:
        interval_seconds = max(60, int(self.interval_hours * 3600))

        def run_periodic() -> None:
            if self._stop_event.is_set():
                return
            try:
                self.publish_digest(reason="periodic", hours=self.interval_hours)
            except Exception as exc:
                log(f"Periodic digest failed: {exc}")
            finally:
                self._schedule_periodic()

        with self._timer_lock:
            if self._stop_event.is_set():
                return
            self._timer = threading.Timer(interval_seconds, run_periodic)
            self._timer.daemon = True
            self._timer.start()

    def run(self) -> None:
        log(f"Starting bot. Channel={self.channel_id}, interval={self.interval_hours}h")
        self._schedule_periodic()
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as exc:
                log(f"Polling loop error: {exc}")
                time.sleep(2)

    def stop(self) -> None:
        self._stop_event.set()
        with self._timer_lock:
            if self._timer:
                self._timer.cancel()
        log("Stopped")


def main() -> None:
    bot = EdenTelegramBot()
    try:
        bot.run()
    except KeyboardInterrupt:
        bot.stop()


if __name__ == "__main__":
    main()

