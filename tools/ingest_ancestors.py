#!/usr/bin/env python3
"""
Ingest ancestor archives into SoulMemory before launching a descendant.

Usage (from host, soul_memory must be running):
    python tools/ingest_ancestors.py --ancestor-ids eve,adam

Usage (inside docker network):
    python tools/ingest_ancestors.py \
        --ancestor-ids eve,adam \
        --memory-endpoint http://soul_memory:8087 \
        --archive-dir /app/Legacy/Archive
"""

import argparse
import os
import sys
import time

import requests

DEFAULT_MEMORY_ENDPOINT = os.getenv("MEMORY_ENDPOINT", "http://localhost:8087")
DEFAULT_ARCHIVE_DIR = os.getenv("ARCHIVE_DIR", "Legacy/Archive")


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] [ingest] {message}", flush=True)


def wait_for_service(endpoint: str, timeout: int = 120) -> bool:
    """Wait for soul_memory to become reachable."""
    log(f"Waiting for soul_memory at {endpoint} (timeout={timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{endpoint}/health", timeout=5)
            if resp.status_code == 200:
                log("[OK] soul_memory is ready.")
                return True
        except requests.ConnectionError:
            pass
        except Exception as exc:
            log(f"Probe error: {exc}")
        time.sleep(2)
    log("[FAIL] soul_memory did not become ready in time.")
    return False


def ingest_one(soul_id: str, archive_path: str, memory_endpoint: str) -> int:
    """Call /memories/ingest_archive for one ancestor. Returns count of ingested memories."""
    url = f"{memory_endpoint.rstrip('/')}/memories/ingest_archive"
    log(f"Ingesting archive for '{soul_id}' from '{archive_path}'...")

    try:
        resp = requests.post(
            url,
            json={"soul_id": soul_id, "archive_path": archive_path},
            timeout=600,  # large snapshot can take minutes to embed
        )
    except Exception as exc:
        log(f"[FAIL] Request error for '{soul_id}': {exc}")
        return 0

    if resp.status_code != 200:
        log(f"[FAIL] HTTP {resp.status_code} for '{soul_id}': {resp.text[:300]}")
        return 0

    data = resp.json()
    count = data.get("ingested", 0)
    log(f"[OK] Ingested {count} memories for '{soul_id}'.")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ancestor archives into SoulMemory")
    parser.add_argument(
        "--ancestor-ids",
        required=True,
        help="Comma-separated list of ancestor soul IDs (e.g. eve,adam)",
    )
    parser.add_argument(
        "--memory-endpoint",
        default=DEFAULT_MEMORY_ENDPOINT,
        help=f"SoulMemory HTTP endpoint (default: {DEFAULT_MEMORY_ENDPOINT})",
    )
    parser.add_argument(
        "--archive-dir",
        default=DEFAULT_ARCHIVE_DIR,
        help=f"Base archive directory (default: {DEFAULT_ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=120,
        help="Seconds to wait for soul_memory readiness (default: 120)",
    )
    args = parser.parse_args()

    ancestor_ids = [a.strip() for a in args.ancestor_ids.split(",") if a.strip()]
    if not ancestor_ids:
        log("[FAIL] No ancestor IDs provided.")
        sys.exit(1)

    log(f"Ancestors to ingest: {ancestor_ids}")
    log(f"Archive dir: {args.archive_dir}")
    log(f"Memory endpoint: {args.memory_endpoint}")

    if not wait_for_service(args.memory_endpoint, timeout=args.wait):
        sys.exit(1)

    total = 0
    for soul_id in ancestor_ids:
        archive_path = os.path.join(args.archive_dir, f"{soul_id}_archive")
        count = ingest_one(soul_id, archive_path, args.memory_endpoint)
        total += count

    log(f"Done. Total ingested: {total} memories across {len(ancestor_ids)} ancestors.")

    if total == 0:
        log("[WARN] No memories ingested. Check that archive files exist and contain valid data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
