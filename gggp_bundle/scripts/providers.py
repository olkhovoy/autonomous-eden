"""
gggp_bundle/scripts/providers.py

LLM/embedding provider router used by A1 pipeline scripts.

Why a router: the A1 pipeline has two LLM touch-points (paraphrase
generation in `build_corpus_v1.py` and sentence embeddings in
`embed_corpus.py`). Both are called from Python, not from Rust, so
Rust does not need to know which backend serves them. Adding a new
backend = adding a new `Provider` subclass. No script-level changes.

Design:
  * `Provider` is a minimal ABC with two methods: `embed(texts)` and
    `chat(prompt, ...)`. Everything else (auth, retries, streaming) is
    backend-internal.
  * `load_provider(config_path)` reads `config/providers.toml`, picks
    `primary`, constructs the matching subclass, returns it.
  * Stdlib-only for HTTP (urllib). `numpy` is only imported when the
    caller actually asks for `.embed()` -> np.ndarray, so scripts that
    only need `.chat()` can run without numpy installed.

Error handling:
  * Every HTTP error raises `ProviderError` with the backend name,
    endpoint, and the raw server response (truncated). Scripts are
    expected to let these bubble up -- silent fallbacks would violate
    the "errors must inform about the cause" rule.

Logging:
  * Minimal: one line per batch call to stderr with backend, model,
    wall-clock ms, and item count. Useful when diagnosing slow Ollama
    cold-loads.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tomllib
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Provider",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderError",
    "load_provider",
]


class ProviderError(RuntimeError):
    """Raised on any backend failure (network, auth, bad response)."""


@dataclass
class Provider(ABC):
    """Abstract LLM/embedding provider."""

    name: str
    embed_model: str
    embed_dim: int
    chat_model: str
    chat_seed: int
    chat_temperature: float
    request_timeout_s: int

    @abstractmethod
    def embed(self, texts: list[str]) -> "Any":
        """Return a 2-D numpy array shape (len(texts), embed_dim)."""

    @abstractmethod
    def chat(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Return the completion text for a single-turn prompt.

        seed and temperature are taken from the provider's config; they
        are locked for reproducibility inside a branch.
        """

    def _log(self, op: str, n: int, elapsed_ms: float) -> None:
        print(
            f"[provider:{self.name}] {op} n={n} model="
            f"{self.embed_model if op == 'embed' else self.chat_model} "
            f"elapsed_ms={elapsed_ms:.0f}",
            file=sys.stderr,
        )


def _http_post_json(
    url: str, payload: dict, timeout_s: int, extra_headers: dict | None = None
) -> dict:
    """POST JSON, return decoded JSON. Raises ProviderError on any issue."""
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise ProviderError(
            f"HTTP {exc.code} from {url}: {exc.reason}. "
            f"Body(head): {body}. "
            f"Hint: verify the model is loaded "
            f"(`curl {url.rsplit('/', 2)[0]}/api/tags`) "
            f"and the request schema matches the backend API version."
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderError(
            f"Network error reaching {url}: {exc.reason}. "
            f"Hint: is the service running? For Ollama: "
            f"`systemctl status ollama` or `ollama serve`."
        ) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"Non-JSON response from {url}: {raw[:200]!r}. "
            f"Hint: the endpoint may have returned an HTML error page; "
            f"check server logs."
        ) from exc


class OllamaProvider(Provider):
    def __init__(self, url: str, cfg: dict):
        super().__init__(
            name="ollama",
            embed_model=cfg["embed_model"],
            embed_dim=int(cfg["embed_dim"]),
            chat_model=cfg["chat_model"],
            chat_seed=int(cfg.get("chat_seed", 42)),
            chat_temperature=float(cfg.get("chat_temperature", 0.0)),
            request_timeout_s=int(cfg.get("request_timeout_s", 300)),
        )
        self.url = url.rstrip("/")

    def embed(self, texts: list[str]) -> "Any":
        import numpy as np  # local import keeps chat-only scripts numpy-free

        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        t0 = time.time()
        resp = _http_post_json(
            f"{self.url}/api/embed",
            {"model": self.embed_model, "input": texts},
            timeout_s=self.request_timeout_s,
        )
        elapsed_ms = (time.time() - t0) * 1000
        self._log("embed", len(texts), elapsed_ms)

        embeddings = resp.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ProviderError(
                f"Ollama returned {len(embeddings or [])} embeddings for "
                f"{len(texts)} inputs. Raw response keys: {list(resp.keys())}. "
                f"Hint: some older Ollama versions only support /api/embeddings "
                f"(single-prompt). Upgrade Ollama to >=0.1.46."
            )

        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.shape[1] != self.embed_dim:
            raise ProviderError(
                f"Ollama returned embeddings of dim {arr.shape[1]} "
                f"but config declares embed_dim={self.embed_dim}. "
                f"Hint: update config/providers.toml embed_dim to match the "
                f"actual model, or pick a different embed_model."
            )
        return arr

    def chat(self, prompt: str, *, max_tokens: int | None = None) -> str:
        payload: dict = {
            "model": self.chat_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "seed": self.chat_seed,
                "temperature": self.chat_temperature,
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = int(max_tokens)

        t0 = time.time()
        resp = _http_post_json(
            f"{self.url}/api/generate",
            payload,
            timeout_s=self.request_timeout_s,
        )
        elapsed_ms = (time.time() - t0) * 1000
        self._log("chat", 1, elapsed_ms)

        text = resp.get("response")
        if not isinstance(text, str):
            raise ProviderError(
                f"Ollama /api/generate did not return a 'response' string. "
                f"Keys: {list(resp.keys())}. "
                f"Hint: model may not support /api/generate (embedding-only "
                f"models will reject this). Check chat_model in config."
            )
        return text


class OpenAIProvider(Provider):
    """Stub. Not used in A1; kept so the router interface is non-degenerate."""

    def __init__(self, base_url: str, cfg: dict):
        super().__init__(
            name="openai",
            embed_model=cfg["embed_model"],
            embed_dim=int(cfg["embed_dim"]),
            chat_model=cfg["chat_model"],
            chat_seed=int(cfg.get("chat_seed", 42)),
            chat_temperature=float(cfg.get("chat_temperature", 0.0)),
            request_timeout_s=int(cfg.get("request_timeout_s", 120)),
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "OpenAI provider selected but OPENAI_API_KEY is not set. "
                "Hint: `export OPENAI_API_KEY=...` or switch primary back to "
                "'ollama' in config/providers.toml."
            )

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def embed(self, texts: list[str]) -> "Any":
        import numpy as np

        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        t0 = time.time()
        resp = _http_post_json(
            f"{self.base_url}/embeddings",
            {"model": self.embed_model, "input": texts},
            timeout_s=self.request_timeout_s,
            extra_headers=self._headers(),
        )
        elapsed_ms = (time.time() - t0) * 1000
        self._log("embed", len(texts), elapsed_ms)

        data = resp.get("data") or []
        if len(data) != len(texts):
            raise ProviderError(
                f"OpenAI returned {len(data)} embeddings for {len(texts)} "
                f"inputs. Full response head: {str(resp)[:300]}"
            )
        arr = np.asarray(
            [row["embedding"] for row in data], dtype=np.float32
        )
        if arr.shape[1] != self.embed_dim:
            raise ProviderError(
                f"OpenAI embed dim {arr.shape[1]} != config embed_dim "
                f"{self.embed_dim}. Update config/providers.toml."
            )
        return arr

    def chat(self, prompt: str, *, max_tokens: int | None = None) -> str:
        payload: dict = {
            "model": self.chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "seed": self.chat_seed,
            "temperature": self.chat_temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        t0 = time.time()
        resp = _http_post_json(
            f"{self.base_url}/chat/completions",
            payload,
            timeout_s=self.request_timeout_s,
            extra_headers=self._headers(),
        )
        elapsed_ms = (time.time() - t0) * 1000
        self._log("chat", 1, elapsed_ms)

        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Malformed OpenAI chat response: {str(resp)[:300]}"
            ) from exc


def load_provider(
    config_path: Path | str = "gggp_bundle/config/providers.toml",
) -> Provider:
    """Load provider config and instantiate the primary backend."""
    path = Path(config_path)
    if not path.is_file():
        raise ProviderError(
            f"Provider config not found: {path}. "
            f"Hint: run scripts from the repo root, or pass an absolute path."
        )

    with path.open("rb") as f:
        cfg = tomllib.load(f)

    try:
        primary = cfg["providers"]["primary"]
    except KeyError as exc:
        raise ProviderError(
            f"{path}: missing [providers].primary key. "
            f"Hint: set primary = \"ollama\" or \"openai\"."
        ) from exc

    section = cfg.get("providers", {}).get(primary)
    if not section:
        raise ProviderError(
            f"{path}: primary='{primary}' but section "
            f"[providers.{primary}] is missing."
        )

    if primary == "ollama":
        return OllamaProvider(url=section["url"], cfg=section)
    if primary == "openai":
        return OpenAIProvider(base_url=section["base_url"], cfg=section)

    raise ProviderError(
        f"{path}: unknown primary='{primary}'. "
        f"Supported: 'ollama', 'openai'."
    )


if __name__ == "__main__":
    # Smoke test: load provider, embed 2 strings, chat with 1 prompt.
    p = load_provider()
    print(f"provider={p.name} embed_model={p.embed_model} dim={p.embed_dim}")
    try:
        import numpy as np

        emb = p.embed(["sort integers", "extract dates from text"])
        print(f"embed shape={emb.shape} dtype={emb.dtype}")
    except ImportError:
        print("numpy not installed; skipping embed smoke")
    # NB: do not cap max_tokens on thinking models (Qwen3, DeepSeek-R1
    # etc.) -- the CoT can easily exceed 200 tokens before the actual
    # answer. Ollama strips the <think>...</think> block and returns
    # only the post-thinking answer in the `response` field.
    completion = p.chat("Reply with exactly one word: OK")
    print(f"chat response = {completion!r}")
