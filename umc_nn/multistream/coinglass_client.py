from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import requests


DEFAULT_BASE_URL = "https://open-api-v4.coinglass.com/api"
DEFAULT_CACHE_DIR = Path("cache") / "coinglass_raw"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_REQUESTS_PER_MINUTE = 75

AGGREGATED_OPEN_INTEREST_HISTORY_PATH = "/futures/open-interest/aggregated-history"
OI_WEIGHT_FUNDING_RATE_HISTORY_PATH = "/futures/funding-rate/oi-weight-history"
AGGREGATED_LIQUIDATION_HISTORY_PATH = "/futures/liquidation/aggregated-history"
GLOBAL_LONG_SHORT_ACCOUNT_RATIO_HISTORY_PATH = "/futures/global-long-short-account-ratio/history"
FEAR_GREED_HISTORY_PATH = "/index/fear-greed-history"
STABLECOIN_MARKET_CAP_HISTORY_PATH = "/index/stableCoin-marketCap-history"
BTC_ETF_FLOW_HISTORY_PATH = "/etf/bitcoin/flow-history"


class CoinGlassAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class CoinGlassQuotaSnapshot:
    max_limit: int | None
    used_limit: int | None
    remaining_limit: int | None


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_success_payload(payload: dict[str, Any]) -> bool:
    code = payload.get("code")
    if code in (None, 0, "0"):
        return True
    if payload.get("success") is True:
        return True
    return False


class CoinGlassClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_requests_per_minute: int = DEFAULT_MAX_REQUESTS_PER_MINUTE,
        session: requests.Session | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("COINGLASS_API_KEY")
        if not resolved_key:
            raise ValueError("COINGLASS_API_KEY is not set")

        self.api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.max_requests_per_minute = max_requests_per_minute
        self.session = session or requests.Session()
        self._request_times: deque[float] = deque()
        self._lock = threading.Lock()
        self.last_quota_snapshot: CoinGlassQuotaSnapshot | None = None

    def _cache_path(self, path: str, params: dict[str, Any] | None) -> Path:
        identity = {
            "path": path,
            "params": params or {},
        }
        digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._request_times and now - self._request_times[0] >= 60.0:
                self._request_times.popleft()

            if len(self._request_times) >= self.max_requests_per_minute:
                sleep_for = 60.0 - (now - self._request_times[0]) + 0.05
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._request_times and now - self._request_times[0] >= 60.0:
                    self._request_times.popleft()

            self._request_times.append(time.monotonic())

    @staticmethod
    def quota_snapshot_from_headers(headers: Mapping[str, Any]) -> CoinGlassQuotaSnapshot:
        normalized = {str(key).upper(): value for key, value in headers.items()}
        max_limit = _safe_int(normalized.get("API-KEY-MAX-LIMIT"))
        used_limit = _safe_int(normalized.get("API-KEY-USE-LIMIT"))
        remaining = None
        if max_limit is not None and used_limit is not None:
            remaining = max(max_limit - used_limit, 0)
        return CoinGlassQuotaSnapshot(max_limit=max_limit, used_limit=used_limit, remaining_limit=remaining)

    def request_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        cache_path = self._cache_path(path, params)
        if cache_path.exists() and not force_refresh:
            return json.loads(cache_path.read_text())

        self._throttle()
        response = self.session.get(
            f"{self.base_url}{path}",
            params=params,
            headers={
                "accept": "application/json",
                "CG-API-KEY": self.api_key,
            },
            timeout=self.timeout_seconds,
        )
        self.last_quota_snapshot = self.quota_snapshot_from_headers(response.headers)
        if not response.ok:
            raise CoinGlassAPIError(f"CoinGlass request failed: {response.status_code} {response.text[:500]}")

        payload = response.json()
        if not _is_success_payload(payload):
            raise CoinGlassAPIError(f"CoinGlass API returned an error payload: {response.text[:500]}")
        cache_path.write_text(
            json.dumps(
                {
                    "fetched_at_epoch": int(time.time()),
                    "path": path,
                    "params": params or {},
                    "quota_snapshot": {
                        "max_limit": self.last_quota_snapshot.max_limit if self.last_quota_snapshot else None,
                        "used_limit": self.last_quota_snapshot.used_limit if self.last_quota_snapshot else None,
                        "remaining_limit": self.last_quota_snapshot.remaining_limit if self.last_quota_snapshot else None,
                    },
                    "payload": payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return json.loads(cache_path.read_text())

    def aggregated_open_interest_history(
        self,
        *,
        symbol: str = "BTC",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        return self.request_json(
            AGGREGATED_OPEN_INTEREST_HISTORY_PATH,
            params={"symbol": symbol, "interval": interval},
            force_refresh=force_refresh,
        )

    def oi_weight_funding_rate_history(
        self,
        *,
        symbol: str = "BTC",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        return self.request_json(
            OI_WEIGHT_FUNDING_RATE_HISTORY_PATH,
            params={"symbol": symbol, "interval": interval},
            force_refresh=force_refresh,
        )

    def aggregated_liquidation_history(
        self,
        *,
        symbol: str = "BTC",
        exchange_list: str = "Bybit",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        return self.request_json(
            AGGREGATED_LIQUIDATION_HISTORY_PATH,
            params={"symbol": symbol, "exchange_list": exchange_list, "interval": interval},
            force_refresh=force_refresh,
        )

    def global_long_short_account_ratio_history(
        self,
        *,
        exchange: str = "Bybit",
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        return self.request_json(
            GLOBAL_LONG_SHORT_ACCOUNT_RATIO_HISTORY_PATH,
            params={"exchange": exchange, "symbol": symbol, "interval": interval},
            force_refresh=force_refresh,
        )

    def fear_greed_history(self, *, force_refresh: bool = False) -> dict[str, Any]:
        return self.request_json(FEAR_GREED_HISTORY_PATH, force_refresh=force_refresh)

    def stablecoin_market_cap_history(self, *, force_refresh: bool = False) -> dict[str, Any]:
        return self.request_json(STABLECOIN_MARKET_CAP_HISTORY_PATH, force_refresh=force_refresh)

    def btc_etf_history(self, *, force_refresh: bool = False) -> dict[str, Any]:
        return self.request_json(BTC_ETF_FLOW_HISTORY_PATH, force_refresh=force_refresh)
