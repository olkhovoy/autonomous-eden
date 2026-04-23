from __future__ import annotations

import json

import pytest

from umc_nn.multistream.coinglass_client import CoinGlassAPIError, CoinGlassClient


class _FakeResponse:
    def __init__(self, payload: dict, headers: dict[str, str] | None = None):
        self._payload = payload
        self.headers = headers or {}
        self.ok = True
        self.status_code = 200
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, *, params=None, headers=None, timeout=None):  # noqa: ANN001
        self.calls.append((url, params))
        return _FakeResponse(
            {"success": True, "data": [{"time": 1_700_000_000, "value": 1.5}]},
            headers={"API-KEY-MAX-LIMIT": "80", "API-KEY-USE-LIMIT": "3"},
        )


def test_coinglass_client_caches_raw_payloads_and_tracks_quota(tmp_path, monkeypatch):
    monkeypatch.setenv("COINGLASS_API_KEY", "test-key")
    session = _FakeSession()
    client = CoinGlassClient(cache_dir=tmp_path / "raw", session=session)

    first = client.fear_greed_history()
    second = client.fear_greed_history()

    assert first["payload"]["data"][0]["value"] == 1.5
    assert second["payload"]["data"][0]["value"] == 1.5
    assert len(session.calls) == 1
    assert client.last_quota_snapshot is not None
    assert client.last_quota_snapshot.max_limit == 80
    assert client.last_quota_snapshot.used_limit == 3
    assert client.last_quota_snapshot.remaining_limit == 77


class _ErrorSession:
    def get(self, url, *, params=None, headers=None, timeout=None):  # noqa: ANN001
        return _FakeResponse({"code": "400", "msg": "bad request"})


def test_coinglass_client_raises_on_api_error_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("COINGLASS_API_KEY", "test-key")
    client = CoinGlassClient(cache_dir=tmp_path / "raw", session=_ErrorSession())

    with pytest.raises(CoinGlassAPIError):
        client.fear_greed_history()
