from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MarketScope:
    market_scope_id: str
    venue_id: str
    symbol: str
    instrument_type: str
    contract_type: str
    quote_currency: str
    account_scope: str
    timeframe_family: str
    dataset_id: str
    revision: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedTimeSeriesRecord:
    source_id: str
    market_scope: str
    observed_at: int
    published_at: int
    available_at: int
    revision: str
    fields: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        row = {
            "source_id": self.source_id,
            "market_scope": self.market_scope,
            "observed_at": self.observed_at,
            "published_at": self.published_at,
            "available_at": self.available_at,
            "revision": self.revision,
        }
        row.update(self.fields)
        return row
