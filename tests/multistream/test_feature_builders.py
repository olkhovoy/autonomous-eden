from __future__ import annotations

from umc_nn.multistream.feature_builders import CoinGlassSlowContextConfig, build_coinglass_slow_context_frame


def test_feature_builders_derive_daily_slow_context_columns():
    config = CoinGlassSlowContextConfig()
    frames = build_coinglass_slow_context_frame(
        open_interest_payload={"payload": {"data": [{"time": "2025-05-01", "value": 100.0}, {"time": "2025-05-02", "value": 125.0}]}},
        funding_payload={"payload": {"data": [{"time": "2025-05-01", "value": 0.01}, {"time": "2025-05-02", "value": 0.03}]}},
        liquidation_payload={"payload": {"data": [{"time": "2025-05-01", "aggregated_long_liquidation_usd": 20.0, "aggregated_short_liquidation_usd": 10.0}, {"time": "2025-05-02", "aggregated_long_liquidation_usd": 5.0, "aggregated_short_liquidation_usd": 15.0}]}},
        long_short_ratio_payload={"payload": {"data": [{"time": "2025-05-01", "global_account_long_short_ratio": 1.2}, {"time": "2025-05-02", "global_account_long_short_ratio": 0.9}]}},
        fear_greed_payload={"payload": {"data": [{"time": "2025-05-01", "value": 40.0}, {"time": "2025-05-02", "value": 45.0}]}},
        stablecoin_market_cap_payload={"payload": {"data": {"time_list": ["2025-05-01", "2025-05-02"], "data_list": [{"USDT": 1000.0, "USDC": 20.0}, {"USDT": 1100.0, "USDC": 30.0}]}}},
        etf_payload={"payload": {"data": [{"date": "2025-05-01", "flow_usd": 50.0}, {"date": "2025-05-02", "flow_usd": -20.0}]}},
        config=config,
    )

    derivatives = frames["derivatives"].to_dicts()
    sentiment = frames["sentiment"].to_dicts()
    etf = frames["etf"].to_dicts()

    assert derivatives[1]["cg_derivatives_oi_delta"] == 25.0
    assert derivatives[0]["cg_derivatives_funding_rolling_mean"] == 0.01
    assert derivatives[1]["cg_derivatives_funding_rolling_mean"] == 0.02
    assert derivatives[0]["cg_derivatives_liquidation_imbalance"] > 0.0
    assert derivatives[1]["cg_derivatives_long_short_ratio"] == 0.9

    assert sentiment[1]["cg_sentiment_fear_greed_delta"] == 5.0
    assert sentiment[0]["cg_sentiment_stablecoin_market_cap_level"] == 1020.0
    assert sentiment[1]["cg_sentiment_stablecoin_market_cap_delta"] == 110.0

    assert etf[0]["cg_etf_btc_net_flow"] == 50.0
    assert etf[1]["cg_etf_btc_net_flow_rolling_5d"] == 30.0
