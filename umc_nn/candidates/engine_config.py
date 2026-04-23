from __future__ import annotations

from umc_nn.candidate_engines import EngineConfig


def candidate_engine_config(candidate) -> EngineConfig:
    manifest = candidate.manifest
    search = {} if manifest is None else dict(manifest.search_config or {})
    family = candidate.engine_family
    if family is None and manifest is not None:
        family = manifest.engine_family
    return EngineConfig(
        family=str(family or search.get("engine_family") or "umc"),
        hidden_dim=int(search.get("engine_hidden_dim", search.get("hidden_dim", 64))),
        alpha=float(search.get("engine_alpha", search.get("alpha", 0.5))),
        action_head_mode=str(search.get("action_head_mode", "argmax")),
        action_threshold=float(search.get("action_threshold", 0.55)),
    )
