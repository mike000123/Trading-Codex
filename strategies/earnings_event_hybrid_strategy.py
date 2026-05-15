"""
Unified research-only earnings event hybrid.

This strategy combines the validated positive and negative earnings research
families behind one Backtester-facing entry:
  - positive off-hours earnings reaction days route into the positive
    overshoot hybrid
  - negative off-hours earnings reaction days route into the negative
    rebound/failure hybrid

The underlying positive/negative strategies stay available internally for
research scripts, but the UI should present this as the single earnings
strategy choice.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy
from strategies.earnings_negative_hybrid_strategy import EarningsNegativeHybridStrategy
from strategies.earnings_overshoot_hybrid_strategy import EarningsOvershootHybridStrategy


@register_strategy
class EarningsEventHybridStrategy(BaseStrategy):
    strategy_id = "earnings_event_hybrid"
    name = "Earnings Event Hybrid (Research)"
    description = (
        "Unified research-only event-day stock strategy. It auto-uses the local earnings-event table and "
        "routes known positive off-hours reactions into the positive overshoot-or-continuation family, and known negative "
        "off-hours reactions into the mirrored negative continuation/rebound/failure family."
    )

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params or {})

    def _family_params(self, family: str) -> dict[str, Any]:
        resolved = self.resolve_params()
        family_key = str(family).strip().lower()
        nested = resolved.get(family_key, {})
        nested_params = dict(nested) if isinstance(nested, dict) else {}
        prefix = f"{family_key}__"
        prefixed = {
            str(k)[len(prefix):]: v
            for k, v in resolved.items()
            if isinstance(k, str) and k.startswith(prefix)
        }
        merged = dict(nested_params)
        merged.update(prefixed)
        return merged

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        actions, meta = self.generate_signals_bulk(data, symbol)
        if not actions or not meta:
            return Signal(strategy_id=self.strategy_id, symbol=symbol, action=SignalAction.HOLD)
        last_action = actions[-1]
        last_meta = meta[-1] or {}
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=last_action if isinstance(last_action, SignalAction) else SignalAction(str(last_action)),
            suggested_tp=last_meta.get("suggested_tp"),
            suggested_sl=last_meta.get("suggested_sl"),
            metadata=dict(last_meta.get("metadata") or {}),
        )

    def default_params(self) -> dict[str, Any]:
        return {}

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str) -> tuple[list, list]:
        n = len(data)
        hold_actions = [SignalAction.HOLD] * n
        hold_meta = [{"suggested_tp": None, "suggested_sl": None, "metadata": {}} for _ in range(n)]
        if data.empty or "date" not in data.columns:
            return hold_actions, hold_meta

        positive = EarningsOvershootHybridStrategy(params=self._family_params("positive"))
        negative = EarningsNegativeHybridStrategy(params=self._family_params("negative"))

        pos_actions, pos_meta = positive.generate_signals_bulk(data, symbol)
        neg_actions, neg_meta = negative.generate_signals_bulk(data, symbol)

        merged_actions: list[SignalAction] = []
        merged_meta: list[dict[str, Any]] = []
        for idx in range(n):
            pa = pos_actions[idx] if idx < len(pos_actions) else SignalAction.HOLD
            na = neg_actions[idx] if idx < len(neg_actions) else SignalAction.HOLD
            pm = pos_meta[idx] if idx < len(pos_meta) else hold_meta[idx]
            nm = neg_meta[idx] if idx < len(neg_meta) else hold_meta[idx]

            if pa != SignalAction.HOLD and na != SignalAction.HOLD:
                chosen_action = pa
                chosen_meta = dict(pm or {})
                meta_block = dict(chosen_meta.get("metadata") or {})
                meta_block["earnings_event_hybrid_collision"] = True
                meta_block["earnings_event_hybrid_sources"] = ["positive", "negative"]
                chosen_meta["metadata"] = meta_block
            elif pa != SignalAction.HOLD:
                chosen_action = pa
                chosen_meta = dict(pm or {})
                meta_block = dict(chosen_meta.get("metadata") or {})
                meta_block["earnings_event_hybrid_family"] = "positive"
                chosen_meta["metadata"] = meta_block
            elif na != SignalAction.HOLD:
                chosen_action = na
                chosen_meta = dict(nm or {})
                meta_block = dict(chosen_meta.get("metadata") or {})
                meta_block["earnings_event_hybrid_family"] = "negative"
                chosen_meta["metadata"] = meta_block
            else:
                chosen_action = SignalAction.HOLD
                chosen_meta = hold_meta[idx]

            merged_actions.append(chosen_action)
            merged_meta.append(chosen_meta)
        return merged_actions, merged_meta
