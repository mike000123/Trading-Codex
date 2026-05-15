# Fair-value regime A/B evaluation

- Candidate: `gld_best_1x_sweep_20260419`
- Leverage: `1.0x`  ·  Risk cap: `50.0%`
- Rolling window: `12` months, step `3` months
- Generated from `scripts/evaluate_fair_value_regime.py`

## Per-window deltas (ON − OFF)

| Window | Bars | Return Δ | Sharpe Δ | Max DD Δ | Win-rate Δ | Trades Δ |
|---|---:|---:|---:|---:|---:|---:|
| 2024-04→2025-04 | 134,897 | +0.00% | +0.000 | +0.00% | +0.0 pp | +0 |
| 2024-07→2025-07 | 144,958 | +0.00% | +0.000 | +0.00% | +0.0 pp | +0 |
| 2024-10→2025-10 | 150,381 | +0.00% | +0.000 | +0.00% | +0.0 pp | +0 |
| 2025-01→2026-01 | 162,708 | +0.11% | +0.009 | +0.00% | +0.0 pp | +0 |
| FULL | 315,063 | +2.85% | +0.043 | +0.00% | +1.4 pp | +1 |

## Rolling-window summary

- Windows: **4**
- Return-positive windows: **1/4**  (25%)
- Sharpe-positive windows: **1/4**  (25%)
- Mean Δ return: **+0.03%**
- Mean Δ Sharpe: **+0.002**
- Mean Δ max-DD: **+0.00%**  (negative = improvement)
- Mean Δ trade-count: **+0.0**

## Full-range headline

- Window: `2024-04-04` → `2026-04-02`  (315,063 bars)
- OFF: return `+78.41%`  sharpe `4.025`  DD `-3.26%`  win `52.9%`  trades `34`
- ON : return `+81.26%`  sharpe `4.068`  DD `-3.26%`  win `54.3%`  trades `35`
- Δ : return **+2.85%**  sharpe **+0.043**  DD **+0.00%**  win **+1.4 pp**  trades **+1**