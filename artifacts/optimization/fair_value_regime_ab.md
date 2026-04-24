# Fair-value regime A/B evaluation

- Candidate: `gld_best_1x_sweep_20260419`
- Leverage: `1.0x`  ·  Risk cap: `50.0%`
- Rolling window: `12` months, step `3` months
- Generated from `scripts/evaluate_fair_value_regime.py`

## Per-window deltas (ON − OFF)

| Window | Bars | Return Δ | Sharpe Δ | Max DD Δ | Win-rate Δ | Trades Δ |
|---|---:|---:|---:|---:|---:|---:|
| 2024-04→2025-04 | 134,897 | -0.38% | +0.039 | +0.00% | -1.7 pp | -1 |
| 2024-07→2025-07 | 144,958 | -0.37% | +0.072 | -0.00% | -1.8 pp | -1 |
| 2024-10→2025-10 | 150,381 | +0.02% | +0.002 | +0.00% | +0.0 pp | +0 |
| 2025-01→2026-01 | 162,708 | +0.02% | +0.002 | +0.00% | +0.0 pp | +0 |
| FULL | 315,063 | -3.59% | -0.087 | +0.00% | -4.9 pp | -1 |

## Rolling-window summary

- Windows: **4**
- Return-positive windows: **2/4**  (50%)
- Sharpe-positive windows: **4/4**  (100%)
- Mean Δ return: **-0.18%**
- Mean Δ Sharpe: **+0.029**
- Mean Δ max-DD: **-0.00%**  (negative = improvement)
- Mean Δ trade-count: **-0.5**

## Full-range headline

- Window: `2024-04-04` → `2026-04-02`  (315,063 bars)
- OFF: return `+90.57%`  sharpe `3.829`  DD `-3.09%`  win `54.9%`  trades `51`
- ON : return `+86.98%`  sharpe `3.743`  DD `-3.09%`  win `50.0%`  trades `50`
- Δ : return **-3.59%**  sharpe **-0.087**  DD **+0.00%**  win **-4.9 pp**  trades **-1**