# Fair-value regime A/B evaluation

- Candidate: `gld_best_1x_sweep_20260419`
- Leverage: `1.0x`  ·  Risk cap: `50.0%`
- Rolling window: `12` months, step `3` months
- Generated from `scripts/evaluate_fair_value_regime.py`

## Per-window deltas (ON − OFF)

| Window | Bars | Return Δ | Sharpe Δ | Max DD Δ | Win-rate Δ | Trades Δ |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 315,063 | -6.67% | -0.326 | +0.00% | -8.8 pp | +1 |

## Full-range headline

- Window: `2024-04-04` → `2026-04-02`  (315,063 bars)
- OFF: return `+91.23%`  sharpe `3.850`  DD `-3.09%`  win `56.9%`  trades `51`
- ON : return `+84.56%`  sharpe `3.524`  DD `-3.09%`  win `48.1%`  trades `52`
- Δ : return **-6.67%**  sharpe **-0.326**  DD **+0.00%**  win **-8.8 pp**  trades **+1**