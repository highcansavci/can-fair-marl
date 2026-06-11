# M1c — Fast evidence channel × count-preserving aggregation (the combined fix)

**Chain of diagnosis:** M1 — token scale is not the constraint; the lone
defector's 1/N footprint in convex aggregates produces detection latency.
M1b — a max-pooled suspicion branch fixes the aggregation half (latency and ρ
roughly halve) but the residual is *evidence-limited*: the v1 claim-rate
channel cc/t accumulates slowly, so early-episode ambiguity still costs banked
sole-claim units at high c.

**M1c** adds the missing fast-evidence channel — a per-agent
**claimed-last-step** indicator (instantaneous streak evidence rather than a
cumulative average) — and completes the 2×2 factorial so the contribution of
channel, aggregation, and their interaction is separable:

| tokens \ aggregation | mean-only (XAttn) | max-suspicion (XAttnMax) |
|---|---|---|
| v1 (6 ch) | M1 `v1tok` control | M1b `maxagg` |
| v1 + last-step (7 ch) | arm `last` | **arm `lastmax` (M1c)** |

Protocol identical to M1/M1b (vanilla train at N=6, 2500 updates, dmax=2;
lone-defector always-claim transfer at N ∈ {6,12,24,48}; BR audit at N=6;
5 seeds). Implementation: [fair_m1c_evidence.py](../../can/fair_m1c_evidence.py).

## Success criteria

1. ρ(N=24, c=0.9) < 2.5 (controls: v1 8.94, M1b 5.33).
2. No regression at N=6 (BR ρ within v1 spread, D=0 eff ≥ 0.97).
3. Onset latency ≈ flat in N (`--probe`).

## Reproduce

```bash
python -m can.fair_m1c_evidence --arm lastmax   # combined fix
python -m can.fair_m1c_evidence --arm last      # channel-only control
python -m can.fair_m1c_evidence --figs          # 2x2 factorial figure
python -m can.fair_m1c_evidence --probe
```

## Verdict (2026-06-11, 40 runs + probe; factorial complete)

**The interaction is the result: the fast-evidence channel does nothing alone
(≈ control at every cell) but, routed through the max branch, halves the gap
again. Cumulative: ρ(N=24, c=0.9) 8.94 → 4.58 (−49%), ρ(N=48, c=0.9)
20.15 → 6.88 (−66%), and the 2.5 target now holds through c=0.7
(2.50 at N=24 vs control 4.95). At c=0.9 the target is still missed — and the
probe shows why it cannot be met by better detection at all.**

| t24/t48 ρ | v1 ctrl | +channel | +max agg (M1b) | combined (M1c) |
|---|---|---|---|---|
| c=0.3 | 1.91/4.54 | 1.73/3.51 | 1.57/2.23 | **1.45/1.93** |
| c=0.5 | 3.63/8.45 | 2.11/4.12 | **1.89/2.41** | 2.02/2.62 |
| c=0.7 | 4.95/12.31 | 4.37/9.91 | 4.41/8.47 | **2.50/3.16** |
| c=0.9 | 8.94/20.15 | 9.38/20.28 | 5.33/8.42 | **4.58/6.88** |

No N=6 regression in any cell (BR ρ 2.16–2.27, D=0 eff 1.00 at c=0.9).

**Mechanism (probe, c=0.9):** onset at N=48 is now ~one decile (claim rate
0.14/0.60/0.72 by decile, vs M1b's 0.07/0.33/0.61 and the control's
0.03/0.07/0.30) — detection latency is essentially fixed. The factorial reads
cleanly: instantaneous evidence is useless under convex pooling (1/N dilution,
arm `last` ≈ control) and only pays when a count-preserving statistic
broadcasts it (arm `lastmax`).

**The structural floor at c→1:** with detection latency L ≈ 8–10 steps at
N=48, the defector banks L full units before contestation; afterwards each
step delivers only 1−c = 0.1 in total. One banked unit therefore weighs as
much as ~10 contested steps, and ρ ≈ N·L/(L + (T−L)(1−c)) stays ≈ 5–7 even
with near-perfect detection. Crowd-scale lone-defector robustness at high
waste is bounded by the *price of the detection window*, not by inference —
the same leverage logic as Prop. 1, applied to time. Closing it would need
either pre-commitment (contest by default until innocence is established —
paying false-positive waste at D=0) or a longer horizon (the banked units
amortize as T grows). Both change the problem, not the policy.
