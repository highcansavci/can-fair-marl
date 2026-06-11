# M2c — Regime unification: one policy from graded contesting to commitment

**Goal:** a single set of weights that contests proportionally for surplus in
the graded regime (c < 1) *and* holds grim-level commitment at the
all-or-nothing boundary (c = 1) — unifying the program's two specialists
(M0's league CAN and M2b's distill8).

**Setup** ([fair_m2c_unified.py](../../can/fair_m2c_unified.py)): 9-channel
tokens = M2b's absorbing-flag channels + an explicit c channel; training c
sampled per episode from {0.3, 0.5, 0.7, 0.9, 0.95, 1.0}; multi-c league
(6 generations, BR defectors trained on the same c-mix). Three arms test how
commitment enters a multi-regime policy:

- **uni-rl** — multi-c league RL from scratch (control; M2b predicts no
  commitment at c=1).
- **uni-bcft** — grim distilled first (high-c episodes), then league
  fine-tuned across all c: does RL *retain* taught commitment?
- **uni-joint** — league RL with an auxiliary BC loss toward the grim teacher
  on high-c episodes in every update (guided learning, λ=1).

**Audit** (same weights at every c): c=1.0 full M2 suite (BR / always-claim /
3× patient + D=0 eff + FP); c ∈ {0.5, 0.9} BR + always-claim + D=0.
References: per-c specialists (M0 CAN league at 0.5/0.9; distill8 at 1.0).

## Success criteria

- c=0.5/0.9: D=0 eff ≥ 0.95 and BR ρ comparable to the per-c league CAN
  (≈1.2–1.4 / ≈2.1).
- c=1.0: BR and patient ρ ≤ 1.5 (aspiration: grim parity 1.0), D=0 eff ≥ 0.95,
  FP small.
- The interesting contrasts: uni-rl vs uni-bcft (is taught commitment
  *retained* under RL fine-tuning?) and uni-bcft vs uni-joint (is continual
  teaching needed?).

## Reproduce

```bash
python -m can.fair_m2c_unified --arm uni-rl
python -m can.fair_m2c_unified --arm uni-bcft
python -m can.fair_m2c_unified --arm uni-joint
python -m can.fair_m2c_unified --figs
```

## Verdict (2026-06-11, 15 multi-c league runs; table in `results_table.md`)

**uni-joint doesn't just unify the regimes — it beats the per-c specialists on
the dual audit at every c.** One set of weights:

| uni-joint | D=0 eff / FP | BR ρ | always-claim ρ | patient ρ |
|---|---|---|---|---|
| c=0.5 | 0.979 / 0.041 | 1.30 | **1.25** | — |
| c=0.9 | **1.000 / 0.000** | 1.19 | **1.46** | — |
| c=1.0 | **1.000 / 0.000** | **1.00** | 6.00 (boundary) | **1.00** |

- **Grim parity at the boundary on every seed** (br = patient = 1.00, eff
  1.000, FP 0.000) while the same weights contest proportionally at c<1.
- **It repairs M0's committed-defector weakness in the graded regime.** The
  per-c specialists conceded always-claim ρ ≈ 1.7 / 4.3 at c = 0.5 / 0.9;
  uni-joint holds the committed script to **1.25 / 1.46** — taught commitment
  transfers *down* the contention range, where contesting is cheap enough
  that the absorbing trigger simply executes. Only exactly at c=1 does the
  committed claimer retain ρ→N, and there it earns ~1 absolute unit
  (levelling-down, the irreducible Prop. 1 boundary).
- **Arm comparison settles the mechanism question:** uni-rl (RL only) is
  seed-unstable everywhere and never commits (c=1 br 2.67); uni-bcft (teach
  then train) erodes from parity to ≈1.63 — without reinforcement the trigger
  decays under entropy pressure; uni-joint (teach *while* training, λ=1 BC on
  high-c teacher episodes) pins parity and lets RL own the graded regime.
  Commitment must not only be taught (M2b) — it must be *maintained*.

**Headline for the next paper:** a single decentralized 9-channel policy
(absorbing public-recursion flags + c-conditioning), trained by league RL with
a scripted-teacher auxiliary loss, matches or beats every specialist in the
program on every audit at every contention level — deterrence of adaptive,
patient, and committed adversaries alike, with zero false-positive cost at
high c. The only fragility it does not address is crowd-scale lone-defector
transfer (M1c's structural floor), which is orthogonal.
