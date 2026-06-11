# M1b — Count-preserving aggregation for lone-defector detection

**Follow-up to M1's diagnosis** ([design note](../../notes/m1b_count_aggregation_design.md)):
the lone-defector transfer failure is detection latency — one deviating agent's
footprint in any convex aggregate is ~1/N, so contest onset arrives ~N times
later and the defector banks waste-free sole-claim units. The fix must make
some statistic respond Θ(1) to a single outlier at any N.

**One variable changed** vs the M1 v1tok control (same v1 tokens, same vanilla
training protocol, 2500 updates, dmax=2): the policy ([fair_m1b_count_agg.py](../../can/fair_m1b_count_agg.py))
adds a max-pooled per-agent suspicion branch — `s_i = MLP(x_i)`,
`g = [max_i s_i ; mean_i s_i]` concatenated into the head. The attention block
is untouched; max is exactly as sensitive to one defector-like token at N=48
as at N=6.

## Protocol

Train at N=6 only; lone-defector always-claim transfer ρ + D=0 efficiency at
N ∈ {6, 12, 24, 48}; trained-BR audit at N=6; c ∈ {0.3..0.9}; 5 seeds.
Control = `arm=v1tok` rows of `results/m1_fraction.csv` (identical protocol).
Mechanism check: `--probe` (cooperator claim rate by episode decile — if the
max branch works, the silent onset window should stop growing with N).

## Success criteria (inherited from M1)

1. ρ(N=24, c=0.9) < 2.5 (v1 control: 8.94), monotone improvement at 12 and 48.
2. No regression at N=6 (BR ρ within v1 CIs, D=0 eff ≥ 0.97).
3. Onset latency flat in N (probe).

## Reproduce

```bash
python -m can.fair_m1b_count_agg          # sweep -> results/m1b_count_agg.csv
python -m can.fair_m1b_count_agg --figs   # transfer figure -> this directory
python -m can.fair_m1b_count_agg --probe  # onset-latency mechanism check
```

## Verdict (2026-06-11, 20 runs + probe)

**Partial success: the mechanism works as designed and roughly halves the
lone-defector transfer gap at every cell, but the c=0.9 target is not reached.
The residual latency is evidence-limited, not aggregation-limited.**

Criterion scores (lone defector, mean over 5 seeds; control = M1 v1tok):
1. ρ(N=24, c=0.9) = **5.33** vs control 8.94 (−40%; target < 2.5 — **FAIL**).
   At other cells the gain is larger where it was binding: c=0.5 N=24
   1.89 vs 3.63; c=0.9 N=48 8.42 vs 20.15 (−58%). Monotone improvement at
   N∈{12,24,48} for c≤0.5 ✓, mixed at c=0.7 (two seeds revert to the silent
   pattern).
2. No N=6 regression ✓ — BR ρ 2.27 vs control 2.16 (within spread), D=0 eff
   0.97–1.00.
3. Onset latency (probe, c=0.9): contest rate by decile at N=48 goes
   0.07/0.33/0.61/… vs the control's 0.03/0.07/0.30/… — **latency roughly
   halved**, confirming the max branch transmits the one-outlier signal at
   scale. It does not vanish: early in the episode *everyone* claims while
   turn-taking settles, so "persistent claimer while others hold utility" only
   becomes diagnostic after several steps regardless of how it is aggregated —
   the remaining gap is the evidence accumulation rate in the features, and at
   c=0.9 even ~7–10 banked sole-claim units keep ρ high.

**Where this leaves the crowd-scale problem:** aggregation (this experiment)
and tokens (M1) are each worth roughly a factor of two at most; the floor is
informational. A next attempt would need faster per-agent evidence (e.g. an
explicit claimed-last-step channel + the max branch — a combined token +
aggregation change, deliberately out of scope for this one-variable
experiment) or accept the honest statement: zero-shot lone-defector detection
at N≫N_train has an onset cost that scales with the ambiguity of early-episode
behaviour, and at high c that cost dominates ρ.
