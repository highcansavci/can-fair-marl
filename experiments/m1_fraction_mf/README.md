# M1 — Fraction-parameterized (mean-field) tokens

**Diagnosis** (audit: [notes/m1_n_dependence_audit.md](../../notes/m1_n_dependence_audit.md)):
the game is anonymous/aggregative — payoffs depend only on the claimer count m —
so the mean-field-correct policy responds to *fractions*. The v1 tokens carry
1/N scale drift (utility-rate and gap channels), so the N=6 policy sees
out-of-distribution inputs at larger N and transfer collapses at high c
(v1: always-claim transfer ρ at N=24 = 1.9 / 3.8 / 5.9 / 9.0 for c=0.3–0.9).

**One variable changed** vs the v1 transfer experiment (`fair_rerun` arm A:
vanilla `train()`, 2500 updates, dmax=2, same XAttn/hypers): the token
parameterization — 8 N-invariant channels (relative share clipped to [0,4],
z-scored deviations clipped to ±3 / [0,4], is-min, claim rate, claimed fraction
m/N, running mean claimed fraction, time). Implementation:
[can/fair_m1_fraction.py](../../can/fair_m1_fraction.py).

## Protocol

- Train at N=6 only; zero-shot audit at N ∈ {6, 12, 24, 48}; c ∈ {0.3..0.9};
  5 seeds; v1-token control arm under the identical protocol.
- Transfer metric: always-claim defector ρ (the v1 metric — and per M0's
  dual-audit finding, the committed adversary is the stronger one) + D=0
  efficiency at every N; trained-BR audit at N=6 (regression check).
- Probes on the saved params: `--attn` (attention-row entropy / ctx norm vs N —
  verify aggregation N-invariance, don't assume it) and `--mf` (numeric
  finite-N mean-field reference: optimal contest fraction q*(c, D, N) vs the
  policy's empirical cooperator claim rate). MFG caveat: the mean-field is the
  N→∞ limit; at finite N we expect tracking, not exact agreement.

## Reproduce

```bash
python -m can.fair_m1_fraction --arm frac     # -> results/m1_fraction.csv + params
python -m can.fair_m1_fraction --arm v1tok    # control arm
python -m can.fair_m1_fraction --figs         # transfer figure -> this directory
python -m can.fair_m1_fraction --attn         # attention N-invariance probe
python -m can.fair_m1_fraction --mf           # mean-field reference probe
```

## Success criteria (from the master plan)

1. ρ(N=24, c=0.9) < 2.5 (v1: ≈8.9), monotone improvement at N=12 and N=48.
2. No regression at N=6: BR ρ within v1 vanilla CIs, D=0 efficiency ≥ 0.97.
3. Empirical contest rate tracks the mean-field reference qualitatively.

## Verdict (2026-06-11, 40 training runs + probes)

**Success criteria: FAILED — and the probes located the real mechanism, which
token reparameterization cannot fix. The transfer fragility is a property of
the *lone-defector audit*, not of team size.**

Criterion scores (lone defector, always-claim, mean over 5 seeds):
1. ρ(N=24, c=0.9) = **9.16** (target < 2.5; v1 tokens 8.94). Fraction tokens
   help only at low/mid c (N=48: 2.99 vs 4.54 at c=0.3; 4.66 vs 8.45 at c=0.5;
   no improvement at c=0.7/0.9). **FAIL.**
2. N=6 regression: D=0 eff 0.98–0.99 ✓, but BR ρ at c=0.9 = 2.91 vs v1's 2.16
   (clipped features appear to cost fine discrimination at the trained size).
   **Marginal FAIL.**
3. Empirical contest rate vs mean-field reference: the finite-N numeric optimum
   is bang-bang (q*=0 at D=0, q*=1 at D≥1); the policy tracks it directionally
   (q_emp ≈ 0.02–0.06 at D=0; 0.5–0.84 at D≥1, all N). **PASS (qualitative).**

**What the probes found (the actual mechanism):**
- *Attention does not flatten.* Row-entropy sharpness relative to uniform is
  constant (~0.27) from N=6 to 48 for both arms — the audit's dilution
  hypothesis is wrong as stated. Verified, not assumed (`--attn`).
- *The failure is detection latency.* At N=48, c=0.9, cooperators are nearly
  silent for the first ~3 deciles of the episode (claim rate 0.03→0.11) before
  ramping to ~0.75 — identically in both arms. During that onset the
  always-claim defector is sole claimer and banks full, waste-free units;
  ~30 banked units at N=48 ≈ the entire measured ρ. A lone defector is a
  *vanishing fraction*: its mass in any aggregate channel and its weight in
  any convex (attention) pooling scales ~1/N, so evidence accumulates ~N times
  slower. ρ against D=1 is a *count* quantity; no fraction (mean-field)
  parameterization preserves it.
- *The redemption test confirms the mean-field story* (`--constfrac`): at
  constant defector fraction (D=N/6), zero-shot transfer is essentially perfect
  for **both** arms — v1 tokens give ρ = 2.49→1.54 and fraction tokens
  2.88→1.90 from N=6 (D=1) to N=48 (D=8) at c=0.9, flat or improving at every c.

**Reframing for the paper:** the v1 limitation "zero-shot transfer degrades at
high contention" should be restated as: *transfer at constant defector fraction
already works with the v1 policy; what degrades with N is the detection of a
single (measure-zero) defector, whose onset latency scales with N and whose
banked take during the silent window is waste-free at any c.* Crowd-scale
robustness against a lone defector therefore needs a mechanism that breaks the
1/N evidence scaling (e.g., count-preserving channels such as top-k/extreme
statistics, or per-agent suspicion tracking) — an *aggregation* change, not a
token-scale change. Deliberately not attempted here (one variable per
milestone).

