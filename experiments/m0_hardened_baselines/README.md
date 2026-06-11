# M0 — Hardened baselines (decision gate)

**Question.** Is CAN's advantage the architecture/objective, or just adversarial
training? V1 only league-trained CAN's own architecture; here the fair-MARL
objectives (GGF, FEN, SOTO) are wrapped in the *identical* league (PSRO) loop
and audited identically.

## Protocol

- Base graded-contention game, N=6, T=100, c ∈ {0.3, 0.5, 0.7, 0.9}, 5 seeds.
- League: 6 generations × (1200 cooperator updates vs the frozen defector pool
  + 1000 updates for a fresh best-response defector added to the pool); pool
  seeded with one random-init defector; D ~ unif{0..2} per episode at train
  time — byte-for-byte CAN's schedule (`fair_xattn.league_train`).
- Estimator parity: REINFORCE on per-step welfare increments (discounted
  reward-to-go), batch-mean baseline, entropy 0.05→0.003 annealed per
  generation — CAN's machinery, so a failure cannot be blamed on a weaker
  estimator. Only the welfare function differs per method (masked to
  cooperators; each reduces to its published v1 objective at D=0).
- Architecture: the baselines keep their v1 per-agent MLP (64-64) and the same
  6 behaviour tokens CAN sees — adversarial training is the only new variable.
- SOTO: β annealed 1→0 globally over the 6×1200 cooperator updates (anneal
  point 0.7, the v1 schedule lifted onto league alternation); the audited
  policy is the converged Team-Oriented net.
- Audit: v1 protocol — 1500 best-response defector updates vs the frozen team
  at a random index; plus a scripted always-claim ρ as the stop-trigger sanity
  check (`ac_rho`); D=0 efficiency/Jain and D≥1 efficiency reported.

## Reproduce

```bash
# full sweep (resumable; or run methods in parallel, one process each):
python -m can.fair_m0_hardened --methods GGF  --out results/m0_hardened_GGF.csv
python -m can.fair_m0_hardened --methods FEN  --out results/m0_hardened_FEN.csv
python -m can.fair_m0_hardened --methods SOTO --out results/m0_hardened_SOTO.csv
python -m can.fair_m0_hardened --methods CAN  --out results/m0_hardened_CAN.csv
python -m can.fair_m0_hardened --figs    # figure + gate table -> this directory
```

`CAN` retrains the v1 league verbatim and adds the always-claim audit column, so
the gate scores all methods on the same max(learned BR, always-claim)
exploitability; it doubles as the v1 reproduction check (fresh BR ρ within v1
pooled 95% CIs at every c).

## Decision gate

If any hardened baseline reaches D=0 eff ≥ 0.95 AND ρ ≤ 1.6 (CAN's corner),
the paper's claim is reframed to "adversarial training is the mechanism; CAN is
the most stable instance." If they stay exploitable or become wasteful, "fair
welfare objectives are structurally hard to harden" is promoted to a headline
claim.

## Verdict (2026-06-11, 80 league runs: 4 methods × 4 c × 5 seeds)

**Gate outcome: the negative branch — no hardened baseline reaches the good
corner; "fair welfare objectives are structurally hard to harden" is promoted
to a headline claim.** Full numbers in `results_table.md`; scatter in
`fig_m0_tradeoff.png`. Exploitability is scored as **max(learned-BR ρ,
always-claim ρ)** per seed, because the sweep exposed that the learned
best-response auditor is *deterrable* (it gets punished during its own training
and converges to soft claiming) while a committed always-claim script is not;
the v1 single-audit protocol under-measured exploitability at high c. Per-method:

- **FEN+PSRO: two-attractor collapse, 0/20 seeds in the corner.** At
  c ∈ {0.3,0.5,0.7} every seed (one exception) locks into all-contest — D=0
  efficiency exactly 1−c, ρ≈1. At c=0.9 seeds bifurcate: yield (eff ≈0.99 but
  ρ≈6.0 = a lone defector takes everything, after six league generations) or
  all-contest (eff 0.10). The league amplifies FEN's v1 instability into a
  commitment to one bad corner.
- **GGF+PSRO: efficient but never deters the committed claimer.** D=0 eff
  0.94–0.99 at all c, and the *learned* auditor is partially deterred (BR ρ
  1.5–2.3), but always-claim collects ρ = 1.6 / 2.0 / 2.9 / 4.8 as c rises.
  1/20 seeds in the corner.
- **SOTO+PSRO** (β annealed globally over the league; the schedule does survive
  league alternation): same signature as GGF — eff 0.90–0.99, learned-BR ρ
  1.4–1.8, but always-claim ρ 1.6 / 2.0 / 2.5 / 4.3. 2/20 seeds in the corner
  (both at c=0.3).
- **CAN (league, retrained with the dual audit): still the best at every c,
  and the only gate pass (c=0.3) — but the corrected audit revises v1's
  high-c claim.** Fresh BR ρ reproduces v1 within 95% CIs at every c
  (1.23/1.22/1.25/2.13 vs v1 pooled 1.33/1.36/1.51/1.20). Under max scoring,
  however: 1.28 / 1.72 / 2.29 / 3.93. CAN's robustness-to-a-committed-claimer
  decays with c — consistent with Prop. 1's leverage (1−c)/2 → 0, and a more
  honest empirical match to the theory than v1's flat ρ≈1.2–1.5.

**Mechanistic finding (feeds M2):** adversarial training produces *deterrence*,
not *immunity*. Probe (`can/fair_defector_probe.py`, c=0.5 and 0.9): the trained
BR defector converges to **end-game claiming** — claim rate ≈0/0/0/0.65 by
episode quarter — hiding while punishment has a future and harvesting at the
horizon; the always-claim script ignores punishment throughout. All league-trained teams (CAN included) punish an adaptive
defector into soft claiming, but at high c they do not sustain punishment
against a committed script — contesting costs the punisher (1−c)/2 → 0, so the
threat loses credibility exactly when it is most needed. The
deterrence-vs-commitment gap (always-claim ρ minus learned-BR ρ) grows
monotonically with c for every method.

**Consequences for the paper:** (i) claim reframed per the gate — adversarial
training alone does not rescue welfare-fair objectives; CAN's
objective+architecture is what makes hardening *work at all*, and only as far
as leverage allows; (ii) all audits must report the dual (learned BR +
always-claim) exploitability; v1's c=0.9 headline ρ for CAN is corrected
upward; (iii) the c→1 deterrence problem (M2) is now the program's central
question rather than a boundary curiosity.
