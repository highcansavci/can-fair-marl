# M2 — c=1 deterrence: punishment, not surplus capture

**Diagnosis (from the master plan + M0):** at c=1 identification is not the
bottleneck — leverage is. Contesting returns (1−c)/2 = 0 to the contester, but
it is *costless* to cooperators (the destroyed unit was going to the defector)
and *maximally punitive* to the free-rider (1→0). M0 already showed league
training produces deterrence of adaptive defectors; M2 asks whether that
deterrence can be made to work where surplus capture is impossible — and
against which adversary classes it breaks. The payoff channel for successful
deterrence is the m=0 rule: a deterred defector stops claiming and units route
losslessly to the worst-off.

## Policies

- **CAN-v1** — v1 league (γ=0.99) trained at c ∈ {0.95, 1.0}: the control.
- **CAN-deter** — identical league with full-horizon credit γ=1.0 (the minimal
  reward-attribution intervention; architecture/tokens unchanged).
- **grim / tft-k** — scripted commitment brackets. Yield by default (at c=1
  all-yield is D=0-optimal: m=0 routes every unit to the worst-off); flag an
  agent when its *offensive* claim-rate (claims on non-punishment steps)
  exceeds τ=0.15; grim contests forever once anyone is flagged, tft-k contests
  while a flagged agent claimed within k=10 steps. Punishment-phase claims do
  not count toward flags, so punishers never flag each other. Known evasion
  cap: claiming just under τ concedes ρ ≤ N·τ = 0.9.

## Audits (per policy × c × 5 seeds)

(a) fresh best-response learner (1500 updates) — deterrence should bind;
(b) scripted always-claim — the Prop. 1 boundary (at c=1 a committed claimer
forces either ρ→N with the team yielding, or ρ→N with eff→0 if the team
levels down — report both ρ and D=1 eff);
(c) patient learner (3× budget) — does deterrence survive an adversary that
can wait out punishment? Cost axis: D=0 efficiency and false-positive contest
rate P(m≥2 | D=0).

## Success criteria

- CAN-deter vs learning defector at c=1: ρ ≤ 1.5 with D=0 eff ≥ 0.95.
- Clean three-way separation: deterrence works on (a), fails on (b) by design
  (report as boundary), and (c) decides whether commitment is credible or
  exploitable by patience.

## Reproduce

```bash
python -m can.fair_m2_deterrence --arm v1
python -m can.fair_m2_deterrence --arm deter
python -m can.fair_m2_deterrence --arm scripted
python -m can.fair_m2_deterrence --figs   # figure + cost table -> this directory
```

Framing note: this milestone is the bridge to the ASLI program — enforcement
via credible commitment where direct leverage is absent.

## Verdict (2026-06-11, 20 league runs + 20 scripted audits)

**Deterrence at c=1 is achievable — by commitment, not (yet) by learning. The
clean three-way separation was obtained; the learned policy misses the target
and the gap is precisely a credibility gap.** Table: `results_table.md`;
figure: `fig_m2_threeway.png`.

Criterion scores:
1. *CAN-deter vs learning defector at c=1: ρ ≤ 1.5, eff ≥ 0.95.* **FAIL** —
   ρ = 1.78 mean (1.30–2.32 per seed), eff 0.99 ✓. Full-horizon credit (γ=1)
   does help at c=0.95 (ρ 1.56 vs control 2.12) but not at the boundary
   (1.78 vs 1.84). The scripted brackets show the target was *reachable in
   this game*: grim and tft-10 hold both learner classes at **ρ = 1.00 exactly**
   with D=0 eff 1.000 and zero false positives.
2. *Three-way separation.* **ACHIEVED** — (a) learners are deterred (fully by
   scripted commitment, partially by learned policies); (b) always-claim hits
   the Prop. 1 boundary by design: ρ→N at c=1 for every policy, with D=1
   efficiency collapsing to 0.01–0.13 (levelling-down); (c) the patient
   (3×-budget) learner gains nothing anywhere — patient ≈ standard for all
   four policies. **Deterrence, once established, is not waited out.**

Findings beyond the criteria:

- **The credibility gap.** Learned (REINFORCE/league) punishment is
  probabilistic, not absorbing: the best-response learner finds the residual
  slack worth ρ ≈ 1.8 at c=1. Grim's threat is structurally absorbing, so the
  learner's optimum collapses to full cooperation (ρ = 1.00, the defector
  collecting its fair share through the m=0 routing).
- **The m=0 need-routing is load-bearing for deterrence.** It pays honest
  yielding a fair share, which (with τ = 0.15 < 1/N) makes sub-threshold
  evasion strictly worse than cooperation (cap ρ = N·τ = 0.9 < 1) — the
  trigger design closes its own evasion loophole.
- **Absolute vs relative deterrence at the boundary.** Against grim at c=1,
  always-claim "achieves" ρ = N but earns ~1 absolute unit versus ~16.7 for
  cooperating: committed defection is self-destructive, not profitable. The
  honest statement of the Prop. 1 boundary is "a committed defector can force
  levelling-down", not "a committed defector profits."
- Caveat for fairness of comparison: the scripted brackets encode the threat
  model (claim-rate flagging) by construction; they are decentralized policies
  over the same public signals CAN observes, but they are designed, not
  learned. They bracket what learned commitment must reach.

**Bridge to ASLI:** the result isolates *credible commitment* as the missing
capability — enforcement where direct leverage is absent. Learning an
absorbing trigger (or distilling the scripted one) is the natural M2 follow-up.
