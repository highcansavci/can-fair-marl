# M2b — Learned commitment: closing M2's credibility gap

**M2 found:** scripted grim/tft hold learning defectors at ρ=1.00 at c=1;
league-learned policies leave ρ≈1.8. **Refined hypothesis:** the v1 tokens
cannot *represent* an absorbing trigger — their only memory is the claim-rate
cc/t, which forgives by dilution (a defector that claims then stops watches
its own flag decay). Grim's bookkeeping is a deterministic recursion over
*public* claim history, so it can legally live in the feature extractor of a
decentralized policy: two extra channels (absorbing per-agent flag once the
offensive claim-rate exceeds τ=0.15, and the team trigger any(flag)).

Three arms separate **representation** from **optimization**, plus a
**determinism** probe ([fair_m2b_commitment.py](../../can/fair_m2b_commitment.py)):

| arm | tokens | training | tests |
|---|---|---|---|
| distill6 | v1 (6ch) | behaviour-clone grim | can 6ch represent absorption? (expect no) |
| distill8 | +flag/trig (8ch) | behaviour-clone grim | is representation sufficient? (expect ρ≈1) |
| league8 | +flag/trig (8ch) | M2 league (γ=0.99) | does RL *learn* commitment when representable? |
| --probe | 6ch | argmax-eval of saved M2 params | how much gap was sampling softness? |

BC teacher data mixes adversary types (none / always-claim / rate-p /
burst-then-stop) so the stop-and-wait exploit is in-distribution. Audits are
M2's verbatim (fresh BR 1500 / always-claim / patient 3× + D=0 eff and
false-positive rate); league8 also runs c=0.5 to verify the flag channels do
not break proportional contesting in the graded regime (one policy, both
regimes).

## Success criteria

- distill8 ≈ grim parity (ρ ≈ 1.0, D=0 eff ≥ 0.95) — representation proven.
- league8 at c=1: ρ ≤ 1.5 (M2's bar), aspiration grim parity; D=0 eff ≥ 0.95;
  no regression at c=0.5 vs M0's CAN league (eff ≥ 0.95, ρ comparable).

## Reproduce

```bash
python -m can.fair_m2b_commitment --arm distill8
python -m can.fair_m2b_commitment --arm distill6
python -m can.fair_m2b_commitment --arm league8
python -m can.fair_m2b_commitment --probe
python -m can.fair_m2b_commitment --figs
```

## Verdict (2026-06-11; 20 distill runs, 15 league runs, 10-checkpoint probe)

**The credibility gap is closed — by teaching, not by discovery. Commitment
must be distilled into the policy; league RL does not find it even when it is
representable.** Table: `results_table.md`; figure: `fig_m2b_gap.png`.

| c=1.0 | D=0 eff / FP | ρ learner | ρ patient |
|---|---|---|---|
| CAN-deter (M2, 6ch RL) | 0.991 / 0.009 | 1.78 | 1.79 |
| distill6 (BC, no flag) | 0.916 / **0.084** | 1.05 | 1.05 |
| **distill8 (BC + flags)** | **0.992 / 0.008** | **1.00** | **1.00** |
| league8 (RL + flags) | 0.987 / 0.013 | 2.16 | 2.20 |
| grim (script) | 1.000 / 0.000 | 1.00 | 1.00 |

1. **Determinism (probe): ruled out.** Argmax evaluation of the M2 checkpoints
   *worsens* exploitability (9/10 cells; three collapse to ρ≈6) — the
   stochastic policy's randomized contesting was itself a weak mixed threat.
2. **Representation (distill6 vs distill8): necessary, with a measurable
   price.** Without absorbing memory, behaviour-cloned grim still deters at
   T=100 (claim-rate dilution is slow within a 100-step horizon) but pays a
   permanent **7–10% false-positive contest tax at D=0** — paranoia as a
   substitute for memory. With the two absorbing-flag channels (a public
   deterministic recursion, decentralized-legal), distill8 reaches **exact
   grim parity**: ρ=1.00 vs fresh *and* patient learners, D=0 eff 0.99,
   FP 0.8%.
3. **Optimization (league8): RL fails even with representable absorption** —
   ρ=2.16 at c=1, no better than the 6ch league. The incentive diagnosis:
   the league's adversaries are *learned best responses*, which (M0) are
   themselves deterrable and converge to soft/end-game claiming; against
   such opponents an absorbing trigger earns no more training welfare than a
   probabilistic one, so the gradient toward commitment is ≈0. Commitment
   pays only against adversaries that test it — which deterrable training
   opponents never do. A chicken-and-egg that the teacher breaks.
4. **No graded-regime damage:** league8 at c=0.5 keeps eff 0.97 with ρ≈1.7 —
   the flag channels are compatible with proportional contesting. (distill8
   is a c≈1 specialist by construction; unifying it with the graded-regime
   policy — e.g. BC-init + league fine-tune, or a c-conditioned policy — is
   the natural next step.)

**Headline:** a decentralized learned policy can hold *both* adaptive and
patient defectors to exactly fair shares at the all-or-nothing boundary, at
0.8% false-positive cost — but the commitment that achieves this had to be
distilled from a scripted teacher; neither full-horizon credit (M2) nor
representable absorption (league8) made RL discover it.
