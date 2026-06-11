# M1 audit — N-dependence of CAN's tokens and attention context

Written **before** any M1 code change (M1 task 1). Audited code:
`can/fair_xattn.py::features` (the 6 token channels) and `::XAttn` (the
aggregation). Every claim below is about the *base graded-contention game*
(`graded_step`): one unit per step, so `sum_i u_i(t) <= t` always, with equality
unless a contested step wastes `c`.

## Channel-by-channel audit of `features(u, cc, t, T)`

| # | channel | formula | N-dependence | severity |
|---|---------|---------|--------------|----------|
| 1 | utility rate | `u_i / T` | per-capita utility ≈ `t/(N·T)` under fair play: the channel's scale shrinks ∝ 1/N. At N=6 a fair agent ends at ≈0.167; at N=24 at ≈0.042. The policy never saw values this small at train time. | **high** |
| 2 | dev from mean | `(u_i − ū)/T` | gaps between agents shrink roughly ∝ 1/N for the same *relative* inequity (a defector holding k× its fair share sits at `(k−1)·t/(N·T)`). The decision-relevant signal "agent j has 3× its share" maps to an N-dependent magnitude. | **high** |
| 3 | dev from min | `(u_i − u_min)/T` | same 1/N shrinkage as #2. | **high** |
| 4 | is-min flag | `1[u_i = u_min]` | per-token value is N-invariant, but the *fraction* of flagged tokens shifts: at small t ties are common (many zeros ⇒ many flags); at large N, post-transient exactly ~1 of N tokens is flagged, so the flag's share of the attention pool dilutes ∝ 1/N. | medium |
| 5 | claim rate | `cc_i / t` | per-agent rate, N-invariant per token. The *defector signature* (claim-rate ≈ 1 vs cooperators' ≈ 1/N turn-taking rate at D=0) actually *sharpens* with N — but the cooperator's own baseline rate 1/N shifts, so "how high is suspicious" is N-dependent. | medium |
| 6 | time | `t/T` | invariant. | none |

Aggregate features the policy *cannot* read off directly at unseen N:
- **claimed fraction `m_t/N`** — inferable only by averaging claim-rate tokens
  through attention; there is no explicit channel.
- **defector fraction estimate `d̂`** — likewise implicit.

## Attention-context audit of `XAttn`

`a = softmax(QK^T/√d)` rows are convex weights over N tokens, and `ctx = aV` is a
convex combination of values — so the *output scale* is N-invariant **only if the
attention distribution keeps the same shape**. Two N-dependent effects:

1. **Logit-scale dilution.** Q,K are linear in the tokens; channels 1–3 shrink
   ∝ 1/N, so attention logits flatten as N grows ⇒ rows tend toward uniform
   1/N weights ⇒ ctx tends toward the token *mean*, washing out the worst-off /
   defector tokens exactly when they matter. (To verify empirically in M1 step 2:
   compare attention-row entropy at N=6 vs 24 for a frozen v1 policy.)
2. **Pool composition.** Even at fixed logit sharpness, the one defector token is
   1/N of the pool; a uniform-ish row gives it weight ∝ 1/N. A *correct*
   mean-field readout (claimed fraction, mean claim rate) wants this; the
   *targeting* readout (who is the free-rider) does not. The single head must do
   both.

## Where this bites (v1 evidence)

- v1 zero-shot N=24, c=0.9: ρ = 8.8 (vs 1.2 at N=6). High c is where precise,
  *selective* contesting matters most — exactly where flattened attention and
  out-of-distribution channel scales hurt.
- The v1 mixed-N curriculum (`fair_curriculum.py`, trains over N∈{4,6,8,12})
  did **not** close the gap (negative result in the paper): seeing more sizes
  does not fix a parameterization whose feature scales drift with N; it just
  averages the drift.

## M1 reparameterization plan (consequences of the audit)

1. Replace raw-gap channels with N-invariant forms: `(u_i−ū)/std(u)`-style
   normalized deviations, and utility *shares* `u_i/(Σu + ε)` rescaled by N
   (i.e. `N·u_i/Σu`, = 1 under fairness at every N) instead of `u_i/T`.
2. Add explicit aggregate channels: claimed fraction `m_{t−1}/N` (previous step)
   and running defector-fraction estimate `d̂` from claim rates (e.g. fraction of
   agents with claim-rate above a threshold, or mean claim rate minus the
   turn-taking baseline `1/N`).
3. Keep claim rate (#5), is-min (#4), time (#6).
4. Verify (don't assume) ctx N-invariance after the change: attention-row entropy
   and ctx norms at N∈{6,12,24,48} for a frozen policy; check the 2-logit head's
   output distribution shift.
5. Success criteria and protocol per the master prompt (train N=6 only; audit
   zero-shot N∈{6,12,24,48}; ρ(N=24,c=0.9) < 2.5; no regression at N=6).

## Postscript — scorecard after running M1 (2026-06-11)

The reparameterization was implemented and swept (see
`experiments/m1_fraction_mf/`). Honest scoring of this audit's predictions:

- **Channel scale drift (high severity): real but second-order.** Fraction
  tokens halved lone-defector transfer ρ at c≤0.5 and changed nothing at
  c≥0.7. ρ(24, 0.9) stayed ≈9.2.
- **Attention-logit dilution (effect #1): WRONG.** Measured row-entropy
  sharpness is constant (~0.27 of uniform) from N=6 to 48, both token sets.
- **Pool composition (effect #2): RIGHT, and it is the whole story.** The lone
  defector's ~1/N weight in any convex aggregate produces *detection latency*
  that scales with N (cooperators silent for ~30 steps at N=48, c=0.9, in both
  arms). The banked sole-claims during the silent window are the transfer gap.
- **Decisive control:** at constant defector fraction (D=N/6), the *v1* policy
  already transfers (ρ 2.5→1.5 from N=6→48 at c=0.9). The fragility is the
  vanishing-fraction (count-vs-fraction) adversary, not team size.

Lesson: the next attack on crowd-scale transfer is an aggregation change
(count-preserving extreme statistics / per-agent suspicion memory), not a token
change.
