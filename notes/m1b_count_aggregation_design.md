# Design note — count-preserving aggregation for lone-defector detection (M1b)

Status: design only, written while M2 runs. Do not implement before M2 closes
(one variable at a time).

## The constraint discovered in M1

Lone-defector transfer fails because every aggregate CAN can form is a convex
combination over N tokens: one agent's footprint is ~1/N, so detection latency
grows ~N and the defector banks ~latency full units before contestation starts
(experiments/m1_fraction_mf/). Token changes cannot fix this; the statistic
itself must not dilute. Requirement: a channel whose response to ONE deviating
agent is Θ(1) in N.

## Candidate A — extreme-statistic context (minimal, preferred)

Augment the attention context with max-pooled (and optionally min-pooled)
per-channel statistics over a learned per-agent "suspicion" embedding:

    s_i = MLP(x_i)                     # (N, d_s) per-agent suspicion features
    g   = [max_i s_i ; mean_i s_i]     # (2 d_s,) broadcast to all agents
    logits_i = head([x_i ; ctx_i ; g])

max_i is exactly as sensitive to one outlier at N=48 as at N=6 (response Θ(1));
mean is kept so D=0 statistics remain available. Cost: one small MLP + concat;
permutation-invariant; N-agnostic. The natural learned content of s_i is
"how defector-like is agent i" (claim-rate ≈ 1, share ≫ 1), and max-pooling
broadcasts the *worst* agent's signature to everyone immediately at any N.

## Candidate B — top-k attention sharpening

Replace softmax rows with sparse top-k attention (k fixed, e.g. 2): each agent
attends to at most k others, so a salient defector token receives Θ(1) weight
regardless of N. Riskier: changes the D=0 turn-taking computation too, and k
introduces a new hyperparameter; harder to attribute the effect.

## Candidate C — per-agent suspicion memory (recurrent)

GRU per agent over its own token history; detection integrates evidence at a
rate independent of N. Most expressive, most expensive, and the v1 ablation
already showed GRU aggregation is erratic at high c — avoid as a first move.

## Decision

Implement A first (single new variable: the pooled-statistic branch; attention
block untouched). Protocol: identical to M1 (vanilla train at N=6, 2500
updates, dmax=2; zero-shot lone-defector audit at N∈{6,12,24,48}; v1 tokens —
the M1 result says tokens are not the binding constraint; keep them to isolate
the aggregation variable). Success bar inherited from M1:
ρ(N=24, c=0.9) < 2.5, no N=6 regression. Add the onset-latency probe (claim
rate by decile) as the mechanism check: if A works, the silent window should
be flat in N.

If A succeeds at the vanilla budget, graduate it to the league protocol and
re-run the M0 dual audit at N=6 to confirm no robustness regression, then fold
into the paper as the crowd-scale fix. If A fails, B is the fallback; C only
with strong cause.
