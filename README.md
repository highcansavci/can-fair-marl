# CAN — Decentralized Robust Fairness via Cross-Attention

Code and paper for **"Learning to Contest: Decentralized Robust Fairness in
Cooperative MARL via Cross-Attention."**

Welfare-fair cooperative MARL teams (FEN, SOTO) are *exploitable*: a self-interested
agent free-rides on the surplus fair agents forgo to raise the worst-off. A
centralized need-based allocator fixes this but removes agents' control over
allocation. **CAN** asks whether a *decentralized* policy can stay fair under a
free-rider — and shows it can, as far as the contest *leverage* of the game allows.

- **Graded contention + Prop. 1.** When a contested resource still delivers a
  fraction `1-c` (waste `c`), a worst-off cooperator that contests a free-rider gets
  `(1-c)/2 > 0` for any `c < 1` — so contesting strictly dominates yielding and
  decentralized leverage exists. The all-or-nothing case is the boundary `c = 1`.
- **The hard part is inference.** The number of free-riders `D` is unknown and
  varies per episode, so a fixed rule loses (always-contest wastes `c` when nobody
  defects; always-yield collapses when someone does). A robust cooperator must infer
  `D` from observed behaviour and respond proportionally.
- **CAN.** A shared, permutation-equivariant cross-attention policy over per-agent
  *behaviour tokens* (utilities, claim-rates, who is worst-off), trained against an
  adversarial **league** (PSRO) and audited by a fresh best-response defector with a
  bounded free-ride metric `ρ ∈ [0, N]` (`ρ=1` fair, `ρ=N` a lone defector takes
  everything).

## Headline results (`N=6`, 5 seeds, 95% bootstrap CIs)

| | base | congestion | stakes |
|---|---|---|---|
| GGF (welfare) | (eff 1.00, ρ 6.0) | (1.00, 2.0) | (0.98, 5.8) |
| FEN | (0.81, 4.2) | (0.96, 2.1) | (0.70, 3.3) |
| SOTO | (0.40, 1.0) wasteful | (1.00, 2.0) | (0.47, 1.6) wasteful |
| **CAN (league)** | **(0.98, 1.35)** | **(1.00, 1.60)** | **(0.94, 2.32)** |

`(D=0 efficiency, best-response ρ)`, mean over `c ∈ {0.3,0.5,0.7,0.9}`. CAN keeps
efficiency `0.94–1.00` everywhere and is the least-exploitable *efficient* policy;
its robustness is **not bought with waste** — even with a defector present it
delivers `D≥1` efficiency `0.83–0.96` (vs the all-contest equilibrium's `1-c`, i.e.
`0.10` at `c=0.9`). Robustness holds **in proportion to leverage**: strong on base,
structurally capped on congestion, partial on stakes (degrades at high `c`), and
absent under a winner-take-all (Matthew) rule.

## Environments

All share `N=6` agents over `T=100` steps; each step a divisible resource is
available; an agent **claims** (self-interested) or **yields**.

- **base** — single resource, binary action; equal split `(1-c)/m` among `m≥2`
  claimers, empty → worst-off. (`can/fair_graded.py`)
- **congestion** — `M=3` parallel servers; claim one or yield; per-server equal
  split, empties → the neediest. Inference: *which* server each free-rider targets.
  Exploit ceiling `ρ = N/M = 2`. (`can/fair_congestion.py`)
- **stakes** — single resource of random value `v_t` (lumpy jackpots, `E[v]=1`);
  contest *selectively* on high-value steps. (`can/fair_stakes.py`)
- **matthew (boundary)** — among claimers the *richest* wins, so a poor contester
  captures nothing and Prop. 1 leverage vanishes; CAN is exploited here, marking the
  method's boundary. (`can/fair_matthew.py`)

## Install

Python 3.11 with JAX (GPU strongly recommended), Flax, Optax, NumPy, Matplotlib:

```bash
pip install "jax[cuda12]" flax optax numpy matplotlib
```

Development used JAX 0.10 on GPU (WSL). Each environment has a fast CPU self-test,
e.g. `JAX_PLATFORMS=cpu python -m can.fair_congestion --selftest`.

## Reproduce

Run from the repo root; results land in `results/`, figures in `paper/`.

```bash
# CAN: base game (vanilla + league + reliability + transfer), 5 seeds
python -m can.fair_rerun

# Welfare-fair baselines (GGF / FEN / SOTO)
python -m can.fair_baselines_graded                       # base
python -m can.fair_baselines_envs --env congestion        # 2nd env
python -m can.fair_baselines_envs --env stakes            # 3rd env

# CAN on the other environments (full league budget)
python -m can.fair_congestion --generations 6 --coop_iters 1200 --br_iters 1000
python -m can.fair_stakes     --generations 6 --coop_iters 1200 --br_iters 1000
python -m can.fair_matthew    --generations 6 --coop_iters 1200 --br_iters 1000

# Architecture ablation (mean-pool / deep-sets / GRU vs cross-attention)
python -m can.fair_arch_ablation                          # all four, gen5
python -m can.fair_arch_ablation --archs XAttn GRU --generations 6 \
    --coop_iters 1200 --br_iters 1000 --out results/arch_ablation_gen6.csv

# Stress / extras
python -m can.fair_dmax --dmax 4        # larger d_max + defector coalitions
python -m can.fair_curriculum           # mixed-N transfer (negative result)
python -m can.fair_d1eff                # D>=1 efficiency under a defector
python -m can.fair_pool                 # pool independent runs -> results/*_pooled.csv

# Rebuild all paper figures from the CSVs
python -m can.fair_paper2_figs
```

## Paper

The full paper (with appendix on the environments and metric) is in
[`paper/main.pdf`](paper/main.pdf); LaTeX source in `paper/`.

## Repository layout

```
can/        # the CAN policy, environments, training schemes, baselines, figures
paper/      # LaTeX source, figures, and built PDF
results/    # experiment outputs (CSV), regenerable by the commands above
```
