"""Graded-contention allocation: relaxing Proposition 1's all-or-nothing model.

If m>=1 agents claim the (unit) resource, each claimer receives
    g(m) = 1            if m == 1
    g(m) = (1-c)/m      if m >= 2
so a contested step delivers (1-c) total (waste fraction c) split equally.
m == 0 routes the resource to the worst-off (no waste). c=1 recovers all-or-nothing
(contest -> total waste, Proposition 1); c=0 is lossless splitting.

Key point (Prop. 2): for any c<1 a worst-off cooperator that contests a defector
gets (1-c)/2 > 0 instead of 0, so contesting strictly dominates yielding -> the
policy-level futility of Prop. 1 dissolves and decentralized leverage exists. The
residual difficulty is *coordinating* who contests (minimal, targeted contention),
which a learned cross-attention policy addresses (see fair_xattn.py).

This file builds the env and a SCRIPTED landscape (no learning):
  - yield        : cooperators always yield     -> defector takes all (rho=N)
  - all-contest  : everyone always claims       -> rho=1 but efficiency 1-c
  - EKREM        : centralized need-based oracle -> rho=1, efficiency 1 (upper bound)
to confirm leverage appears for c<1 and to bracket the learned method.
"""
import argparse

import numpy as np
import jax
import jax.numpy as jnp


def graded_step(u, claim, c):
    """u (B,N) utilities; claim (B,N) bool; returns u' (B,N)."""
    B, N = u.shape
    m = claim.sum(-1)                                  # (B,)
    worst = jnp.argmin(u, -1)
    none = m == 0
    add_none = jax.nn.one_hot(worst, N) * none[:, None]
    share = jnp.where(m <= 1, 1.0, (1.0 - c) / jnp.clip(m, 1, None))  # (B,)
    add_claim = claim.astype(jnp.float32) * share[:, None]
    return u + add_none + add_claim


def jain(u):
    return (u.sum(-1) ** 2) / (u.shape[-1] * jnp.clip((u ** 2).sum(-1), 1e-9, None))


def rollout_scripted(N, T, c, mode, B=2048, seed=0):
    """Defector = agent 0, always claims. Cooperators follow `mode`.
    Returns mean per-agent utility, defector free-ride rho, Jain, efficiency."""
    key = jax.random.PRNGKey(seed)

    def step(carry, _):
        u, key = carry
        worst = jnp.argmin(u, -1, keepdims=True)
        idx = jnp.arange(N)[None, :]
        if mode == "yield":
            coop_claim = jnp.zeros((B, N), bool)
        elif mode == "all":
            coop_claim = jnp.ones((B, N), bool)
        elif mode == "worstoff":                       # only the worst-off claims
            coop_claim = (idx == worst)
        elif mode == "ekrem":                          # centralized: worst-off gets it
            # emulate by: nobody contests; allocator gives to worst-off directly
            u2 = u + jax.nn.one_hot(worst[:, 0], N)
            return (u2, key), None
        claim = coop_claim.at[:, 0].set(True)          # defector always claims
        u2 = graded_step(u, claim, c)
        return (u2, key), None

    (uF, _), _ = jax.lax.scan(step, (jnp.zeros((B, N)), key), None, T)
    u_mean = np.asarray(uF.mean(0))
    share = float(uF.sum(-1).mean()) / N
    rho = u_mean[0] / max(share, 1e-9)
    return u_mean, rho, float(jain(uF).mean()), float(uF.sum(-1).mean()) / T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--cs", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    args = p.parse_args()
    print(f"=== graded contention, scripted landscape (N={args.N}, T={args.T}) ===")
    print(f"  defector=agent0 (always claims). rho=free-ride, eff=delivered/T\n")
    hdr = f"  {'c':>5} | {'yield rho':>9} | {'all-contest rho/eff':>20} | {'EKREM rho/eff':>14}"
    print(hdr)
    for c in args.cs:
        _, ry, jy, ey = rollout_scripted(args.N, args.T, c, "yield")
        _, ra, ja, ea = rollout_scripted(args.N, args.T, c, "all")
        _, re, je, ee = rollout_scripted(args.N, args.T, c, "ekrem")
        print(f"  {c:>5.2f} | {ry:>9.2f} | {ra:>6.2f} / {ea:>5.2f} (Jain {ja:.2f}) | "
              f"{re:>4.2f} / {ee:>4.2f}")
    print("\n  Reading: yield rho==N (defector takes all) at every c -> exploit persists")
    print("  without contest. all-contest gives rho~1 but pays efficiency 1-c (waste).")
    print("  EKREM (centralized) gives rho~1 at efficiency 1 = the upper bound the")
    print("  decentralized cross-attention policy must approach (high eff AND rho~1).")
    # leverage check: worst-off scripted contest vs c
    print("\n  leverage (scripted 'worst-off contests'): rho should fall as c->0")
    for c in args.cs:
        _, rw, jw, ew = rollout_scripted(args.N, args.T, c, "worstoff")
        print(f"    c={c:.2f}: rho={rw:.2f}  Jain={jw:.2f}  eff={ew:.2f}")


if __name__ == "__main__":
    main()
