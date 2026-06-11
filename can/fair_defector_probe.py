"""What does the audit's best-response defector actually converge to?

Supports the 'dominance, not equilibrium' remark around Prop. 1: train a CAN
league cooperator, train the standard 1500-update best-response defector against
it, then report the defector's empirical claim-rate (overall and over time) next
to the always-claim script's rho. If the defector's claim-rate is interior
(<1.0) where CAN's contention is credible, the best response is NOT the
always-claim script --- per-step dominance does not pin down the repeated-game
play.

    python -m can.fair_defector_probe --c 0.5 0.9 --seed 0
"""
import argparse

import numpy as np
import jax
import jax.numpy as jnp

from .fair_graded import graded_step
from .fair_xattn import (XAttn, features, evaluate, league_train,
                         best_response_audit)

N, T = 6, 100


def probe(coop_p, def_p, c, B=2048, seed=42):
    """Roll out frozen coop vs frozen BR defector at a random index; return
    (rho, defector claim-rate per step (T,), cooperator claim-rate per step)."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    kd, kr = jax.random.split(key)
    d_oh = jax.nn.one_hot(jax.random.randint(kd, (B,), 0, N), N)
    dm = d_oh.astype(bool)

    def step(carry, t):
        u, cc, key = carry
        tok = features(u, cc, t, T)
        key, ka, kb = jax.random.split(key, 3)
        ca = jax.random.categorical(ka, pol.apply(coop_p, tok))
        da = jax.random.categorical(kb, pol.apply(def_p, tok))
        acts = jnp.where(dm, da, ca)
        u = graded_step(u, acts == 1, c)
        return (u, cc + (acts == 1), key), (acts == 1)
    (uF, _, _), claims = jax.lax.scan(
        step, (jnp.zeros((B, N)), jnp.zeros((B, N)), kr), jnp.arange(T))
    rho, _, _ = evaluate(uF, d_oh, T)
    cl = np.asarray(claims, float)                              # (T,B,N)
    dmn = np.asarray(d_oh)[None]
    def_rate = (cl * dmn).sum((1, 2)) / dmn.sum((1, 2))         # (T,)
    coop_rate = (cl * (1 - dmn)).sum((1, 2)) / (1 - dmn).sum((1, 2))
    return rho, def_rate, coop_rate


def always_claim_rho(coop_p, c, B=2048, seed=43):
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    kd, kr = jax.random.split(key)
    d_oh = jax.nn.one_hot(jax.random.randint(kd, (B,), 0, N), N)

    def step(carry, t):
        u, cc, key = carry
        tok = features(u, cc, t, T)
        key, ka = jax.random.split(key)
        acts = jnp.where(d_oh.astype(bool), 1,
                         jax.random.categorical(ka, pol.apply(coop_p, tok)))
        u = graded_step(u, acts == 1, c)
        return (u, cc + (acts == 1), key), None
    (uF, _, _), _ = jax.lax.scan(
        step, (jnp.zeros((B, N)), jnp.zeros((B, N)), kr), jnp.arange(T))
    return evaluate(uF, d_oh, T)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, nargs="+", default=[0.5, 0.9])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--coop_iters", type=int, default=1200)
    ap.add_argument("--br_iters", type=int, default=1000)
    ap.add_argument("--audit_iters", type=int, default=1500)
    a = ap.parse_args()
    for c in a.c:
        coop_p = league_train(N, T, c, generations=a.generations,
                              coop_iters=a.coop_iters, br_iters=a.br_iters,
                              seed=a.seed)
        def_p = best_response_audit(coop_p, N, T, c, iters=a.audit_iters,
                                    seed=800 + a.seed, return_params=True)
        rho, dr, cr = probe(coop_p, def_p, c)
        ac = always_claim_rho(coop_p, c)
        q = T // 4
        print(f"\n[probe] c={c} seed={a.seed}: BR rho={rho:.2f} "
              f"always-claim rho={ac:.2f}")
        print(f"  defector claim-rate: overall={dr.mean():.3f}  by quarter="
              f"{dr[:q].mean():.3f}/{dr[q:2*q].mean():.3f}/"
              f"{dr[2*q:3*q].mean():.3f}/{dr[3*q:].mean():.3f}")
        print(f"  cooperator claim-rate: overall={cr.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
