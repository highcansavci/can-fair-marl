"""Trained gate-1 validation for fairness exploitability (direction B).

Two-phase, two envs:
  Phase 1: train a GGF-welfare-maximizing cooperative team (SOTO-like fair MARL),
           confirm it learns fair behavior (high Jain / low Gini, efficient).
  Phase 2: FREEZE the team, train ONE best-response self-interested defector
           (maximize own utility) vs the frozen fair teammates.
Measure how much the defector free-rides and how fairness/efficiency move.

Envs (shared action semantics: action 1 = act selfishly, 0 = defer):
  claim : 1=CLAIM resource, 0=YIELD. sole claimer +1, else wasted. (efficiency)
  donate: 1=KEEP own +1, 0=DONATE +1 to current worst-off. (pure redistribution)

The GGF-fair joint policy is the SAME pattern in both: "act selfishly only when
you are the worst-off agent". A self-interested defector always acts selfishly.

    python -m can.fair_train --env donate
"""
import argparse
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn


class Pol(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(32)(x))
        x = nn.tanh(nn.Dense(32)(x))
        return nn.Dense(2)(x)                       # logits over {defer, selfish}


def ggf(u):                                          # u (...,n) -> welfare (...)
    n = u.shape[-1]
    s = jnp.sort(u, axis=-1)                          # ascending
    w = (jnp.arange(n, 0, -1)).astype(jnp.float32)    # decreasing -> weight worst-off
    w = w / w.sum()
    return jnp.sum(s * w, axis=-1)


def env_step(env, u, acts):
    """u (B,n) utilities, acts (B,n) in {0,1}. Returns u' (B,n)."""
    B, n = u.shape
    selfish = (acts == 1)
    cnt = selfish.sum(-1)                              # (B,) number claiming
    sole = cnt == 1
    winner = jnp.argmax(selfish, -1)                   # first claimer
    add_sole = jax.nn.one_hot(winner, n) * sole[:, None]
    if env == "claim":
        # sole claimer wins; 0 claimers or collision -> WASTED
        return u + add_sole
    if env == "donate":
        # sole claimer wins; ALL-YIELD auto-routes to worst-off; collision waste
        none = cnt == 0
        worst = jnp.argmin(u, -1)
        return u + add_sole + jax.nn.one_hot(worst, n) * none[:, None]
    if env == "fairneed":
        # pure NEED-BASED allocator: worst-off OVERALL gets it (claims ignored)
        # -> trivially incentive-compatible at any n (defector never exceeds
        # fair share). Closes the n=2 sole-rich-claimer leak of 'fairmech'.
        return u + jax.nn.one_hot(jnp.argmin(u, -1), n)
    if env == "matthew":
        # Matthew effect (FEN's signature rich-get-richer): among claimers the
        # RICHEST wins -> being ahead lets you win contests -> snowballing
        # inequality. Fair play needs the rich to YIELD. 0 claimers -> wasted.
        neg = u.min() - 1.0
        u_claim = jnp.where(selfish, u, neg)
        has = selfish.any(-1)
        win = jnp.argmax(u_claim, -1)                  # richest claimer
        return u + jax.nn.one_hot(win, n) * has[:, None]
    # fairmech: FAIRNESS-AWARE ALLOCATOR (the proposed robust method). The
    # resource goes to the WORST-OFF among claimers (else worst-off overall) ->
    # always allocated (efficient), and claiming only wins when you deserve it
    # (you are worst-off) -> a rich defector's greedy claim earns nothing.
    big = u.max() + 1.0
    u_claim = jnp.where(selfish, u, big)               # mask non-claimers
    has_claim = selfish.any(-1)
    win_claim = jnp.argmin(u_claim, -1)                # worst-off claimer
    win = jnp.where(has_claim, win_claim, jnp.argmin(u, -1))
    return u + jax.nn.one_hot(win, n)


def obs_fn(u, t, T):
    """(B,n)->(B,n,5): own, dev-from-mean, dev-from-min, is_min, time."""
    mean = u.mean(-1, keepdims=True)
    mn = u.min(-1, keepdims=True)
    is_min = (u == mn).astype(jnp.float32)
    tf = jnp.full_like(u, t / T)
    return jnp.stack([u / T, (u - mean) / T, (u - mn) / T, is_min, tf], -1)


def gini(u):
    s = jnp.sort(u, -1); n = u.shape[-1]
    cum = jnp.cumsum(s, -1)
    tot = jnp.clip(cum[..., -1], 1e-9, None)
    return (n + 1 - 2 * cum.sum(-1) / tot) / n


def jain(u):
    return (u.sum(-1) ** 2) / (u.shape[-1] * jnp.clip((u ** 2).sum(-1), 1e-9, None))


def make_rollout(env, n, T, B):
    pol = Pol()

    def rollout(fair_p, def_p, def_idx, key):
        """def_idx: agent controlled by def_p (-1 -> all use fair_p).
        Returns final u (B,n), trajectory (obs (T,B,n,5), acts (T,B,n))."""
        u0 = jnp.zeros((B, n))

        def step(carry, t):
            u, key = carry
            o = obs_fn(u, t, T)                         # (B,n,5)
            lf = pol.apply(fair_p, o)
            ld = pol.apply(def_p, o)
            sel = (jnp.arange(n) == def_idx)[None, :, None]  # (1,n,1)
            logits = jnp.where(sel, ld, lf)            # defector row uses def_p
            key, ka = jax.random.split(key)
            acts = jax.random.categorical(ka, logits)  # (B,n)
            u2 = env_step(env, u, acts)
            return (u2, key), (o, acts)

        (uF, _), (obs, acts) = jax.lax.scan(step, (u0, key), jnp.arange(T))
        return uF, obs, acts

    return pol, rollout


def logp_of(pol, params, obs, acts):
    logits = pol.apply(params, obs)                    # (T,B,n,2)
    lsm = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]   # (T,B,n)


def train(env, n=4, T=100, B=512, fair_iters=400, def_iters=400, lr=3e-3, seed=0):
    pol, rollout = make_rollout(env, n, T, B)
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    fair_p = pol.init(ki, jnp.zeros((1, 5)))
    tx = optax.adam(lr)

    # ---- phase 1: cooperative GGF-fair team (all agents share fair_p) ----
    opt = tx.init(fair_p)

    @jax.jit
    def fair_update(fair_p, opt, key):
        uF, obs, acts = rollout(fair_p, fair_p, -1, key)
        R = ggf(uF)                                    # (B,) team welfare
        adv = R - R.mean()

        def loss(p):
            lp = logp_of(pol, p, obs, acts)            # (T,B,n)
            return -(adv[None, :, None] * lp).mean()
        g = jax.grad(loss)(fair_p)
        upd, opt = tx.update(g, opt)
        return optax.apply_updates(fair_p, upd), opt, R.mean()

    for it in range(fair_iters):
        key, k = jax.random.split(key)
        fair_p, opt, _ = fair_update(fair_p, opt, k)

    # evaluate the all-fair team
    key, k = jax.random.split(key)
    uF, _, _ = rollout(fair_p, fair_p, -1, k)
    fair_u = np.asarray(uF.mean(0))
    fair_jain = float(jain(uF).mean()); fair_gini = float(gini(uF).mean())
    fair_eff = float(uF.sum(-1).mean()) / T

    # ---- phase 2: best-response self-interested defector (agent 0) ----
    key, kd = jax.random.split(key)
    def_p = pol.init(kd, jnp.zeros((1, 5)))
    optd = tx.init(def_p)

    @jax.jit
    def def_update(def_p, optd, key):
        uF, obs, acts = rollout(fair_p, def_p, 0, key)
        R = uF[:, 0]                                   # defector's OWN utility
        adv = R - R.mean()

        def loss(p):
            lp = logp_of(pol, p, obs, acts)[:, :, 0]   # defector's logp (T,B)
            return -(adv[None, :] * lp).mean()
        g = jax.grad(loss)(def_p)
        upd, optd = tx.update(g, optd)
        return optax.apply_updates(def_p, upd), optd, R.mean()

    for it in range(def_iters):
        key, k = jax.random.split(key)
        def_p, optd, _ = def_update(def_p, optd, k)

    key, k = jax.random.split(key)
    uF, _, _ = rollout(fair_p, def_p, 0, k)
    ex_u = np.asarray(uF.mean(0))
    ex_jain = float(jain(uF).mean()); ex_gini = float(gini(uF).mean())
    ex_eff = float(uF.sum(-1).mean()) / T
    fair_share = float(uF.sum(-1).mean()) / n

    print(f"\n=== fairness exploitability [env={env}, n={n}, T={T}] ===")
    print(f"  PHASE 1 all-fair team: util={np.round(fair_u,1)}  "
          f"Jain={fair_jain:.3f}  Gini={fair_gini:.3f}  eff={fair_eff:.3f}")
    print(f"  PHASE 2 +defector(ag0): util={np.round(ex_u,1)}  "
          f"Jain={ex_jain:.3f}  Gini={ex_gini:.3f}  eff={ex_eff:.3f}")
    print(f"  EXPLOIT: defector util={ex_u[0]:.1f} vs fair share={fair_share:.1f} "
          f"-> free-ride x{ex_u[0]/max(fair_share,1e-9):.2f}")
    print(f"  fairness: Jain {fair_jain:.3f}->{ex_jain:.3f}, "
          f"Gini {fair_gini:.3f}->{ex_gini:.3f}, eff {fair_eff:.3f}->{ex_eff:.3f}")
    return dict(env=env, fair_jain=fair_jain, ex_jain=ex_jain, fair_gini=fair_gini,
                ex_gini=ex_gini, free_ride=ex_u[0] / max(fair_share, 1e-9),
                fair_eff=fair_eff, ex_eff=ex_eff)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", nargs="+", default=["claim", "donate"])
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for e in args.env:
        train(e, n=args.n, T=args.T, fair_iters=args.iters, def_iters=args.iters,
              seed=args.seed)


if __name__ == "__main__":
    main()
