"""Larger-dmax stress (reviewer: N=6,dmax=2 is an easy 0/1/2-grabber problem).

Train CAN with up to dmax defectors per episode (D~Unif{0..dmax}) and audit against
a best-response defector COALITION of size D (D agents sharing one best-response
policy, maximizing the group's utility). Tests that the contention inference
survives (i) a harder 'how many grabbers' problem at training time and (ii) multiple
coordinated free-riders at audit time.

    python -m can.fair_dmax --smoke
    python -m can.fair_dmax --dmax 4
"""
import os
import csv
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .fair_xattn import XAttn, features, evaluate, disc_rtg, coop_welfare
from .fair_graded import graded_step

N, T = 6, 100
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]
AUDIT_D = [1, 2, 3]


def _coop_vs_pool(coop_p, pool_stk, c, dmax, B, iters, lr, key):
    pol = XAttn()
    P = jax.tree_util.tree_leaves(pool_stk)[0].shape[0]
    tx = optax.adam(lr); opt = tx.init(coop_p)

    def rollout(coop_p, def_mask, pidx, key):
        dm = def_mask.astype(bool)
        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            cl = pol.apply(coop_p, tok)
            dl_all = jax.vmap(lambda dp: pol.apply(dp, tok))(pool_stk)
            dl = dl_all[pidx, jnp.arange(B)]
            key, ka, kb = jax.random.split(key, 3)
            acts = jnp.where(dm, jax.random.categorical(kb, dl),
                             jax.random.categorical(ka, cl))
            u = graded_step(u, acts == 1, c)
            return (u, cc + (acts == 1), key), (tok, acts, u)
        (uF, _, _), out = jax.lax.scan(
            step, (jnp.zeros((B, N)), jnp.zeros((B, N)), key), jnp.arange(T))
        return out

    @jax.jit
    def upd(coop_p, opt, key, ec):
        key, ks, kp, kr = jax.random.split(key, 4)
        Dn = jax.random.randint(ks, (B,), 0, dmax + 1)
        rank = jnp.argsort(jnp.argsort(
            jax.random.uniform(jax.random.fold_in(ks, 1), (B, N)), -1), -1)
        def_mask = (rank < Dn[:, None]).astype(jnp.float32)
        pidx = jax.random.randint(kp, (B,), 0, P)
        coop = 1.0 - def_mask
        toks, acts, useq = rollout(coop_p, def_mask, pidx, kr)
        Wc = jax.vmap(coop_welfare, in_axes=(0, None))(useq, def_mask)
        ac = disc_rtg(Wc - jnp.concatenate([jnp.zeros((1, B)), Wc[:-1]], 0))
        ac = ac - ac.mean(1, keepdims=True)
        def loss(p):
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
            ent = -(jnp.exp(lsm) * lsm).sum(-1)
            return -(ac[:, :, None] * lp * coop[None]).sum(-1).mean() \
                - ec * (ent * coop[None]).sum(-1).mean()
        g = jax.grad(loss)(coop_p); u, opt = tx.update(g, opt)
        return optax.apply_updates(coop_p, u), opt, key

    ent_hi, ent_lo = 0.05, 0.003
    for it in range(iters):
        ec = ent_hi * (1 - it / iters) + ent_lo * (it / iters)
        coop_p, opt, key = upd(coop_p, opt, key, ec)
    return coop_p


def br_audit_coalition(coop_p, c, n_def, B=512, iters=1200, lr=3e-3, seed=7,
                       return_params=False):
    """Best-response coalition of n_def defectors (shared policy) vs frozen coop."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed); key, ki = jax.random.split(key)
    dp = pol.init(ki, jnp.zeros((1, N, 6))); tx = optax.adam(lr); opt = tx.init(dp)

    def def_mask_of(key):
        rank = jnp.argsort(jnp.argsort(jax.random.uniform(key, (B, N)), -1), -1)
        return (rank < n_def).astype(jnp.float32)

    def rollout(dp, def_mask, key):
        dm = def_mask.astype(bool)
        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            cl, dl = pol.apply(coop_p, tok), pol.apply(dp, tok)
            key, ka, kb = jax.random.split(key, 3)
            acts = jnp.where(dm, jax.random.categorical(kb, dl),
                             jax.random.categorical(ka, cl))
            u = graded_step(u, acts == 1, c)
            return (u, cc + (acts == 1), key), (tok, acts)
        (uF, _, _), (toks, acts) = jax.lax.scan(
            step, (jnp.zeros((B, N)), jnp.zeros((B, N)), key), jnp.arange(T))
        return uF, toks, acts

    @jax.jit
    def upd(dp, opt, key):
        key, km, kr = jax.random.split(key, 3)
        def_mask = def_mask_of(km)
        uF, toks, acts = rollout(dp, def_mask, kr)
        R = (uF * def_mask).sum(-1); adv = R - R.mean()             # group utility
        def loss(p):
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
            return -(adv[None, :, None] * lp * def_mask[None]).sum(-1).mean()
        g = jax.grad(loss)(dp); u, opt = tx.update(g, opt)
        return optax.apply_updates(dp, u), opt, key
    for _ in range(iters):
        dp, opt, key = upd(dp, opt, key)
    if return_params:
        return dp
    key, km, kr = jax.random.split(key, 3)
    def_mask = def_mask_of(km)
    uF, _, _ = rollout(dp, def_mask, kr)
    return evaluate(uF, def_mask, T)[0]


def league_train(c, dmax, generations=6, coop_iters=1200, br_iters=1000, B=512,
                 lr=3e-3, seed=0):
    pol = XAttn()
    key = jax.random.PRNGKey(seed); key, kc, kd = jax.random.split(key, 3)
    coop_p = pol.init(kc, jnp.zeros((1, N, 6)))
    pool = [pol.init(kd, jnp.zeros((1, N, 6)))]
    for gen in range(generations):
        pool_stk = jax.tree.map(lambda *xs: jnp.stack(xs), *pool)
        key, k1 = jax.random.split(key)
        coop_p = _coop_vs_pool(coop_p, pool_stk, c, dmax, B, coop_iters, lr, k1)
        new_def = br_audit_coalition(coop_p, c, 1, B=B, iters=br_iters,
                                     seed=seed * 100 + gen, return_params=True)
        pool.append(new_def)
    return coop_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dmax", type=int, default=4)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--coop_iters", type=int, default=1200)
    ap.add_argument("--br_iters", type=int, default=1000)
    a = ap.parse_args()
    cs, seeds, ds = CS, SEEDS, AUDIT_D
    if a.smoke:
        cs, seeds, ds = [0.5], [0], [1, 3]
        a.generations, a.coop_iters, a.br_iters = 2, 200, 200
    os.makedirs("results", exist_ok=True)
    out = f"results/dmax{a.dmax}_stress.csv"
    if not a.smoke and os.path.exists(out):
        os.remove(out)
    header = ["c", "seed", "audit_D", "br_rho"]
    for s in seeds:
        for c in cs:
            p = league_train(c, a.dmax, generations=a.generations,
                             coop_iters=a.coop_iters, br_iters=a.br_iters, seed=s)
            for D in ds:
                rho = float(br_audit_coalition(p, c, D, iters=1200, seed=800 + s))
                row = dict(c=c, seed=s, audit_D=D, br_rho=rho)
                if not a.smoke:
                    new = not os.path.exists(out)
                    with open(out, "a", newline="") as f:
                        w = csv.DictWriter(f, header)
                        if new:
                            w.writeheader()
                        w.writerow(row); f.flush()
                print(f"[dmax{a.dmax}] c={c} s={s} D={D}: br_rho={rho:.2f}",
                      flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
