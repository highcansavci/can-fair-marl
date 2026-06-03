"""Architecture ablation: is CAN's win from ATTENTION, or just from behaviour
features + league training? We compare the cross-attention policy against two
permutation-equivariant non-attention policies (mean-pool, deep-sets) and a
recurrent aggregator (bi-GRU over agents), ALL under identical league (PSRO)
training, the same constant best-response adversary (an XAttn defector), and the
same bounded audit on the graded-contention game.

We also record the efficiency the reviewer asked for: D=0 efficiency (no
free-rider --- the easy case) AND D>=1 efficiency (delivered/T WITH the
best-response defector present --- the honest waste a decentralized contester
pays, where the centralized oracle is uniquely zero-waste).

    python -m can.fair_arch_ablation --smoke
    python -m can.fair_arch_ablation
"""
import os
import csv
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from .fair_graded import graded_step
from .fair_xattn import XAttn, features, evaluate, disc_rtg, coop_welfare

N, T = 6, 100
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]
DMAX = 2


# --------------------------- policy architectures ------------------------------
class MeanPool(nn.Module):
    """Permutation-equivariant: each agent conditions on its token + the mean of
    all raw tokens (no learned aggregation). The minimal count-aware baseline."""
    hidden: int = 64

    @nn.compact
    def __call__(self, tok):                                   # (...,N,F)->(...,N,2)
        ctx = jnp.broadcast_to(tok.mean(-2, keepdims=True), tok.shape)
        h = nn.tanh(nn.Dense(self.hidden)(jnp.concatenate([tok, ctx], -1)))
        h = nn.tanh(nn.Dense(self.hidden)(h))
        return nn.Dense(2)(h)


class DeepSets(nn.Module):
    """Permutation-equivariant: mean-pool a LEARNED per-agent embedding (Zaheer
    et al.), then decode per agent. A strictly more expressive pool than MeanPool."""
    hidden: int = 64

    @nn.compact
    def __call__(self, tok):
        phi = nn.tanh(nn.Dense(self.hidden)(tok))              # learned embedding
        ctx = jnp.broadcast_to(phi.mean(-2, keepdims=True), phi.shape)
        h = nn.tanh(nn.Dense(self.hidden)(jnp.concatenate([tok, ctx], -1)))
        return nn.Dense(2)(h)


class GRUAgg(nn.Module):
    """Recurrent aggregator: a bi-directional GRU over the agent tokens (index
    order), per-agent output. Order-sensitive (not permutation-equivariant) ---
    included as the recurrent alternative."""
    hidden: int = 64

    @nn.compact
    def __call__(self, tok):                                   # (...,N,F)->(...,N,2)
        lead = tok.shape[:-2]
        x = tok.reshape((-1, N, tok.shape[-1]))                # (M,N,F)
        fwd = nn.RNN(nn.GRUCell(features=self.hidden))(x)      # scans over N
        bwd = nn.RNN(nn.GRUCell(features=self.hidden))(x[:, ::-1])[:, ::-1]
        h = nn.tanh(nn.Dense(self.hidden)(
            jnp.concatenate([x, fwd, bwd], -1)))
        out = nn.Dense(2)(h)
        return out.reshape(lead + (N, 2))


ARCHES = {"XAttn": XAttn, "MeanPool": MeanPool, "DeepSets": DeepSets,
          "GRU": GRUAgg}


def _logp(pol, p, toks, acts):
    lsm = jax.nn.log_softmax(pol.apply(p, toks))
    return jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]


# --------------------------- generic training ----------------------------------
def _coop_vs_pool(coop_cls, coop_p, pool_stk, c, B, iters, lr, key, dmax=DMAX):
    """Cooperators (coop_cls) vs a FROZEN pool of XAttn defectors (stacked)."""
    cpol, dpol = coop_cls(), XAttn()
    P = jax.tree_util.tree_leaves(pool_stk)[0].shape[0]
    tx = optax.adam(lr); opt = tx.init(coop_p)

    def rollout(coop_p, def_mask, pidx, key):
        dm = def_mask.astype(bool)
        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            cl = cpol.apply(coop_p, tok)
            dl_all = jax.vmap(lambda dp: dpol.apply(dp, tok))(pool_stk)
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
            lsm = jax.nn.log_softmax(cpol.apply(p, toks))
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


def br_audit(coop_cls, coop_p, c, B=512, iters=1200, lr=3e-3, seed=7,
             return_params=False):
    """Train an XAttn best-response defector vs the FROZEN cooperators (coop_cls).
    Returns (rho, D=1 efficiency) or the defector params."""
    cpol, dpol = coop_cls(), XAttn()
    key = jax.random.PRNGKey(seed); key, ki = jax.random.split(key)
    dp = dpol.init(ki, jnp.zeros((1, N, 6))); tx = optax.adam(lr); opt = tx.init(dp)

    def rollout(dp, d_idx, key):
        d_oh = jax.nn.one_hot(d_idx, N)
        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            cl, dl = cpol.apply(coop_p, tok), dpol.apply(dp, tok)
            key, ka, kb = jax.random.split(key, 3)
            acts = jnp.where(d_oh.astype(bool), jax.random.categorical(kb, dl),
                             jax.random.categorical(ka, cl))
            u = graded_step(u, acts == 1, c)
            return (u, cc + (acts == 1), key), (tok, acts)
        (uF, _, _), (toks, acts) = jax.lax.scan(
            step, (jnp.zeros((B, N)), jnp.zeros((B, N)), key), jnp.arange(T))
        return uF, toks, acts, d_oh

    @jax.jit
    def upd(dp, opt, key):
        key, kd, kr = jax.random.split(key, 3)
        d_idx = jax.random.randint(kd, (B,), 0, N)
        uF, toks, acts, d_oh = rollout(dp, d_idx, kr)
        R = (uF * d_oh).sum(-1); adv = R - R.mean()
        def loss(p):
            return -(adv[None, :, None] * _logp(dpol, p, toks, acts)
                     * d_oh[None]).sum(-1).mean()
        g = jax.grad(loss)(dp); u, opt = tx.update(g, opt)
        return optax.apply_updates(dp, u), opt, key
    for _ in range(iters):
        dp, opt, key = upd(dp, opt, key)
    if return_params:
        return dp
    key, kd, kr = jax.random.split(key, 3)
    d_idx = jax.random.randint(kd, (B,), 0, N)
    uF, _, _, _ = rollout(dp, d_idx, kr)
    rho, _, eff = evaluate(uF, jax.nn.one_hot(d_idx, N), T)
    return rho, eff


def league_train(coop_cls, c, generations=4, coop_iters=900, br_iters=700,
                 B=512, lr=3e-3, seed=0):
    cpol, dpol = coop_cls(), XAttn()
    key = jax.random.PRNGKey(seed); key, kc, kd = jax.random.split(key, 3)
    coop_p = cpol.init(kc, jnp.zeros((1, N, 6)))
    pool = [dpol.init(kd, jnp.zeros((1, N, 6)))]
    for gen in range(generations):
        pool_stk = jax.tree.map(lambda *xs: jnp.stack(xs), *pool)
        key, k1 = jax.random.split(key)
        coop_p = _coop_vs_pool(coop_cls, coop_p, pool_stk, c, B, coop_iters, lr, k1)
        new_def = br_audit(coop_cls, coop_p, c, B=B, iters=br_iters,
                           seed=seed * 100 + gen, return_params=True)
        pool.append(new_def)
    return coop_p


def d0_eff(coop_cls, coop_p, c, B=512, seed=321):
    """Efficiency with NO defector (delivered/T)."""
    cpol = coop_cls()
    def step(carry, t):
        u, cc, key = carry
        tok = features(u, cc, t, T)
        key, ka = jax.random.split(key)
        acts = jax.random.categorical(ka, cpol.apply(coop_p, tok))
        u = graded_step(u, acts == 1, c)
        return (u, cc + (acts == 1), key), None
    (uF, _, _), _ = jax.lax.scan(
        step, (jnp.zeros((B, N)), jnp.zeros((B, N)), jax.random.PRNGKey(seed)),
        jnp.arange(T))
    return evaluate(uF, jnp.zeros((B, N)), T)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--coop_iters", type=int, default=1000)
    ap.add_argument("--br_iters", type=int, default=800)
    ap.add_argument("--archs", nargs="+", default=list(ARCHES),
                    help="subset of architectures to run")
    ap.add_argument("--out", default="results/arch_ablation.csv")
    a = ap.parse_args()
    arches = {k: ARCHES[k] for k in a.archs}
    cs, seeds = CS, SEEDS
    if a.smoke:
        cs, seeds = [0.5], [0]
        a.generations, a.coop_iters, a.br_iters = 2, 200, 200
    os.makedirs("results", exist_ok=True)
    out = a.out
    if not a.smoke and os.path.exists(out):
        os.remove(out)
    header = ["arch", "c", "seed", "d0_eff", "d1_eff", "br_rho"]
    # seed-outer so an early stop still covers all architectures (more seeds = better)
    for s in seeds:
        for arch, cls in arches.items():
            for c in cs:
                p = league_train(cls, c, generations=a.generations,
                                 coop_iters=a.coop_iters, br_iters=a.br_iters,
                                 seed=s)
                e0 = float(d0_eff(cls, p, c))
                rho, e1 = br_audit(cls, p, c, iters=1200, seed=800 + s)
                row = dict(arch=arch, c=c, seed=s, d0_eff=e0,
                           d1_eff=float(e1), br_rho=float(rho))
                if not a.smoke:
                    new = not os.path.exists(out)
                    with open(out, "a", newline="") as f:
                        w = csv.DictWriter(f, header)
                        if new:
                            w.writeheader()
                        w.writerow(row); f.flush()
                print(f"[{arch}] c={c} s={s}: d0_eff={e0:.3f} d1_eff={e1:.3f} "
                      f"br_rho={rho:.2f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
