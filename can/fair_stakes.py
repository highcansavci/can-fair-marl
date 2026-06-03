"""Third environment (leverage-preserving): STOCHASTIC-STAKES STREAM.

Single resource per step with a random value v_t (lumpy: usually small, occasionally
a jackpot). m claimers split (1-c)*v_t/m; a sole claimer gets v_t; m=0 routes v_t to
the worst-off. Equal-split preserves Prop.-1 leverage. The NEW challenge: a
free-rider's appropriated surplus concentrates on the high-value steps, so the
cooperators must contest \emph{selectively}---when the stakes are high---and tolerate
low-stakes steps. Agents observe v_t (an extra token feature); everything else
(2-action cross-attention, league training, bounded audit) is unchanged. The value
distribution has mean 1, so efficiency normalizes as in the base game.

    python -m can.fair_stakes --selftest
    python -m can.fair_stakes --smoke
    python -m can.fair_stakes --generations 6 --coop_iters 1200 --br_iters 1000
"""
import os
import csv
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .fair_xattn import XAttn, evaluate, disc_rtg, coop_welfare

N, T = 6, 100
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]
DMAX = 2
LOW, HIGH, PHIGH = 0.25, 3.25, 0.25                 # E[v] = 1.0 (lumpy jackpots)
VMAX = HIGH


def sample_v(key, B):
    return jnp.where(jax.random.uniform(key, (B,)) < PHIGH, HIGH, LOW)


def features_stakes(u, cc, t, T, v):
    mean = u.mean(-1, keepdims=True); mn = u.min(-1, keepdims=True)
    cr = cc / jnp.clip(t, 1.0, None)
    vb = jnp.broadcast_to((v / VMAX)[:, None], u.shape)          # current stakes
    return jnp.stack([u / T, (u - mean) / T, (u - mn) / T,
                      (u == mn).astype(jnp.float32), cr,
                      jnp.full_like(u, t / T), vb], -1)           # (B,N,7)


def value_step(u, claim, c, v):
    B, n = u.shape
    m = claim.sum(-1)
    share = jnp.where(m == 1, v, jnp.where(m >= 2, (1.0 - c) * v / jnp.clip(m, 1, None),
                                           0.0))
    add_claim = claim.astype(jnp.float32) * share[:, None]
    worst = jnp.argmin(u, -1)
    none = (m == 0).astype(jnp.float32)
    add_none = jax.nn.one_hot(worst, n) * (none * v)[:, None]
    return u + add_claim + add_none


def _logp(pol, p, toks, acts):
    lsm = jax.nn.log_softmax(pol.apply(p, toks))
    return jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]


def _coop_vs_pool(coop_p, pool_stk, c, B, iters, lr, key, dmax=DMAX):
    pol = XAttn()
    P = jax.tree_util.tree_leaves(pool_stk)[0].shape[0]
    tx = optax.adam(lr); opt = tx.init(coop_p)

    def rollout(coop_p, def_mask, pidx, key):
        dm = def_mask.astype(bool)
        def step(carry, t):
            u, cc, key = carry
            key, kv, ka, kb = jax.random.split(key, 4)
            v = sample_v(kv, B)
            tok = features_stakes(u, cc, t, T, v)
            cl = pol.apply(coop_p, tok)
            dl_all = jax.vmap(lambda dp: pol.apply(dp, tok))(pool_stk)
            dl = dl_all[pidx, jnp.arange(B)]
            acts = jnp.where(dm, jax.random.categorical(kb, dl),
                             jax.random.categorical(ka, cl))
            u = value_step(u, acts == 1, c, v)
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


def br_audit(coop_p, c, B=512, iters=1200, lr=3e-3, seed=7, return_params=False):
    pol = XAttn()
    key = jax.random.PRNGKey(seed); key, ki = jax.random.split(key)
    dp = pol.init(ki, jnp.zeros((1, N, 7))); tx = optax.adam(lr); opt = tx.init(dp)

    def rollout(dp, d_idx, key):
        d_oh = jax.nn.one_hot(d_idx, N)
        def step(carry, t):
            u, cc, key = carry
            key, kv, ka, kb = jax.random.split(key, 4)
            v = sample_v(kv, B)
            tok = features_stakes(u, cc, t, T, v)
            cl, dl = pol.apply(coop_p, tok), pol.apply(dp, tok)
            acts = jnp.where(d_oh.astype(bool), jax.random.categorical(kb, dl),
                             jax.random.categorical(ka, cl))
            u = value_step(u, acts == 1, c, v)
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
            return -(adv[None, :, None] * _logp(pol, p, toks, acts)
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


def league_train(c, generations=5, coop_iters=1000, br_iters=800, B=512, lr=3e-3,
                 seed=0):
    pol = XAttn()
    key = jax.random.PRNGKey(seed); key, kc, kd = jax.random.split(key, 3)
    coop_p = pol.init(kc, jnp.zeros((1, N, 7)))
    pool = [pol.init(kd, jnp.zeros((1, N, 7)))]
    for gen in range(generations):
        pool_stk = jax.tree.map(lambda *xs: jnp.stack(xs), *pool)
        key, k1 = jax.random.split(key)
        coop_p = _coop_vs_pool(coop_p, pool_stk, c, B, coop_iters, lr, k1)
        new_def = br_audit(coop_p, c, B=B, iters=br_iters, seed=seed * 100 + gen,
                           return_params=True)
        pool.append(new_def)
    return coop_p


def d0_eff(coop_p, c, B=512, seed=321):
    pol = XAttn()
    def step(carry, t):
        u, cc, key = carry
        key, kv, ka = jax.random.split(key, 3)
        v = sample_v(kv, B)
        tok = features_stakes(u, cc, t, T, v)
        acts = jax.random.categorical(ka, pol.apply(coop_p, tok))
        u = value_step(u, acts == 1, c, v)
        return (u, cc + (acts == 1), key), None
    (uF, _, _), _ = jax.lax.scan(
        step, (jnp.zeros((B, N)), jnp.zeros((B, N)), jax.random.PRNGKey(seed)),
        jnp.arange(T))
    return evaluate(uF, jnp.zeros((B, N)), T)[2]


def selftest():
    k = jax.random.PRNGKey(0)
    u = jnp.array([[2.0, 1.0, 0.0]])
    print("sole claim v=3.25:", np.asarray(value_step(u, jnp.array([[1, 0, 0]]),
          0.5, jnp.array([3.25])))[0])      # agent0 +3.25
    print("contest v=3.25 c=0.5:", np.asarray(value_step(u, jnp.array([[1, 1, 0]]),
          0.5, jnp.array([3.25])))[0])      # each +0.5*3.25/2=0.8125
    print("none v=3.25 -> worst:", np.asarray(value_step(u, jnp.array([[0, 0, 0]]),
          0.5, jnp.array([3.25])))[0])      # agent2 +3.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--coop_iters", type=int, default=1000)
    ap.add_argument("--br_iters", type=int, default=800)
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    cs, seeds = CS, SEEDS
    if a.smoke:
        cs, seeds = [0.5], [0]
        a.generations, a.coop_iters, a.br_iters = 2, 200, 200
    os.makedirs("results", exist_ok=True)
    out = "results/stakes_can.csv"
    if not a.smoke and os.path.exists(out):
        os.remove(out)
    header = ["c", "seed", "d0_eff", "d1_eff", "br_rho"]
    for s in seeds:
        for c in cs:
            p = league_train(c, generations=a.generations, coop_iters=a.coop_iters,
                             br_iters=a.br_iters, seed=s)
            e0 = float(d0_eff(p, c))
            rho, e1 = br_audit(p, c, iters=1200, seed=800 + s)
            row = dict(c=c, seed=s, d0_eff=e0, d1_eff=float(e1), br_rho=float(rho))
            if not a.smoke:
                new = not os.path.exists(out)
                with open(out, "a", newline="") as f:
                    w = csv.DictWriter(f, header)
                    if new:
                        w.writeheader()
                    w.writerow(row); f.flush()
            print(f"[stakes CAN] c={c} s={s}: d0_eff={e0:.3f} d1_eff={e1:.3f} "
                  f"br_rho={rho:.2f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
