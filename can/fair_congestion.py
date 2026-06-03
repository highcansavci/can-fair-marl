"""Second environment (leverage-preserving): MULTI-SERVER CONGESTION.

M parallel unit resources ('servers') each step. Each agent CLAIMS one server or
YIELDS. On a server claimed by k agents: k=1 -> sole claimer gets 1; k>=2 -> each
claimer gets (1-c)/k (waste c); k=0 -> the server routes to a neediest (lowest-u)
agent. Equal-split among claimers preserves Prop.-1 leverage: a worst-off
cooperator that contests a free-rider's server gets (1-c)/2 > 0, so contesting
strictly dominates yielding. The NEW challenge vs.\ the single-resource game: the
cooperators must infer WHICH server each free-rider targets and contest just that
one, leaving the others to turn-take. CAN gets per-server claim-rate features and an
(M+1)-way action head; cross-attention, league training and the bounded audit are
unchanged.

    python -m can.fair_congestion --selftest
    python -m can.fair_congestion --smoke
    python -m can.fair_congestion --generations 6 --coop_iters 1200 --br_iters 1000
"""
import os
import csv
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from .fair_xattn import disc_rtg, coop_welfare

N, T, M = 6, 100, 3
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]
DMAX = 2


class XAttnMS(nn.Module):
    out: int
    hidden: int = 64

    @nn.compact
    def __call__(self, tok):                                   # (...,N,F)->(...,N,out)
        d = self.hidden
        Q, K, V = nn.Dense(d)(tok), nn.Dense(d)(tok), nn.Dense(d)(tok)
        a = nn.softmax(Q @ jnp.swapaxes(K, -1, -2) / jnp.sqrt(d), -1)
        ctx = a @ V
        h = nn.tanh(nn.Dense(d)(jnp.concatenate([tok, ctx], -1)))
        return nn.Dense(self.out)(h)


def features_ms(u, ccs, t, T, M):
    """Per-agent token: utility stats + per-server claim-rates (B,N,5+M)."""
    mean = u.mean(-1, keepdims=True); mn = u.min(-1, keepdims=True)
    cr = ccs / jnp.clip(t, 1.0, None)                          # (B,N,M)
    base = jnp.stack([u / T, (u - mean) / T, (u - mn) / T,
                      (u == mn).astype(jnp.float32),
                      jnp.full_like(u, t / T)], -1)            # (B,N,5)
    return jnp.concatenate([base, cr], -1)


def multi_server_step(u, action, c, M):
    """u (B,N); action (B,N) in {0..M} (0=yield, j=claim server j)."""
    B, n = u.shape
    add = jnp.zeros((B, n))
    empty = jnp.zeros((B,), jnp.int32)
    for j in range(1, M + 1):
        cj = (action == j).astype(jnp.float32)
        cnt = cj.sum(-1)
        share = jnp.where(cnt == 1, 1.0,
                          jnp.where(cnt >= 2, (1.0 - c) / jnp.clip(cnt, 1, None),
                                    0.0))
        add = add + cj * share[:, None]
        empty = empty + (cnt == 0).astype(jnp.int32)
    rank = jnp.argsort(jnp.argsort(u, -1), -1)                 # 0 = lowest u
    add_empty = (rank < empty[:, None]).astype(jnp.float32)    # neediest get empties
    return u + add + add_empty


def _ccs_update(ccs, action, M):
    oh = jax.nn.one_hot(jnp.clip(action - 1, 0, M - 1), M)
    return ccs + oh * (action > 0)[..., None].astype(jnp.float32)


def evaluate_ms(uF, def_mask, T, M):
    u = np.asarray(uF); dm = np.asarray(def_mask).astype(bool)
    B, n = u.shape
    eff = float(u.sum(1).mean()) / (T * M)                     # max M/step
    coop_u = np.where(dm, np.nan, u)
    cs = np.nan_to_num(coop_u, nan=0.0)
    nc = (~dm).sum(1)
    jc = (cs.sum(1) ** 2) / (np.clip(nc, 1, None) * np.clip((cs ** 2).sum(1), 1e-9, None))
    total = u.sum(1); n_def = dm.sum(1); u_def = (u * dm).sum(1)
    has = n_def > 0
    rho_env = (n * u_def) / np.clip(n_def * total, 1e-9, None)
    rho = float(np.mean(rho_env[has])) if has.any() else float("nan")
    return rho, float(np.nanmean(jc)), eff


def _logp(pol, p, toks, acts):
    lsm = jax.nn.log_softmax(pol.apply(p, toks))
    return jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]


def _coop_vs_pool(coop_p, pool_stk, c, B, iters, lr, key, dmax=DMAX):
    pol = XAttnMS(M + 1)
    P = jax.tree_util.tree_leaves(pool_stk)[0].shape[0]
    tx = optax.adam(lr); opt = tx.init(coop_p)

    def rollout(coop_p, def_mask, pidx, key):
        dm = def_mask.astype(bool)
        def step(carry, t):
            u, ccs, key = carry
            tok = features_ms(u, ccs, t, T, M)
            cl = pol.apply(coop_p, tok)
            dl_all = jax.vmap(lambda dp: pol.apply(dp, tok))(pool_stk)
            dl = dl_all[pidx, jnp.arange(B)].at[..., 0].set(-1e9)   # defector claims
            key, ka, kb = jax.random.split(key, 3)
            acts = jnp.where(dm, jax.random.categorical(kb, dl),
                             jax.random.categorical(ka, cl))
            u = multi_server_step(u, acts, c, M)
            return (u, _ccs_update(ccs, acts, M), key), (tok, acts, u)
        (uF, _, _), out = jax.lax.scan(
            step, (jnp.zeros((B, N)), jnp.zeros((B, N, M)), key), jnp.arange(T))
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
    pol = XAttnMS(M + 1)
    key = jax.random.PRNGKey(seed); key, ki = jax.random.split(key)
    dp = pol.init(ki, jnp.zeros((1, N, 5 + M))); tx = optax.adam(lr); opt = tx.init(dp)

    def rollout(dp, d_idx, key):
        d_oh = jax.nn.one_hot(d_idx, N)
        def step(carry, t):
            u, ccs, key = carry
            tok = features_ms(u, ccs, t, T, M)
            cl = pol.apply(coop_p, tok)
            dl = pol.apply(dp, tok).at[..., 0].set(-1e9)       # defector claims
            key, ka, kb = jax.random.split(key, 3)
            acts = jnp.where(d_oh.astype(bool), jax.random.categorical(kb, dl),
                             jax.random.categorical(ka, cl))
            u = multi_server_step(u, acts, c, M)
            return (u, _ccs_update(ccs, acts, M), key), (tok, acts)
        (uF, _, _), (toks, acts) = jax.lax.scan(
            step, (jnp.zeros((B, N)), jnp.zeros((B, N, M)), key), jnp.arange(T))
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
    rho, _, eff = evaluate_ms(uF, jax.nn.one_hot(d_idx, N), T, M)
    return rho, eff


def league_train(c, generations=5, coop_iters=1000, br_iters=800, B=512, lr=3e-3,
                 seed=0):
    pol = XAttnMS(M + 1)
    key = jax.random.PRNGKey(seed); key, kc, kd = jax.random.split(key, 3)
    coop_p = pol.init(kc, jnp.zeros((1, N, 5 + M)))
    pool = [pol.init(kd, jnp.zeros((1, N, 5 + M)))]
    for gen in range(generations):
        pool_stk = jax.tree.map(lambda *xs: jnp.stack(xs), *pool)
        key, k1 = jax.random.split(key)
        coop_p = _coop_vs_pool(coop_p, pool_stk, c, B, coop_iters, lr, k1)
        new_def = br_audit(coop_p, c, B=B, iters=br_iters, seed=seed * 100 + gen,
                           return_params=True)
        pool.append(new_def)
    return coop_p


def d0_eff(coop_p, c, B=512, seed=321):
    pol = XAttnMS(M + 1)
    def step(carry, t):
        u, ccs, key = carry
        tok = features_ms(u, ccs, t, T, M)
        key, ka = jax.random.split(key)
        acts = jax.random.categorical(ka, pol.apply(coop_p, tok))
        u = multi_server_step(u, acts, c, M)
        return (u, _ccs_update(ccs, acts, M), key), None
    (uF, _, _), _ = jax.lax.scan(
        step, (jnp.zeros((B, N)), jnp.zeros((B, N, M)), jax.random.PRNGKey(seed)),
        jnp.arange(T))
    return evaluate_ms(uF, jnp.zeros((B, N)), T, M)[2]


def selftest():
    u = jnp.array([[2.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    # agents 0,1 both claim server 1 (contest), c=0.5 -> each (1-c)/2=0.25;
    # servers 2,3 empty -> route to two neediest (agents with u=0)
    a = jnp.array([[1, 1, 0, 0, 0, 0]])
    print("contest s1 + 2 empties:", np.asarray(multi_server_step(u, a, 0.5, 3))[0])
    # agent0 sole-claims s1 (gets 1); s2,s3 empty -> neediest
    a2 = jnp.array([[1, 0, 0, 0, 0, 0]])
    print("sole s1 + 3 empties:", np.asarray(multi_server_step(u, a2, 0.5, 3))[0])


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
    out = "results/congestion_can.csv"
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
            print(f"[congestion CAN] c={c} s={s}: d0_eff={e0:.3f} d1_eff={e1:.3f} "
                  f"br_rho={rho:.2f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
