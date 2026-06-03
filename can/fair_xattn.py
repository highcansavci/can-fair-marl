"""Decentralized cross-attention fair policy for the graded-contention game,
with an UNKNOWN, VARIABLE number of free-riders --- the setting where attention
is load-bearing.

Each episode, every agent is independently a DEFECTOR with probability p (so the
number of free-riders D is variable and unknown, including D=0). Defectors always
claim; cooperators share a cross-attention policy and must (i) INFER how many
free-riders are present from observed behavior (utilities + running claim-rates),
and (ii) respond PROPORTIONALLY: turn-take (no waste) when D=0, contest just enough
when D>=1. Fixed baselines cannot do both --- all-contest wastes c even at D=0,
yield collapses at D>=1 --- so beating them requires inference + coordination,
which the attention provides. Decentralized execution: agents act from observed
public state; no central allocator decides the allocation.

Reward = masked mean(coop) - std(coop): contesting a free-rider raises cooperators'
mean (reclaimed utility); turn-taking lowers their std; doing nothing when D=0
avoids waste. Metric (exploitability audit): defector free-ride rho, Jain over
cooperators, efficiency = delivered/T --- evaluated at forced D=0 and D=1 and
compared to scripted baselines and the centralized EKREM upper bound.

    python -m can.fair_xattn --N 6 --c 0.5 --p 0.2 --iters 1200
"""
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from .fair_graded import graded_step


class XAttn(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, tok):                                    # tok (...,N,F)
        d = self.hidden
        Q, K, V = nn.Dense(d)(tok), nn.Dense(d)(tok), nn.Dense(d)(tok)
        a = nn.softmax(Q @ jnp.swapaxes(K, -1, -2) / jnp.sqrt(d), -1)
        ctx = a @ V
        h = nn.tanh(nn.Dense(d)(jnp.concatenate([tok, ctx], -1)))
        return nn.Dense(2)(h)                                   # (...,N,2)


def features(u, cc, t, T):
    mean = u.mean(-1, keepdims=True); mn = u.min(-1, keepdims=True)
    cr = cc / jnp.clip(t, 1.0, None)
    return jnp.stack([u / T, (u - mean) / T, (u - mn) / T,
                      (u == mn).astype(jnp.float32), cr,
                      jnp.full_like(u, t / T)], -1)


def coop_welfare(uF, def_mask):
    coop = 1.0 - def_mask
    nc = jnp.clip(coop.sum(-1), 1.0, None)
    mean_c = (uF * coop).sum(-1) / nc
    var_c = (((uF - mean_c[:, None]) ** 2) * coop).sum(-1) / nc
    return mean_c - jnp.sqrt(var_c + 1e-9)


def make_rollout(N, T, B, c):
    pol = XAttn()

    def rollout(params, def_mask, key):                        # def_mask (B,N) {0,1}
        dm = def_mask.astype(bool)

        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            logits = pol.apply(params, tok)
            key, ka = jax.random.split(key)
            acts = jax.random.categorical(ka, logits)
            acts = jnp.where(dm, 1, acts)                      # defectors always claim
            claim = acts == 1
            u = graded_step(u, claim, c)
            return (u, cc + claim, key), (tok, acts, u)
        (uF, _, _), (toks, acts, useq) = jax.lax.scan(
            step, (jnp.zeros((B, N)), jnp.zeros((B, N)), key), jnp.arange(T))
        return uF, toks, acts, useq                            # useq (T,B,N)

    return pol, rollout


def disc_rtg(r, gamma=0.99):                                   # r (T,B) -> (T,B)
    def f(carry, x):
        g = x + gamma * carry
        return g, g
    _, G = jax.lax.scan(f, jnp.zeros(r.shape[1:]), r, reverse=True)
    return G


def evaluate(uF, def_mask, T):
    u = np.asarray(uF); dm = np.asarray(def_mask).astype(bool)
    B, N = u.shape
    eff = float(u.sum(1).mean()) / T
    coop_u = np.where(dm, np.nan, u)
    cs = np.nan_to_num(coop_u, nan=0.0)
    nc = (~dm).sum(1)
    jc = (cs.sum(1) ** 2) / (np.clip(nc, 1, None) * np.clip((cs ** 2).sum(1), 1e-9, None))
    # bounded free-ride factor: defector group's share of the delivered total,
    # normalized by its fair share (n_def/N). rho in [0, N/n_def]; robust at all c
    # since total>0 always (the old u_def/mean_coop blew up when cooperators were
    # starved near c->1). rho=1 = fair; rho=N = a lone defector takes everything.
    total = u.sum(1)
    n_def = dm.sum(1)
    u_def = (u * dm).sum(1)
    has = n_def > 0
    rho_env = (N * u_def) / np.clip(n_def * total, 1e-9, None)
    rho = float(np.mean(rho_env[has])) if has.any() else float("nan")
    return rho, float(np.nanmean(jc)), eff


class Critic(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, tok):                                   # (...,N,F) -> (...)
        pooled = jnp.concatenate([tok.mean(-2), tok.min(-2), tok.max(-2)], -1)
        h = nn.tanh(nn.Dense(self.hidden)(pooled))
        h = nn.tanh(nn.Dense(self.hidden)(h))
        return nn.Dense(1)(h)[..., 0]


def cotrain_ac(N, T, c, B=512, iters=3000, lr=3e-3, seed=0, dmax=2, vf=0.5):
    """Co-training with a learned VALUE-FUNCTION BASELINE (actor-critic) for the
    cooperators, to cut policy-gradient variance vs the plain batch-mean baseline.
    Returns cooperator params."""
    pol, crt = XAttn(), Critic()
    key = jax.random.PRNGKey(seed)
    key, kc, kv, kd = jax.random.split(key, 4)
    coop_p = pol.init(kc, jnp.zeros((1, N, 6)))
    crt_p = crt.init(kv, jnp.zeros((1, N, 6)))
    def_p = pol.init(kd, jnp.zeros((1, N, 6)))
    txc = optax.adam(lr); optc = txc.init(coop_p)
    txv = optax.adam(lr); optv = txv.init(crt_p)
    txd = optax.adam(lr); optd = txd.init(def_p)

    def rollout(coop_p, def_p, def_mask, key):
        dm = def_mask.astype(bool)
        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            cl, dl = pol.apply(coop_p, tok), pol.apply(def_p, tok)
            key, ka, kb = jax.random.split(key, 3)
            acts = jnp.where(dm, jax.random.categorical(kb, dl),
                             jax.random.categorical(ka, cl))
            u = graded_step(u, acts == 1, c)
            return (u, cc + (acts == 1), key), (tok, acts, u)
        (uF, _, _), out = jax.lax.scan(
            step, (jnp.zeros((B, N)), jnp.zeros((B, N)), key), jnp.arange(T))
        return out                                            # toks,acts,useq

    @jax.jit
    def upd(coop_p, crt_p, def_p, optc, optv, optd, key, ec):
        key, ks, kr = jax.random.split(key, 3)
        Dn = jax.random.randint(ks, (B,), 0, dmax + 1)
        rank = jnp.argsort(jnp.argsort(
            jax.random.uniform(jax.random.fold_in(ks, 1), (B, N)), -1), -1)
        def_mask = (rank < Dn[:, None]).astype(jnp.float32)
        coop = 1.0 - def_mask
        toks, acts, useq = rollout(coop_p, def_p, def_mask, kr)
        Wc = jax.vmap(coop_welfare, in_axes=(0, None))(useq, def_mask)
        G = disc_rtg(Wc - jnp.concatenate([jnp.zeros((1, B)), Wc[:-1]], 0))  # (T,B)
        Wd = (useq * def_mask[None]).sum(-1)
        ad = disc_rtg(Wd - jnp.concatenate([jnp.zeros((1, B)), Wd[:-1]], 0))
        ad = ad - ad.mean(1, keepdims=True)
        V = crt.apply(crt_p, toks)                            # (T,B) value baseline
        adv = G - V                                           # state-dependent advantage

        def closs(p):
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
            ent = -(jnp.exp(lsm) * lsm).sum(-1)
            return -(jax.lax.stop_gradient(adv)[:, :, None] * lp * coop[None]).sum(-1).mean() \
                - ec * (ent * coop[None]).sum(-1).mean()
        def vloss(p):
            return vf * ((crt.apply(p, toks) - jax.lax.stop_gradient(G)) ** 2).mean()
        def dloss(p):
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
            return -(ad[:, :, None] * lp * def_mask[None]).sum(-1).mean()
        gc = jax.grad(closs)(coop_p); uc, optc = txc.update(gc, optc)
        gv = jax.grad(vloss)(crt_p); uv, optv = txv.update(gv, optv)
        gd = jax.grad(dloss)(def_p); ud, optd = txd.update(gd, optd)
        return (optax.apply_updates(coop_p, uc), optax.apply_updates(crt_p, uv),
                optax.apply_updates(def_p, ud), optc, optv, optd, key)

    ent_hi, ent_lo = 0.05, 0.003
    for it in range(iters):
        ec = ent_hi * (1 - it / iters) + ent_lo * (it / iters)
        coop_p, crt_p, def_p, optc, optv, optd, key = upd(
            coop_p, crt_p, def_p, optc, optv, optd, key, ec)
    return coop_p


def cotrain(N, T, c, B=512, iters=3000, lr=3e-3, seed=0, dmax=2):
    """Adversarial co-training: cooperators (shared XAttn, maximize coop welfare)
    AND a defector policy (shared XAttn, maximize defectors' own utility) are
    trained TOGETHER, with D~unif{0..dmax} co-evolving defectors per episode. The
    cooperators thus learn to handle an adaptive adversary, not just always-claim.
    Returns the trained cooperator params."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    key, kc, kd = jax.random.split(key, 3)
    coop_p = pol.init(kc, jnp.zeros((1, N, 6)))
    def_p = pol.init(kd, jnp.zeros((1, N, 6)))
    txc = optax.adam(lr); optc = txc.init(coop_p)
    txd = optax.adam(lr); optd = txd.init(def_p)

    def rollout(coop_p, def_p, def_mask, key):
        dm = def_mask.astype(bool)
        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            cl, dl = pol.apply(coop_p, tok), pol.apply(def_p, tok)
            key, ka, kb = jax.random.split(key, 3)
            acts = jnp.where(dm, jax.random.categorical(kb, dl),
                             jax.random.categorical(ka, cl))
            u = graded_step(u, acts == 1, c)
            return (u, cc + (acts == 1), key), (tok, acts, u)
        (uF, _, _), (toks, acts, useq) = jax.lax.scan(
            step, (jnp.zeros((B, N)), jnp.zeros((B, N)), key), jnp.arange(T))
        return toks, acts, useq

    @jax.jit
    def upd(coop_p, def_p, optc, optd, key, ec):
        key, ks, kr = jax.random.split(key, 3)
        Dn = jax.random.randint(ks, (B,), 0, dmax + 1)
        rank = jnp.argsort(jnp.argsort(
            jax.random.uniform(jax.random.fold_in(ks, 1), (B, N)), -1), -1)
        def_mask = (rank < Dn[:, None]).astype(jnp.float32)
        coop = 1.0 - def_mask
        toks, acts, useq = rollout(coop_p, def_p, def_mask, kr)
        # cooperator objective: per-step coop-welfare reward-to-go
        Wc = jax.vmap(coop_welfare, in_axes=(0, None))(useq, def_mask)
        ac = disc_rtg(Wc - jnp.concatenate([jnp.zeros((1, B)), Wc[:-1]], 0))
        ac = ac - ac.mean(1, keepdims=True)
        # defector objective: per-step defectors' own utility reward-to-go
        Wd = (useq * def_mask[None]).sum(-1)               # (T,B)
        ad = disc_rtg(Wd - jnp.concatenate([jnp.zeros((1, B)), Wd[:-1]], 0))
        ad = ad - ad.mean(1, keepdims=True)

        def closs(p):
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
            ent = -(jnp.exp(lsm) * lsm).sum(-1)
            return -(ac[:, :, None] * lp * coop[None]).sum(-1).mean() \
                - ec * (ent * coop[None]).sum(-1).mean()
        def dloss(p):
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
            return -(ad[:, :, None] * lp * def_mask[None]).sum(-1).mean()
        gc = jax.grad(closs)(coop_p); uc, optc = txc.update(gc, optc)
        gd = jax.grad(dloss)(def_p); ud, optd = txd.update(gd, optd)
        return (optax.apply_updates(coop_p, uc), optax.apply_updates(def_p, ud),
                optc, optd, key)

    ent_hi, ent_lo = 0.05, 0.003
    for it in range(iters):
        ec = ent_hi * (1 - it / iters) + ent_lo * (it / iters)
        coop_p, def_p, optc, optd, key = upd(coop_p, def_p, optc, optd, key, ec)
    return coop_p


def cotrain_pop(N, T, c, K=4, B=512, iters=3000, lr=3e-3, seed=0, dmax=2):
    """Population adversarial co-training: cooperators face a POPULATION of K
    co-evolving defector policies (one assigned per episode). Diversity prevents
    the cooperators from overfitting to a single adversary equilibrium (fixes the
    seed-sensitivity of single-adversary co-training). Returns cooperator params."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    key, kc, kd = jax.random.split(key, 3)
    coop_p = pol.init(kc, jnp.zeros((1, N, 6)))
    def_ps = jax.vmap(lambda k: pol.init(k, jnp.zeros((1, N, 6))))(
        jax.random.split(kd, K))                           # stacked K policies
    txc = optax.adam(lr); optc = txc.init(coop_p)
    txd = optax.adam(lr); optd = txd.init(def_ps)

    def lp_under(prm, toks, acts):
        lsm = jax.nn.log_softmax(pol.apply(prm, toks))
        return jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]

    def rollout(coop_p, def_ps, def_mask, kidx, key):
        dm = def_mask.astype(bool)
        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            cl = pol.apply(coop_p, tok)
            dl_all = jax.vmap(lambda dp: pol.apply(dp, tok))(def_ps)  # (K,B,N,2)
            dl = dl_all[kidx, jnp.arange(B)]                # (B,N,2) assigned defector
            key, ka, kb = jax.random.split(key, 3)
            acts = jnp.where(dm, jax.random.categorical(kb, dl),
                             jax.random.categorical(ka, cl))
            u = graded_step(u, acts == 1, c)
            return (u, cc + (acts == 1), key), (tok, acts, u)
        (uF, _, _), (toks, acts, useq) = jax.lax.scan(
            step, (jnp.zeros((B, N)), jnp.zeros((B, N)), key), jnp.arange(T))
        return toks, acts, useq

    @jax.jit
    def upd(coop_p, def_ps, optc, optd, key, ec):
        key, ks, kk, kr = jax.random.split(key, 4)
        Dn = jax.random.randint(ks, (B,), 0, dmax + 1)
        rank = jnp.argsort(jnp.argsort(
            jax.random.uniform(jax.random.fold_in(ks, 1), (B, N)), -1), -1)
        def_mask = (rank < Dn[:, None]).astype(jnp.float32)
        kidx = jax.random.randint(kk, (B,), 0, K)
        coop = 1.0 - def_mask
        toks, acts, useq = rollout(coop_p, def_ps, def_mask, kidx, kr)
        Wc = jax.vmap(coop_welfare, in_axes=(0, None))(useq, def_mask)
        ac = disc_rtg(Wc - jnp.concatenate([jnp.zeros((1, B)), Wc[:-1]], 0))
        ac = ac - ac.mean(1, keepdims=True)
        Wd = (useq * def_mask[None]).sum(-1)
        ad = disc_rtg(Wd - jnp.concatenate([jnp.zeros((1, B)), Wd[:-1]], 0))
        ad = ad - ad.mean(1, keepdims=True)

        def closs(p):
            lp = lp_under(p, toks, acts)
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            ent = -(jnp.exp(lsm) * lsm).sum(-1)
            return -(ac[:, :, None] * lp * coop[None]).sum(-1).mean() \
                - ec * (ent * coop[None]).sum(-1).mean()
        def dloss(dps):
            lp_all = jax.vmap(lambda dp: lp_under(dp, toks, acts))(dps)  # (K,T,B,N)
            selk = (jnp.arange(K)[:, None] == kidx[None, :]).astype(jnp.float32)
            return -(ad[None, :, :, None] * lp_all * def_mask[None, None]
                     * selk[:, None, :, None]).sum() / (B * T)
        gc = jax.grad(closs)(coop_p); uc, optc = txc.update(gc, optc)
        gd = jax.grad(dloss)(def_ps); ud, optd = txd.update(gd, optd)
        return (optax.apply_updates(coop_p, uc), optax.apply_updates(def_ps, ud),
                optc, optd, key)

    ent_hi, ent_lo = 0.05, 0.003
    for it in range(iters):
        ec = ent_hi * (1 - it / iters) + ent_lo * (it / iters)
        coop_p, def_ps, optc, optd, key = upd(coop_p, def_ps, optc, optd, key, ec)
    return coop_p


def best_response_audit(coop_params, N, T, c, B=512, iters=1500, lr=3e-3, seed=7,
                        return_params=False, return_eff=False):
    """Train a BEST-RESPONSE defector (its own XAttn policy, maximizing its own
    utility) against the FROZEN cooperator policy, at a random index. Returns the
    resulting free-ride rho --- a stronger exploitability audit than always-claim."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    def_params = pol.init(ki, jnp.zeros((1, N, 6)))
    tx = optax.adam(lr); opt = tx.init(def_params)

    def rollout(def_p, d_idx, key):
        d_oh = jax.nn.one_hot(d_idx, N)

        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            cl, dl = pol.apply(coop_params, tok), pol.apply(def_p, tok)
            key, ka, kb = jax.random.split(key, 3)
            ca = jax.random.categorical(ka, cl)
            da = jax.random.categorical(kb, dl)
            acts = jnp.where(d_oh.astype(bool), da, ca)
            u = graded_step(u, acts == 1, c)
            return (u, cc + (acts == 1), key), (tok, acts)
        (uF, _, _), (toks, acts) = jax.lax.scan(
            step, (jnp.zeros((B, N)), jnp.zeros((B, N)), key), jnp.arange(T))
        return uF, toks, acts, d_oh

    @jax.jit
    def upd(def_p, opt, key):
        key, kd, kr = jax.random.split(key, 3)
        d_idx = jax.random.randint(kd, (B,), 0, N)
        uF, toks, acts, d_oh = rollout(def_p, d_idx, kr)
        R = (uF * d_oh).sum(-1); adv = R - R.mean()           # defector's own utility
        def loss(p):
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
            return -(adv[None, :, None] * lp * d_oh[None]).sum(-1).mean()
        g = jax.grad(loss)(def_p); u, opt = tx.update(g, opt)
        return optax.apply_updates(def_p, u), opt, key

    for it in range(iters):
        def_params, opt, key = upd(def_params, opt, key)
    if return_params:
        return def_params
    key, kd, kr = jax.random.split(key, 3)
    d_idx = jax.random.randint(kd, (B,), 0, N)
    uF, _, _, _ = rollout(def_params, d_idx, kr)
    rho, _, eff = evaluate(uF, jax.nn.one_hot(d_idx, N), T)
    return (rho, eff) if return_eff else rho                   # D>=1 efficiency


def _coop_vs_pool(coop_p, pool_stk, N, T, c, B, iters, lr, key, dmax=2):
    """Train cooperators (REINFORCE + entropy anneal) against a FROZEN pool of
    defector policies (stacked, leading dim P): a random pool member is assigned
    per episode. Returns updated cooperator params."""
    pol = XAttn()
    P = jax.tree_util.tree_leaves(pool_stk)[0].shape[0]
    tx = optax.adam(lr); opt = tx.init(coop_p)

    def rollout(coop_p, def_mask, pidx, key):
        dm = def_mask.astype(bool)
        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            cl = pol.apply(coop_p, tok)
            dl_all = jax.vmap(lambda dp: pol.apply(dp, tok))(pool_stk)  # (P,B,N,2)
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


def league_train(N, T, c, generations=6, coop_iters=1200, br_iters=1000,
                 B=512, lr=3e-3, seed=0):
    """PSRO / fictitious-play: alternate (train cooperators vs the frozen pool of
    past best-response defectors) and (add a fresh best-response defector to the
    pool). Targets the multi-equilibrium cause of c=0.3 instability. Returns the
    final cooperator params."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    key, kc, kd = jax.random.split(key, 3)
    coop_p = pol.init(kc, jnp.zeros((1, N, 6)))
    pool = [pol.init(kd, jnp.zeros((1, N, 6)))]                # start: one defector
    for gen in range(generations):
        pool_stk = jax.tree.map(lambda *xs: jnp.stack(xs), *pool)
        key, k1, k2 = jax.random.split(key, 3)
        coop_p = _coop_vs_pool(coop_p, pool_stk, N, T, c, B, coop_iters, lr, k1)
        new_def = best_response_audit(coop_p, N, T, c, B=B, iters=br_iters,
                                      seed=seed * 100 + gen, return_params=True)
        pool.append(new_def)
    return coop_p


def train(N=6, T=100, B=512, c=0.5, p=0.2, iters=1200, lr=3e-3, seed=0,
          ent_coef=0.02, dmax=2):
    pol, rollout = make_rollout(N, T, B, c)
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    params = pol.init(ki, jnp.zeros((1, N, 6)))
    tx = optax.adam(lr); opt = tx.init(params)

    @jax.jit
    def upd(params, opt, key, ec):
        key, kd, ks, kr = jax.random.split(key, 4)
        Dn = jax.random.randint(kd, (B,), 0, dmax + 1)         # D ~ unif{0..dmax}
        rank = jnp.argsort(jnp.argsort(jax.random.uniform(ks, (B, N)), -1), -1)
        def_mask = (rank < Dn[:, None]).astype(jnp.float32)    # exactly Dn defectors
        uF, toks, acts, useq = rollout(params, def_mask, kr)
        coop = 1.0 - def_mask
        W = jax.vmap(coop_welfare, in_axes=(0, None))(useq, def_mask)  # (T,B)
        Wprev = jnp.concatenate([jnp.zeros((1, B)), W[:-1]], 0)
        adv_t = disc_rtg(W - Wprev)                            # (T,B) reward-to-go
        adv_t = adv_t - adv_t.mean(1, keepdims=True)
        def loss(prm):
            logits = pol.apply(prm, toks)                      # (T,B,N,2)
            lsm = jax.nn.log_softmax(logits)
            lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
            ent = -(jnp.exp(lsm) * lsm).sum(-1)
            pg = -(adv_t[:, :, None] * lp * coop[None]).sum(-1).mean()
            ent_term = (ent * coop[None]).sum(-1).mean()
            return pg - ec * ent_term
        g = jax.grad(loss)(params); u, opt = tx.update(g, opt)
        return optax.apply_updates(params, u), opt, key

    ent_hi, ent_lo = 0.05, 0.003
    for it in range(iters):
        ec = ent_hi * (1 - it / iters) + ent_lo * (it / iters)
        params, opt, key = upd(params, opt, key, ec)

    # eval at forced D=0 (no defector) and D=1 (one random index)
    key, k0, k1, kr0, kr1 = jax.random.split(key, 5)
    m0 = jnp.zeros((B, N))
    di = jax.random.randint(k1, (B,), 0, N)
    m1 = jax.nn.one_hot(di, N)
    e0 = evaluate(rollout(params, m0, kr0)[0], m0, T)
    e1 = evaluate(rollout(params, m1, kr1)[0], m1, T)
    return e0, e1, params


def eval_at(params, N, T, c, B=512, seed=123):
    """Zero-shot eval of an (N-agnostic) attention policy at team size N."""
    _, rollout = make_rollout(N, T, B, c)
    key = jax.random.PRNGKey(seed)
    k0, k1, kr0, kr1 = jax.random.split(key, 4)
    m0 = jnp.zeros((B, N))
    m1 = jax.nn.one_hot(jax.random.randint(k1, (B,), 0, N), N)
    e0 = evaluate(rollout(params, m0, kr0)[0], m0, T)
    e1 = evaluate(rollout(params, m1, kr1)[0], m1, T)
    return e0, e1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--p", type=float, default=0.2)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--ent", type=float, default=0.02)
    ap.add_argument("--dmax", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap_eval = ap  # noqa
    a = ap.parse_args()
    (r0, j0, ef0), (r1, j1, ef1), params = train(
        N=a.N, T=a.T, c=a.c, p=a.p, iters=a.iters, seed=a.seed, ent_coef=a.ent,
        dmax=a.dmax)
    print(f"\n=== decentralized cross-attention [trained at N={a.N}, c={a.c}] ===")
    print(f"  {'policy':>16} | {'D=0 eff':>8} | {'D=1 rho':>8} | {'D=1 Jain':>9} | {'D=1 eff':>8}")
    print(f"  {'X-attn (ours)':>16} | {ef0:>8.2f} | {r1:>8.2f} | {j1:>9.2f} | {ef1:>8.2f}")
    print(f"  {'all-contest':>16} | {1-a.c:>8.2f} | {1.0:>8.2f} | {1.0:>9.2f} | {1-a.c:>8.2f}")
    print(f"  {'EKREM (central)':>16} | {1.0:>8.2f} | {1.0:>8.2f} | {1.0:>9.2f} | {1.0:>8.2f}")
    print(f"\n  ZERO-SHOT TRANSFER (same N={a.N} policy applied at larger team sizes):")
    print(f"  {'N_eval':>8} | {'D=0 eff':>8} | {'D=1 rho':>8} | {'D=1 Jain':>9}")
    for Ne in [a.N, 2 * a.N, 4 * a.N]:
        (te0r, te0j, te0e), (te1r, te1j, te1e) = eval_at(params, Ne, a.T, a.c)
        print(f"  {Ne:>8} | {te0e:>8.2f} | {te1r:>8.2f} | {te1j:>9.2f}")


if __name__ == "__main__":
    main()
