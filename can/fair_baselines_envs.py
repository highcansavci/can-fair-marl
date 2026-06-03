"""Welfare-fair baselines (GGF, FEN, SOTO) on the second/third environments
(congestion, stakes), under the SAME bounded best-response audit as CAN, so the
head-to-head (efficient-and-robust corner) can be replicated per environment.

One generic implementation driven by per-env hooks (Env). Baselines train
cooperatively (no defector at train time, as in their originals) with a shared
per-agent MLP over the same features CAN sees; we then freeze and train a
best-response defector.

    python -m can.fair_baselines_envs --env congestion
    python -m can.fair_baselines_envs --env stakes
"""
import os
import csv
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from .fair_xattn import evaluate
from .fair_train import ggf
from . import fair_congestion as cong
from . import fair_stakes as stk

N, T = 6, 100
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]
DMAX = 2
FE_C0 = 0.1


class MLP(nn.Module):
    out: int
    hidden: int = 64

    @nn.compact
    def __call__(self, tok):
        h = nn.tanh(nn.Dense(self.hidden)(tok))
        h = nn.tanh(nn.Dense(self.hidden)(h))
        return nn.Dense(self.out)(h)


# --------------------------- per-env hooks -------------------------------------
class Env:
    def __init__(self, name):
        self.name = name
        if name == "congestion":
            self.M = cong.M
            self.obs_dim, self.act_dim = 5 + cong.M, cong.M + 1
            self.has_v, self.def_claims_only = False, True
        elif name == "stakes":
            self.obs_dim, self.act_dim = 7, 2
            self.has_v, self.def_claims_only = True, False
        else:
            raise ValueError(name)

    def init_aux(self, B):
        return jnp.zeros((B, N, self.M)) if self.name == "congestion" \
            else jnp.zeros((B, N))

    def sample_v(self, key, B):
        return stk.sample_v(key, B) if self.has_v else None

    def feats(self, u, aux, t, v):
        if self.name == "congestion":
            return cong.features_ms(u, aux, t, T, self.M)
        return stk.features_stakes(u, aux, t, T, v)

    def step(self, u, acts, c, v):
        if self.name == "congestion":
            return cong.multi_server_step(u, acts, c, self.M)
        return stk.value_step(u, acts == 1, c, v)

    def aux_up(self, aux, acts):
        if self.name == "congestion":
            return cong._ccs_update(aux, acts, self.M)
        return aux + (acts == 1)

    def eval(self, uF, def_mask):
        if self.name == "congestion":
            return cong.evaluate_ms(uF, def_mask, T, self.M)
        return evaluate(uF, def_mask, T)


def _coop_rollout(env, act_sampler, c, B, key):
    """All-agent rollout under act_sampler(tok,key)->(acts, m). m = SOTO use_so mask
    (or ones)."""
    def step(carry, t):
        u, aux, key = carry
        key, kv, ka = jax.random.split(key, 3)
        v = env.sample_v(kv, B)
        tok = env.feats(u, aux, t, v)
        acts, m = act_sampler(tok, ka)
        u = env.step(u, acts, c, v)
        return (u, env.aux_up(aux, acts), key), (tok, acts, m, u)
    (uF, _, _), out = jax.lax.scan(
        step, (jnp.zeros((B, N)), env.init_aux(B), key), jnp.arange(T))
    return uF, out


def _logp(pol, p, toks, acts):
    lsm = jax.nn.log_softmax(pol.apply(p, toks))
    return jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]


def train_ggf(env, c, B=512, iters=2000, lr=3e-3, seed=0):
    pol = MLP(env.act_dim); key = jax.random.PRNGKey(seed); key, ki = jax.random.split(key)
    p = pol.init(ki, jnp.zeros((1, N, env.obs_dim))); tx = optax.adam(lr); opt = tx.init(p)

    @jax.jit
    def upd(p, opt, key):
        key, kr = jax.random.split(key)
        def samp(tok, k):
            return jax.random.categorical(k, pol.apply(p, tok)), jnp.ones(tok.shape[:-1])
        uF, (toks, acts, _, _) = _coop_rollout(env, samp, c, B, kr)
        adv = ggf(uF) - ggf(uF).mean()
        def loss(pp):
            return -(adv[None, :, None] * _logp(pol, pp, toks, acts)).mean()
        g = jax.grad(loss)(p); u, opt = tx.update(g, opt)
        return optax.apply_updates(p, u), opt, key
    for _ in range(iters):
        p, opt, key = upd(p, opt, key)
    return pol, p


def train_fen(env, c, B=512, iters=2000, lr=3e-3, seed=0):
    pol = MLP(env.act_dim); key = jax.random.PRNGKey(seed); key, ki = jax.random.split(key)
    p = pol.init(ki, jnp.zeros((1, N, env.obs_dim))); tx = optax.adam(lr); opt = tx.init(p)

    @jax.jit
    def upd(p, opt, key):
        key, kr = jax.random.split(key)
        def samp(tok, k):
            return jax.random.categorical(k, pol.apply(p, tok)), jnp.ones(tok.shape[:-1])
        uF, (toks, acts, _, _) = _coop_rollout(env, samp, c, B, kr)
        mean = uF.mean(-1, keepdims=True)
        fe = mean / (FE_C0 + jnp.abs(uF - mean))
        adv = fe - fe.mean(0, keepdims=True)
        def loss(pp):
            return -(adv[None] * _logp(pol, pp, toks, acts)).mean()
        g = jax.grad(loss)(p); u, opt = tx.update(g, opt)
        return optax.apply_updates(p, u), opt, key
    for _ in range(iters):
        p, opt, key = upd(p, opt, key)
    return pol, p


def train_soto(env, c, B=512, iters=2000, lr=3e-3, seed=0):
    pol = MLP(env.act_dim); key = jax.random.PRNGKey(seed)
    key, ks, kt = jax.random.split(key, 3)
    so_p = pol.init(ks, jnp.zeros((1, N, env.obs_dim)))
    to_p = pol.init(kt, jnp.zeros((1, N, env.obs_dim)))
    txs = optax.adam(lr); opts = txs.init(so_p)
    txt = optax.adam(lr); optt = txt.init(to_p)
    anneal = int(0.7 * iters)

    @jax.jit
    def upd(so_p, to_p, opts, optt, beta, key):
        key, kr = jax.random.split(key)
        def samp(tok, k):
            k1, k2, k3 = jax.random.split(k, 3)
            use_so = jax.random.bernoulli(k1, beta, tok.shape[:-1])
            a = jnp.where(use_so, jax.random.categorical(k2, pol.apply(so_p, tok)),
                          jax.random.categorical(k3, pol.apply(to_p, tok)))
            return a, use_so.astype(jnp.float32)
        uF, (toks, acts, m, _) = _coop_rollout(env, samp, c, B, kr)
        own = uF - uF.mean(0, keepdims=True)
        gadv = ggf(uF) - ggf(uF).mean()
        def so_loss(pp):
            return -(own[None] * _logp(pol, pp, toks, acts) * m).mean()
        def to_loss(pp):
            return -(gadv[None, :, None] * _logp(pol, pp, toks, acts) * (1 - m)).mean()
        us, opts = txs.update(jax.grad(so_loss)(so_p), opts)
        ut, optt = txt.update(jax.grad(to_loss)(to_p), optt)
        return optax.apply_updates(so_p, us), optax.apply_updates(to_p, ut), opts, optt, key
    for it in range(iters):
        beta = max(0.0, 1.0 - it / anneal)
        so_p, to_p, opts, optt, key = upd(so_p, to_p, opts, optt, beta, key)
    return pol, to_p


def fair_eval(env, pol, p, c, B=512, seed=321):
    def samp(tok, k):
        return jax.random.categorical(k, pol.apply(p, tok)), jnp.ones(tok.shape[:-1])
    uF, _ = _coop_rollout(env, samp, c, B, jax.random.PRNGKey(seed))
    _, jn, eff = env.eval(uF, jnp.zeros((B, N)))
    return jn, eff


def br_audit(env, pol, fair_p, c, B=512, iters=1500, lr=3e-3, seed=7):
    dpol = MLP(env.act_dim); key = jax.random.PRNGKey(seed); key, ki = jax.random.split(key)
    dp = dpol.init(ki, jnp.zeros((1, N, env.obs_dim))); tx = optax.adam(lr); opt = tx.init(dp)

    def rollout(dp, d_idx, key):
        d_oh = jax.nn.one_hot(d_idx, N)
        def step(carry, t):
            u, aux, key = carry
            key, kv, ka, kb = jax.random.split(key, 4)
            v = env.sample_v(kv, B)
            tok = env.feats(u, aux, t, v)
            cl = pol.apply(fair_p, tok)
            dl = dpol.apply(dp, tok)
            if env.def_claims_only:
                dl = dl.at[..., 0].set(-1e9)
            acts = jnp.where(d_oh.astype(bool), jax.random.categorical(kb, dl),
                             jax.random.categorical(ka, cl))
            u = env.step(u, acts, c, v)
            return (u, env.aux_up(aux, acts), key), (tok, acts)
        (uF, _, _), (toks, acts) = jax.lax.scan(
            step, (jnp.zeros((B, N)), env.init_aux(B), key), jnp.arange(T))
        return uF, toks, acts, d_oh

    @jax.jit
    def upd(dp, opt, key):
        key, kd, kr = jax.random.split(key, 3)
        d_idx = jax.random.randint(kd, (B,), 0, N)
        uF, toks, acts, d_oh = rollout(dp, d_idx, kr)
        R = (uF * d_oh).sum(-1); adv = R - R.mean()
        def loss(pp):
            return -(adv[None, :, None] * _logp(dpol, pp, toks, acts)
                     * d_oh[None]).sum(-1).mean()
        g = jax.grad(loss)(dp); u, opt = tx.update(g, opt)
        return optax.apply_updates(dp, u), opt, key
    for _ in range(iters):
        dp, opt, key = upd(dp, opt, key)
    key, kd, kr = jax.random.split(key, 3)
    d_idx = jax.random.randint(kd, (B,), 0, N)
    uF, _, _, _ = rollout(dp, d_idx, kr)
    rho, _, eff = env.eval(uF, jax.nn.one_hot(d_idx, N))
    return rho, eff


TRAINERS = {"GGF": train_ggf, "FEN": train_fen, "SOTO": train_soto}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["congestion", "stakes"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--iters", type=int, default=2000)
    a = ap.parse_args()
    env = Env(a.env)
    cs, seeds = CS, SEEDS
    if a.smoke:
        cs, seeds, a.iters = [0.5], [0], 300
    os.makedirs("results", exist_ok=True)
    out = f"results/baselines_{a.env}.csv"
    if not a.smoke and os.path.exists(out):
        os.remove(out)
    header = ["method", "c", "seed", "d0_jain", "d0_eff", "d1_eff", "br_rho"]
    for meth, trainer in TRAINERS.items():
        for c in cs:
            for s in seeds:
                pol, p = trainer(env, c, iters=a.iters, seed=s)
                jn, d0 = fair_eval(env, pol, p, c)
                rho, d1 = br_audit(env, pol, p, c, iters=1500, seed=700 + s)
                row = dict(method=meth, c=c, seed=s, d0_jain=float(jn),
                           d0_eff=float(d0), d1_eff=float(d1), br_rho=float(rho))
                if not a.smoke:
                    new = not os.path.exists(out)
                    with open(out, "a", newline="") as f:
                        w = csv.DictWriter(f, header)
                        if new:
                            w.writeheader()
                        w.writerow(row); f.flush()
                print(f"[{a.env}/{meth}] c={c} s={s}: d0_eff={d0:.3f} d1_eff={d1:.3f} "
                      f"br_rho={rho:.2f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
