"""Welfare-fair baselines (GGF, FEN, SOTO) on the SAME graded-contention game and
under the SAME bounded best-response free-ride audit as CAN --- to show the
fair-MARL learners are exploitable (rho >> 1) where CAN (league) is near-fair.

All three are trained COOPERATIVELY (no defector at train time, as in their
originals), share one permutation-equivariant per-agent policy, and observe the
same 6 behaviour features CAN sees (parity --- the difference is the objective and
the absence of adversarial training, not the inputs). Then we freeze the team,
insert one best-response defector at a random index, and report the bounded rho
(Eq. in the paper; defector group's share of delivered total / fair share).

    python -m can.fair_baselines_graded            # full sweep -> CSV
    python -m can.fair_baselines_graded --smoke     # quick check
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
from .fair_xattn import features, evaluate
from .fair_train import ggf

N, T = 6, 100
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]
FE_C0 = 0.1                       # FEN fair-efficient reward stabilizer


class MLP(nn.Module):
    """Per-agent policy shared across agents (permutation-equivariant, N-agnostic),
    matching the standard fair-MARL net of FEN/SOTO."""
    hidden: int = 64

    @nn.compact
    def __call__(self, tok):                                   # (...,N,6)->(...,N,2)
        h = nn.tanh(nn.Dense(self.hidden)(tok))
        h = nn.tanh(nn.Dense(self.hidden)(h))
        return nn.Dense(2)(h)


def _coop_rollout(apply_fn, B, c, key):
    """All agents act under apply_fn(tok)->logits. Returns uF,(toks,acts,useq)."""
    def step(carry, t):
        u, cc, key = carry
        tok = features(u, cc, t, T)
        logits = apply_fn(tok)
        key, ka = jax.random.split(key)
        acts = jax.random.categorical(ka, logits)
        u = graded_step(u, acts == 1, c)
        return (u, cc + (acts == 1), key), (tok, acts, u)
    (uF, _, _), out = jax.lax.scan(
        step, (jnp.zeros((B, N)), jnp.zeros((B, N)), key), jnp.arange(T))
    return uF, out


def _logp(pol, p, toks, acts):
    lsm = jax.nn.log_softmax(pol.apply(p, toks))
    return jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]


# ----------------------------- fair learners -----------------------------------
def train_ggf(c, B=512, iters=2000, lr=3e-3, seed=0):
    """Shared-policy GGF-welfare maximizer (the simplified fair learner)."""
    pol = MLP(); key = jax.random.PRNGKey(seed); key, ki = jax.random.split(key)
    p = pol.init(ki, jnp.zeros((1, N, 6))); tx = optax.adam(lr); opt = tx.init(p)

    @jax.jit
    def upd(p, opt, key):
        key, kr = jax.random.split(key)
        uF, (toks, acts, _) = _coop_rollout(lambda t: pol.apply(p, t), B, c, kr)
        adv = ggf(uF) - ggf(uF).mean()                          # (B,) welfare adv
        def loss(pp):
            return -(adv[None, :, None] * _logp(pol, pp, toks, acts)).mean()
        g = jax.grad(loss)(p); u, opt = tx.update(g, opt)
        return optax.apply_updates(p, u), opt, key
    for _ in range(iters):
        p, opt, key = upd(p, opt, key)
    return pol, p


def train_fen(c, B=512, iters=2000, lr=3e-3, seed=0):
    """FEN-style: each agent maximizes its own fair-efficient reward
    fe_i = mean_u / (c0 + |u_i - mean_u|) (high efficiency, low deviation)."""
    pol = MLP(); key = jax.random.PRNGKey(seed); key, ki = jax.random.split(key)
    p = pol.init(ki, jnp.zeros((1, N, 6))); tx = optax.adam(lr); opt = tx.init(p)

    @jax.jit
    def upd(p, opt, key):
        key, kr = jax.random.split(key)
        uF, (toks, acts, _) = _coop_rollout(lambda t: pol.apply(p, t), B, c, kr)
        mean = uF.mean(-1, keepdims=True)
        fe = mean / (FE_C0 + jnp.abs(uF - mean))                # (B,N) per-agent fe
        adv = fe - fe.mean(0, keepdims=True)                    # per-agent baseline
        def loss(pp):
            return -(adv[None] * _logp(pol, pp, toks, acts)).mean()
        g = jax.grad(loss)(p); u, opt = tx.update(g, opt)
        return optax.apply_updates(p, u), opt, key
    for _ in range(iters):
        p, opt, key = upd(p, opt, key)
    return pol, p


def train_soto(c, B=512, iters=2000, lr=3e-3, seed=0):
    """Faithful SOTO: Self-Oriented (own return) + Team-Oriented (GGF) sub-nets,
    behaviour annealing beta:1->0. Returns the converged Team-Oriented net."""
    pol = MLP(); key = jax.random.PRNGKey(seed)
    key, ks, kt = jax.random.split(key, 3)
    so_p = pol.init(ks, jnp.zeros((1, N, 6)))
    to_p = pol.init(kt, jnp.zeros((1, N, 6)))
    txs = optax.adam(lr); opts = txs.init(so_p)
    txt = optax.adam(lr); optt = txt.init(to_p)
    anneal = int(0.7 * iters)

    def rollout(so_p, to_p, beta, key):
        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            so_l, to_l = pol.apply(so_p, tok), pol.apply(to_p, tok)
            key, km, ks, kt = jax.random.split(key, 4)
            use_so = jax.random.bernoulli(km, beta, (B, N))
            a = jnp.where(use_so, jax.random.categorical(ks, so_l),
                          jax.random.categorical(kt, to_l))
            u = graded_step(u, a == 1, c)
            return (u, cc + (a == 1), key), (tok, a, use_so)
        (uF, _, _), out = jax.lax.scan(
            step, (jnp.zeros((B, N)), jnp.zeros((B, N)), key), jnp.arange(T))
        return uF, out

    @jax.jit
    def upd(so_p, to_p, opts, optt, beta, key):
        key, kr = jax.random.split(key)
        uF, (toks, acts, m) = rollout(so_p, to_p, beta, kr)
        own = uF - uF.mean(0, keepdims=True)                    # (B,N) own adv
        gadv = ggf(uF) - ggf(uF).mean()                         # (B,) team adv
        def so_loss(pp):                                        # acts where use_so
            return -(own[None] * _logp(pol, pp, toks, acts) * m).mean()
        def to_loss(pp):                                        # acts where team
            return -(gadv[None, :, None] * _logp(pol, pp, toks, acts)
                     * (1 - m)).mean()
        us, opts = txs.update(jax.grad(so_loss)(so_p), opts)
        ut, optt = txt.update(jax.grad(to_loss)(to_p), optt)
        return (optax.apply_updates(so_p, us), optax.apply_updates(to_p, ut),
                opts, optt, key)
    for it in range(iters):
        beta = max(0.0, 1.0 - it / anneal)
        so_p, to_p, opts, optt, key = upd(so_p, to_p, opts, optt, beta, key)
    return pol, to_p                                            # converged = TO net


# ----------------------------- audits ------------------------------------------
def fair_eval(pol, params, c, B=512, seed=321):
    """No-defector (D=0) audit: Jain (is the team fair?) and efficiency
    (delivered/T --- does it waste the resource when safe, same metric as CAN)."""
    uF, _ = _coop_rollout(lambda t: pol.apply(params, t), B, c,
                          jax.random.PRNGKey(seed))
    _, jn, eff = evaluate(uF, jnp.zeros((B, N)), T)
    return jn, eff


def br_audit(pol, fair_params, c, B=512, iters=1500, lr=3e-3, seed=7):
    """Train a best-response defector (own MLP) at a random index vs the FROZEN
    fair team; return (bounded free-ride rho, D=1 efficiency) --- same metrics as
    CAN."""
    dpol = MLP(); key = jax.random.PRNGKey(seed); key, ki = jax.random.split(key)
    dp = dpol.init(ki, jnp.zeros((1, N, 6))); tx = optax.adam(lr); opt = tx.init(dp)

    def rollout(dp, d_idx, key):
        d_oh = jax.nn.one_hot(d_idx, N)
        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            cl, dl = pol.apply(fair_params, tok), dpol.apply(dp, tok)
            key, ka, kb = jax.random.split(key, 3)
            a = jnp.where(d_oh.astype(bool), jax.random.categorical(kb, dl),
                          jax.random.categorical(ka, cl))
            u = graded_step(u, a == 1, c)
            return (u, cc + (a == 1), key), (tok, a)
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
    key, kd, kr = jax.random.split(key, 3)
    d_idx = jax.random.randint(kd, (B,), 0, N)
    uF, _, _, _ = rollout(dp, d_idx, kr)
    rho, _, eff = evaluate(uF, jax.nn.one_hot(d_idx, N), T)
    return rho, eff


TRAINERS = {"SOTO": train_soto, "FEN": train_fen, "GGF": train_ggf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--br_iters", type=int, default=1500)
    a = ap.parse_args()
    cs, seeds = (CS, SEEDS)
    if a.smoke:
        cs, seeds, a.iters, a.br_iters = [0.5], [0], 300, 300
    os.makedirs("results", exist_ok=True)
    out = "results/baselines_graded.csv"
    if not a.smoke and os.path.exists(out):
        os.remove(out)
    header = ["method", "c", "seed", "d0_jain", "d0_eff", "d1_eff", "br_rho"]
    for meth, trainer in TRAINERS.items():
        for c in cs:
            for s in seeds:
                pol, p = trainer(c, iters=a.iters, seed=s)
                jn, d0_eff = fair_eval(pol, p, c)
                rho, d1_eff = br_audit(pol, p, c, iters=a.br_iters, seed=700 + s)
                row = dict(method=meth, c=c, seed=s, d0_jain=float(jn),
                           d0_eff=float(d0_eff), d1_eff=float(d1_eff),
                           br_rho=float(rho))
                if not a.smoke:
                    new = not os.path.exists(out)
                    with open(out, "a", newline="") as f:
                        w = csv.DictWriter(f, header)
                        if new:
                            w.writeheader()
                        w.writerow(row); f.flush()
                print(f"[{meth}] c={c} s={s}: d0_jain={jn:.3f} d0_eff={d0_eff:.3f} "
                      f"d1_eff={d1_eff:.3f} br_rho={rho:.2f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
