"""M0 --- HARDENED BASELINES (decision gate): wrap the published fair-MARL
objectives (GGF, FEN, SOTO) in the IDENTICAL league (PSRO) loop used for CAN,
to separate "CAN's architecture/objective" from "adversarial training" as the
source of CAN's robustness.

Held identical to CAN's league (fair_xattn.league_train):
  - league structure: 6 generations x (1200 cooperator updates vs the frozen
    pool + 1000 updates for a fresh best-response defector added to the pool),
    pool seeded with one random-init defector;
  - training distribution: D ~ unif{0..dmax=2} defectors per episode, drawn
    from the pool (one pool member per episode);
  - estimator: REINFORCE on per-step welfare increments (discounted
    reward-to-go), batch-mean baseline, entropy 0.05 -> 0.003 annealed per
    generation --- so a failure cannot be blamed on a weaker estimator;
  - tokens: the same 6 behaviour features CAN sees (fair_xattn.features);
  - audit: the v1 baseline audit (fair_baselines_graded.br_audit) --- 1500
    best-response defector updates vs the frozen team --- plus a scripted
    always-claim audit as the stop-trigger sanity check (the learned defector
    must do at least as well).

Changed (the ONE variable): the cooperator objective is the baseline's own,
not CAN's mean-std coop welfare:
  - GGF+PSRO : GGF welfare over the COOPERATOR utilities (masked; reduces to
    fair_train.ggf when D=0);
  - FEN+PSRO : per-agent fair-efficient reward fe_i = mean_c/(c0+|u_i-mean_c|)
    with mean over cooperators (reduces to the v1 FEN reward when D=0);
  - SOTO+PSRO: Self-Oriented (own utility) + Team-Oriented (masked GGF)
    sub-nets with behaviour mixing beta annealed 1 -> 0 GLOBALLY over the
    6x1200 cooperator updates (anneal point 0.7, as in v1); the converged
    Team-Oriented net is the audited policy. This is the natural lift of the
    v1 schedule onto league alternation; whether it survives is part of the
    result.
Policy class stays the baselines' permutation-equivariant per-agent MLP
(fair_baselines_graded.MLP) --- architecture is NOT changed, so the gate
isolates adversarial training.

    python -m can.fair_m0_hardened             # full sweep -> CSV
    python -m can.fair_m0_hardened --smoke     # quick check
    python -m can.fair_m0_hardened --figs      # figure + gate table from CSV
"""
import os
import csv
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .fair_graded import graded_step
from .fair_xattn import (XAttn, features, evaluate, disc_rtg,
                         league_train as can_league,
                         best_response_audit as can_br)
from .fair_baselines_graded import MLP, FE_C0, fair_eval, br_audit

N, T = 6, 100
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]
OUT = "results/m0_hardened_baselines.csv"
EXP_DIR = "experiments/m0_hardened_baselines"


# ------------------------- masked (cooperator) objectives -----------------------
def ggf_coop(u, def_mask):
    """GGF welfare over cooperator utilities only. u (B,N), def_mask (B,N) {0,1}
    -> (B,). Equals fair_train.ggf(u) when def_mask == 0."""
    nc = (1.0 - def_mask).sum(-1)                              # (B,)
    s = jnp.sort(jnp.where(def_mask.astype(bool), jnp.inf, u), -1)  # coop first
    pos = jnp.arange(u.shape[-1])[None, :]
    w = jnp.clip(nc[:, None] - pos, 0.0, None)                 # nc, nc-1, .., 1, 0
    w = w / jnp.clip(w.sum(-1, keepdims=True), 1e-9, None)
    return (jnp.where(pos < nc[:, None], s, 0.0) * w).sum(-1)


def fe_coop(u, def_mask):
    """FEN fair-efficient reward per agent, computed over cooperators. u (B,N)
    -> (B,N), zero at defector slots. Equals the v1 FEN reward when D=0."""
    coop = 1.0 - def_mask
    nc = jnp.clip(coop.sum(-1, keepdims=True), 1.0, None)
    mean_c = (u * coop).sum(-1, keepdims=True) / nc
    fe = mean_c / (FE_C0 + jnp.abs(u - mean_c))
    return fe * coop


# ------------------------- shared league machinery -------------------------------
def _sample_masks(key, B, n, dmax):
    """D ~ unif{0..dmax} defectors at random indices (CAN's training distribution)."""
    Dn = jax.random.randint(key, (B,), 0, dmax + 1)
    rank = jnp.argsort(jnp.argsort(
        jax.random.uniform(jax.random.fold_in(key, 1), (B, n)), -1), -1)
    return (rank < Dn[:, None]).astype(jnp.float32)


def _pool_rollout(pol, coop_logits_fn, pool_stk, def_mask, pidx, B, c, key):
    """One episode batch: cooperators via coop_logits_fn(tok, key), defectors via
    the assigned pool member. Returns (toks, acts, useq, extras) with extras from
    coop_logits_fn (e.g. SOTO's mixing mask)."""
    dm = def_mask.astype(bool)

    def step(carry, t):
        u, cc, key = carry
        tok = features(u, cc, t, T)
        key, ka, kb = jax.random.split(key, 3)
        ca, extra = coop_logits_fn(tok, ka)                    # (B,N) coop actions
        dl_all = jax.vmap(lambda dp: pol.apply(dp, tok))(pool_stk)   # (P,B,N,2)
        dl = dl_all[pidx, jnp.arange(B)]
        acts = jnp.where(dm, jax.random.categorical(kb, dl), ca)
        u = graded_step(u, acts == 1, c)
        return (u, cc + (acts == 1), key), (tok, acts, u, extra)
    (uF, _, _), out = jax.lax.scan(
        step, (jnp.zeros((B, N)), jnp.zeros((B, N)), key), jnp.arange(T))
    return out


def _adv(W, B):
    """Per-step welfare increments -> discounted reward-to-go, batch-centered.
    W (T,B) or (T,B,N)."""
    Wprev = jnp.concatenate([jnp.zeros_like(W[:1]), W[:-1]], 0)
    G = disc_rtg(W - Wprev)
    return G - G.mean(1, keepdims=True)


def _lp_ent(pol, p, toks, acts):
    lsm = jax.nn.log_softmax(pol.apply(p, toks))
    lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
    ent = -(jnp.exp(lsm) * lsm).sum(-1)
    return lp, ent


def _coop_vs_pool(pol, coop_p, pool_stk, c, iters, lr, key, welfare_fn,
                  per_agent, B=512, dmax=2):
    """CAN's _coop_vs_pool with the welfare function swapped: train the shared
    cooperator MLP vs the frozen defector pool, REINFORCE + entropy anneal."""
    P = jax.tree_util.tree_leaves(pool_stk)[0].shape[0]
    tx = optax.adam(lr); opt = tx.init(coop_p)

    @jax.jit
    def upd(coop_p, opt, key, ec):
        key, ks, kp, kr = jax.random.split(key, 4)
        def_mask = _sample_masks(ks, B, N, dmax)
        pidx = jax.random.randint(kp, (B,), 0, P)
        coop = 1.0 - def_mask

        def coop_logits(tok, ka):
            return jax.random.categorical(ka, pol.apply(coop_p, tok)), 0.0
        toks, acts, useq, _ = _pool_rollout(
            pol, coop_logits, pool_stk, def_mask, pidx, B, c, kr)
        W = jax.vmap(welfare_fn, in_axes=(0, None))(useq, def_mask)
        ac = _adv(W, B)                                        # (T,B[,N])

        def loss(p):
            lp, ent = _lp_ent(pol, p, toks, acts)
            w = ac if per_agent else ac[:, :, None]
            return -(w * lp * coop[None]).sum(-1).mean() \
                - ec * (ent * coop[None]).sum(-1).mean()
        g = jax.grad(loss)(coop_p); u, opt = tx.update(g, opt)
        return optax.apply_updates(coop_p, u), opt, key

    ent_hi, ent_lo = 0.05, 0.003
    for it in range(iters):
        ec = ent_hi * (1 - it / iters) + ent_lo * (it / iters)
        coop_p, opt, key = upd(coop_p, opt, key, ec)
    return coop_p


def _br_defector(pol, coop_apply, c, iters, lr, seed, B=512):
    """Fresh best-response MLP defector vs the FROZEN cooperator behaviour
    coop_apply(tok, key)->acts, at a random index (CAN's protocol). Returns
    defector params."""
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    dp0 = pol.init(ki, jnp.zeros((1, N, 6)))
    tx = optax.adam(lr); opt = tx.init(dp0)

    def rollout(dp, d_idx, key):
        d_oh = jax.nn.one_hot(d_idx, N)
        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            key, ka, kb = jax.random.split(key, 3)
            ca = coop_apply(tok, ka)
            da = jax.random.categorical(kb, pol.apply(dp, tok))
            acts = jnp.where(d_oh.astype(bool), da, ca)
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
            lp, _ = _lp_ent(pol, p, toks, acts)
            return -(adv[None, :, None] * lp * d_oh[None]).sum(-1).mean()
        g = jax.grad(loss)(dp); u, opt = tx.update(g, opt)
        return optax.apply_updates(dp, u), opt, key

    dp = dp0
    for _ in range(iters):
        dp, opt, key = upd(dp, opt, key)
    return dp


# ------------------------- league training per method ----------------------------
def league_ggf_fen(method, c, generations=6, coop_iters=1200, br_iters=1000,
                   lr=3e-3, seed=0):
    welfare_fn, per_agent = {"GGF": (ggf_coop, False),
                             "FEN": (fe_coop, True)}[method]
    pol = MLP()
    key = jax.random.PRNGKey(seed)
    key, kc, kd = jax.random.split(key, 3)
    coop_p = pol.init(kc, jnp.zeros((1, N, 6)))
    pool = [pol.init(kd, jnp.zeros((1, N, 6)))]
    for gen in range(generations):
        pool_stk = jax.tree.map(lambda *xs: jnp.stack(xs), *pool)
        key, k1 = jax.random.split(key)
        coop_p = _coop_vs_pool(pol, coop_p, pool_stk, c, coop_iters, lr, k1,
                               welfare_fn, per_agent)
        cp = coop_p
        pool.append(_br_defector(
            pol, lambda tok, ka: jax.random.categorical(ka, pol.apply(cp, tok)),
            c, br_iters, lr, seed * 100 + gen))
    return pol, coop_p


def league_soto(c, generations=6, coop_iters=1200, br_iters=1000, lr=3e-3,
                seed=0, B=512, dmax=2):
    """SOTO under the league: SO (own utility) + TO (masked GGF) sub-nets,
    behaviour mixing beta annealed 1->0 globally over generations*coop_iters
    cooperator updates (anneal point 0.7, as in v1). Audited policy = TO net."""
    pol = MLP()
    key = jax.random.PRNGKey(seed)
    key, ks, kt, kd = jax.random.split(key, 4)
    so_p = pol.init(ks, jnp.zeros((1, N, 6)))
    to_p = pol.init(kt, jnp.zeros((1, N, 6)))
    pool = [pol.init(kd, jnp.zeros((1, N, 6)))]
    total = generations * coop_iters
    anneal = int(0.7 * total)
    txs = optax.adam(lr); opts = txs.init(so_p)
    txt = optax.adam(lr); optt = txt.init(to_p)

    def make_upd(pool_stk):
        P = jax.tree_util.tree_leaves(pool_stk)[0].shape[0]

        @jax.jit
        def upd(so_p, to_p, opts, optt, key, beta, ec):
            key, kk, kp, kr = jax.random.split(key, 4)
            def_mask = _sample_masks(kk, B, N, dmax)
            pidx = jax.random.randint(kp, (B,), 0, P)
            coop = 1.0 - def_mask

            def coop_logits(tok, ka):
                km, k1, k2 = jax.random.split(ka, 3)
                use_so = jax.random.bernoulli(km, beta, (B, N))
                a = jnp.where(use_so,
                              jax.random.categorical(k1, pol.apply(so_p, tok)),
                              jax.random.categorical(k2, pol.apply(to_p, tok)))
                return a, use_so.astype(jnp.float32)
            toks, acts, useq, m = _pool_rollout(
                pol, coop_logits, pool_stk, def_mask, pidx, B, c, kr)
            own = _adv(useq, B)                                # (T,B,N) own rtg
            gadv = _adv(jax.vmap(ggf_coop, in_axes=(0, None))(useq, def_mask), B)

            def so_loss(p):
                lp, ent = _lp_ent(pol, p, toks, acts)
                msk = m * coop[None]
                return -(own * lp * msk).sum(-1).mean() \
                    - ec * (ent * msk).sum(-1).mean()
            def to_loss(p):
                lp, ent = _lp_ent(pol, p, toks, acts)
                msk = (1 - m) * coop[None]
                return -(gadv[:, :, None] * lp * msk).sum(-1).mean() \
                    - ec * (ent * msk).sum(-1).mean()
            us, opts2 = txs.update(jax.grad(so_loss)(so_p), opts)
            ut, optt2 = txt.update(jax.grad(to_loss)(to_p), optt)
            return (optax.apply_updates(so_p, us), optax.apply_updates(to_p, ut),
                    opts2, optt2, key)
        return upd

    ent_hi, ent_lo = 0.05, 0.003
    for gen in range(generations):
        pool_stk = jax.tree.map(lambda *xs: jnp.stack(xs), *pool)
        upd = make_upd(pool_stk)
        key, k1 = jax.random.split(key)
        for it in range(coop_iters):
            g_it = gen * coop_iters + it
            beta = max(0.0, 1.0 - g_it / anneal)
            ec = ent_hi * (1 - it / coop_iters) + ent_lo * (it / coop_iters)
            so_p, to_p, opts, optt, k1 = upd(so_p, to_p, opts, optt, k1, beta, ec)
        # the BR defector exploits the CURRENT behaviour policy (the beta-mix)
        sp, tp, bnow = so_p, to_p, max(0.0, 1.0 - ((gen + 1) * coop_iters) / anneal)

        def coop_apply(tok, ka, sp=sp, tp=tp, bnow=bnow):
            km, k1_, k2_ = jax.random.split(ka, 3)
            use_so = jax.random.bernoulli(km, bnow, tok.shape[:-1])
            return jnp.where(use_so,
                             jax.random.categorical(k1_, pol.apply(sp, tok)),
                             jax.random.categorical(k2_, pol.apply(tp, tok)))
        pool.append(_br_defector(pol, coop_apply, c, br_iters, lr,
                                 seed * 100 + gen))
    return pol, to_p


# ------------------------- audits ------------------------------------------------
def always_claim_rho(pol, coop_p, c, B=512, seed=99):
    """Scripted always-claim defector at a random index vs the frozen team ---
    the stop-trigger reference the learned BR defector must not lose to."""
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
    rho, _, _ = evaluate(uF, d_oh, T)
    return rho


# ------------------------- figure + gate table -----------------------------------
def make_figs():
    """Fig. 2 scatter with the hardened points + the decision-gate table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import glob
    os.makedirs(EXP_DIR, exist_ok=True)
    brows = list(csv.DictReader(open("results/baselines_graded.csv")))
    lrows = list(csv.DictReader(open("results/league_pooled.csv")))
    hrows = [r for f in sorted(glob.glob("results/m0_hardened_*.csv"))
             for r in csv.DictReader(open(f))]

    def pts(rows, meth=None, ymax=False):
        sel = [r for r in rows if meth is None or r["method"] == meth]
        ys = [max(float(r["br_rho"]), float(r["ac_rho"])) if ymax
              else float(r["br_rho"]) for r in sel]
        return [float(r["d0_eff"]) for r in sel], ys

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.axhspan(0.8, 1.6, xmin=0.9 / 1.1, xmax=1.0, color="#eaf5ed", zorder=0)
    ax.axhline(1.0, ls="--", c="#888", lw=1)
    spec_v1 = [("GGF", "tab:gray", "o"), ("FEN", "#df8a2c", "^"),
               ("SOTO", "#5277bd", "s")]
    for meth, col, mk in spec_v1:
        xs, ys = pts(brows, meth)
        ax.scatter(xs, ys, c=col, marker=mk, s=45, alpha=0.35,
                   edgecolors="none", label=f"{meth} (v1, cooperative)", zorder=3)
        xs, ys = pts(hrows, meth, ymax=True)
        ax.scatter(xs, ys, c=col, marker=mk, s=70, edgecolors="black",
                   linewidths=0.8, label=f"{meth}+PSRO (hardened)", zorder=5)
    xs, ys = pts(lrows)
    ax.scatter(xs, ys, c="#2f9e44", marker="*", s=110, alpha=0.35,
               edgecolors="none", label="CAN (league, v1 audit)", zorder=4)
    xs, ys = pts(hrows, "CAN", ymax=True)
    ax.scatter(xs, ys, c="#2f9e44", marker="*", s=150, edgecolors="black",
               linewidths=0.8, label="CAN (league, dual audit)", zorder=6)
    ax.scatter([1.0], [1.0], marker="*", s=260, c="#f0b429",
               edgecolors="#7a5800", linewidths=0.8, zorder=7,
               label="centralized oracle")
    ax.set_xlim(0.0, 1.1); ax.set_ylim(0.7, 6.4)
    ax.set_xlabel("efficiency when no free-rider present  (D=0; $\\to$1 = no waste)")
    ax.set_ylabel(r"best-response free-ride $\rho$  ($\to$1 = robust)")
    ax.set_title("M0: hardened baselines vs v1 (per seed$\\times c$)")
    ax.legend(fontsize=7, loc="center left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(f"{EXP_DIR}/fig_m0_tradeoff.png", dpi=160)
    plt.close(fig)

    # gate table: per method x c, mean (95% bootstrap CI) of rho and d0_eff
    def boot(v, n=4000, seed=0):
        v = np.asarray(v, float)
        if len(v) < 2:
            return v.mean(), v.mean(), v.mean()
        rng = np.random.default_rng(seed)
        bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
        return v.mean(), *np.percentile(bs, [2.5, 97.5])

    lines = ["| method | c | D=0 eff (95% CI) | BR rho (95% CI) | "
             "always-claim rho | max rho (95% CI) | D>=1 eff | "
             "gate (eff>=0.95 & max rho<=1.6) |",
             "|---|---|---|---|---|---|---|---|"]
    for meth in ["GGF", "FEN", "SOTO", "CAN"]:
        for c in CS:
            sel = [r for r in hrows if r["method"] == meth
                   and abs(float(r["c"]) - c) < 1e-9]
            if not sel:
                continue
            em, el, eh = boot([float(r["d0_eff"]) for r in sel])
            rm, rl, rh = boot([float(r["br_rho"]) for r in sel])
            acr = np.mean([float(r["ac_rho"]) for r in sel])
            # exploitability lower bound: the WORST of the learned BR and the
            # scripted always-claim, per seed (the learned BR alone under-
            # exploits at high c --- it is deterrable, the script is not)
            mm, ml, mh = boot([max(float(r["br_rho"]), float(r["ac_rho"]))
                               for r in sel])
            d1 = np.mean([float(r["d1_eff"]) for r in sel])
            gate = "PASS" if (em >= 0.95 and mm <= 1.6) else "no"
            lines.append(f"| {meth}+PSRO | {c} | {em:.3f} [{el:.3f},{eh:.3f}] | "
                         f"{rm:.2f} [{rl:.2f},{rh:.2f}] | {acr:.2f} | "
                         f"{mm:.2f} [{ml:.2f},{mh:.2f}] | {d1:.2f} | {gate} |")
    table = "\n".join(lines)
    with open(f"{EXP_DIR}/results_table.md", "w") as f:
        f.write("# M0 hardened baselines --- gate table\n\n" + table + "\n")
    print(table)
    print(f"\nwrote {EXP_DIR}/fig_m0_tradeoff.png and results_table.md")


# ------------------------- driver ------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--methods", nargs="+", default=["GGF", "FEN", "SOTO"])
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--coop_iters", type=int, default=1200)
    ap.add_argument("--br_iters", type=int, default=1000)
    ap.add_argument("--audit_iters", type=int, default=1500)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.figs:
        make_figs()
        return
    cs, seeds = CS, SEEDS
    if a.smoke:
        cs, seeds = [0.5], [0]
        a.generations, a.coop_iters, a.br_iters, a.audit_iters = 2, 60, 50, 100
    os.makedirs("results", exist_ok=True)
    header = ["method", "c", "seed", "d0_jain", "d0_eff", "d1_eff", "br_rho",
              "ac_rho"]
    done = set()
    if not a.smoke and os.path.exists(a.out):                  # resume support
        done = {(r["method"], float(r["c"]), int(r["seed"]))
                for r in csv.DictReader(open(a.out))}
    for meth in a.methods:
        for c in cs:
            for s in seeds:
                if (meth, c, s) in done:
                    print(f"[{meth}+PSRO] c={c} s={s}: already in {a.out}, skip",
                          flush=True)
                    continue
                if meth == "CAN":
                    # v1 CAN league retrained verbatim, audited with BOTH the
                    # learned BR (v1 protocol, XAttn defector, seed 800+s as in
                    # fair_rerun) and the always-claim script -- so the gate
                    # scores CAN and the baselines on the same max(br, ac).
                    pol = XAttn()
                    p = can_league(N, T, c, generations=a.generations,
                                   coop_iters=a.coop_iters,
                                   br_iters=a.br_iters, seed=s)
                    rho, d1_eff = can_br(p, N, T, c, iters=a.audit_iters,
                                         seed=800 + s, return_eff=True)
                elif meth == "SOTO":
                    pol, p = league_soto(c, a.generations, a.coop_iters,
                                         a.br_iters, seed=s)
                else:
                    pol, p = league_ggf_fen(meth, c, a.generations,
                                            a.coop_iters, a.br_iters, seed=s)
                if meth != "CAN":
                    rho, d1_eff = br_audit(pol, p, c, iters=a.audit_iters,
                                           seed=700 + s)
                jn, d0_eff = fair_eval(pol, p, c)
                acr = always_claim_rho(pol, p, c)
                row = dict(method=meth, c=c, seed=s, d0_jain=float(jn),
                           d0_eff=float(d0_eff), d1_eff=float(d1_eff),
                           br_rho=float(rho), ac_rho=float(acr))
                if not a.smoke:
                    new = not os.path.exists(a.out)
                    with open(a.out, "a", newline="") as f:
                        w = csv.DictWriter(f, header)
                        if new:
                            w.writeheader()
                        w.writerow(row); f.flush()
                warn = " [BR<always-claim!]" if rho < acr - 0.15 else ""
                print(f"[{meth}+PSRO] c={c} s={s}: d0_jain={jn:.3f} "
                      f"d0_eff={d0_eff:.3f} d1_eff={d1_eff:.3f} "
                      f"br_rho={rho:.2f} ac_rho={acr:.2f}{warn}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
