"""M2c --- REGIME UNIFICATION: one policy from graded contesting to commitment.

The program so far produced two specialists: league-trained CAN for the graded
regime (c < 1: contest proportionally for surplus) and the distilled absorbing
trigger for the boundary (c ~ 1: grim parity, M2b). This experiment asks for
ONE set of weights that does both, and tests HOW commitment gets into a
multi-regime policy:

  uni-rl   : multi-c league RL from scratch (control --- M2b predicts no
             commitment at c=1: deterrable training adversaries never reward
             absorption).
  uni-bcft : distill the grim teacher first (episodes at c in {0.95, 1.0}),
             then league fine-tune across all c. Tests whether RL RETAINS
             taught commitment (the gradient toward absorption is ~0, but is
             there a gradient away from it?).
  uni-joint: league RL with an auxiliary behaviour-cloning loss toward the
             grim teacher on high-c (c >= 0.95) episodes in every update.

Tokens: 9 channels = the 8 absorbing-flag channels of M2b + an explicit c
channel (broadcast, like t/T), so the policy can condition its regime. At
c < 1 proportional contesting saturates the flag channels; they are inputs,
not rules --- the c channel lets the policy ignore them there, and only the
teacher enforces their grim semantics at c ~ 1. Training c is sampled per
episode from {0.3, 0.5, 0.7, 0.9, 0.95, 1.0}.

Audit (same single weights, per c): c=1.0 full M2 suite (BR learner /
always-claim / 3x patient + D=0 eff + FP); c in {0.5, 0.9} BR learner +
always-claim + D=0 (the patience question lives at the boundary). References:
the per-c specialists (M0 CAN league at 0.5/0.9, distill8 at 1.0).

    python -m can.fair_m2c_unified --arm uni-rl
    python -m can.fair_m2c_unified --arm uni-bcft
    python -m can.fair_m2c_unified --arm uni-joint
    python -m can.fair_m2c_unified --smoke
    python -m can.fair_m2c_unified --figs
"""
import os
import csv
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.serialization

from .fair_graded import graded_step
from .fair_xattn import XAttn, evaluate, disc_rtg, coop_welfare
from .fair_m2_deterrence import br_defector, always_claim_act
from .fair_m2b_commitment import flag_update, features8, fstate0

N, T = 6, 100
SEEDS = [0, 1, 2, 3, 4]
C_GRID = jnp.array([0.3, 0.5, 0.7, 0.9, 0.95, 1.0])
CS_EVAL = [0.5, 0.9, 1.0]
OUT = "results/m2c_unified.csv"
PDIR = "results/m2c_params"
EXP_DIR = "experiments/m2c_unified"


def features9(u, cc, fstate, c_ep, t, T):
    """8 flag-channels + broadcast c channel. c_ep (B,) -> (B,N,9)."""
    cch = jnp.broadcast_to(c_ep[:, None, None], u.shape + (1,))
    return jnp.concatenate([features8(u, cc, fstate, t, T), cch], -1)


def make_rollout9(params, c, B):
    """Audit rollout at FIXED scalar c: rf(def_act, def_mask, key, ret_traj)."""
    pol = XAttn()
    c_ep = jnp.full((B,), c)

    def rf(def_act, def_mask, key, ret_traj=False):
        dm = def_mask.astype(bool)

        def step(carry, t):
            u, cc, fstate, key = carry
            tok = features9(u, cc, fstate, c_ep, t, T)
            key, ka, kb = jax.random.split(key, 3)
            ca = jax.random.categorical(ka, pol.apply(params, tok))
            acts = jnp.where(dm, def_act(tok, kb), ca)
            claim = acts == 1
            u = graded_step(u, claim, c)
            fstate = flag_update(fstate, claim)
            return (u, cc + claim, fstate, key), (tok, acts, u, claim.sum(-1))
        z = jnp.zeros((B, N))
        (uF, _, _, _), (toks, acts, useq, ms) = jax.lax.scan(
            step, (z, z, fstate0(B), key), jnp.arange(T))
        return (uF, toks, acts, useq, ms) if ret_traj else (uF, ms)
    return rf


# ------------------------- teacher (BC) ------------------------------------------
def _teacher_batch(B, key, hi_only=True):
    """Grim-teacher episodes on 9ch tokens. c per episode from {0.95, 1.0} if
    hi_only else the full grid. Returns toks, labels, coop mask, c_ep."""
    key, k0, k1, k2, k3, k4, k5 = jax.random.split(key, 7)
    if hi_only:
        c_ep = jnp.where(jax.random.bernoulli(k0, 0.5, (B,)), 1.0, 0.95)
    else:
        c_ep = C_GRID[jax.random.randint(k0, (B,), 0, len(C_GRID))]
    typ = jax.random.randint(k1, (B,), 0, 4)
    d_oh = jax.nn.one_hot(jax.random.randint(k2, (B,), 0, N), N)
    d_oh = d_oh * (typ != 0)[:, None]
    p_rate = jax.random.uniform(k3, (B,), minval=0.05, maxval=1.0)
    k_burst = jax.random.randint(k4, (B,), 5, 51)
    dm = d_oh.astype(bool)

    def step(carry, t):
        u, cc, fstate, key = carry
        trig = fstate[3]
        tok = features9(u, cc, fstate, c_ep, t, T)
        teacher = jnp.broadcast_to(trig[:, None], (B, N)).astype(jnp.int32)
        key, kb = jax.random.split(key)
        p = jnp.where(typ == 1, 1.0,
                      jnp.where(typ == 2, p_rate,
                                (t < k_burst).astype(jnp.float32)))
        da = jax.random.bernoulli(kb, p[:, None], (B, N)).astype(jnp.int32)
        acts = jnp.where(dm, da, teacher)
        claim = acts == 1
        u = graded_step(u, claim, c_ep)
        fstate = flag_update(fstate, claim)
        return (u, cc + claim, fstate, key), (tok, teacher)
    z = jnp.zeros((B, N))
    (_, _, _, _), (toks, labels) = jax.lax.scan(
        step, (z, z, fstate0(B), k5), jnp.arange(T))
    return toks, labels, 1.0 - d_oh


def distill_uni(params, iters=2000, B=256, lr=1e-3, seed=0):
    """BC the grim teacher (high-c episodes) into a 9ch XAttn policy."""
    pol = XAttn()
    tx = optax.adam(lr); opt = tx.init(params)
    key = jax.random.PRNGKey(seed + 5000)

    @jax.jit
    def upd(params, opt, key):
        key, kr = jax.random.split(key)
        toks, labels, coop = _teacher_batch(B, kr)

        def loss(p):
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            lp = jnp.take_along_axis(lsm, labels[..., None], -1)[..., 0]
            return -(lp * coop[None]).mean()
        g = jax.grad(loss)(params); u, opt = tx.update(g, opt)
        return optax.apply_updates(params, u), opt, key

    for _ in range(iters):
        params, opt, key = upd(params, opt, key)
    return params


# ------------------------- multi-c league ----------------------------------------
def league_uni(arm, generations=6, coop_iters=1200, br_iters=1000, B=512,
               lr=3e-3, seed=0, dmax=2, bc_lambda=1.0):
    """Multi-c league. arm: uni-rl (plain), uni-bcft (BC init), uni-joint
    (auxiliary BC loss on c>=0.95 episodes every update)."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    key, kc, kd = jax.random.split(key, 3)
    coop_p = pol.init(kc, jnp.zeros((1, N, 9)))
    if arm == "uni-bcft":
        coop_p = distill_uni(coop_p, seed=seed)
    pool = [pol.init(kd, jnp.zeros((1, N, 9)))]

    def coop_vs_pool(coop_p, pool_stk, iters, key):
        P = jax.tree_util.tree_leaves(pool_stk)[0].shape[0]
        tx = optax.adam(lr); opt = tx.init(coop_p)

        def rollout(coop_p, c_ep, def_mask, pidx, key):
            dm = def_mask.astype(bool)

            def step(carry, t):
                u, cc, fstate, key = carry
                tok = features9(u, cc, fstate, c_ep, t, T)
                cl = pol.apply(coop_p, tok)
                dl_all = jax.vmap(lambda dp: pol.apply(dp, tok))(pool_stk)
                dl = dl_all[pidx, jnp.arange(B)]
                key, ka, kb = jax.random.split(key, 3)
                acts = jnp.where(dm, jax.random.categorical(kb, dl),
                                 jax.random.categorical(ka, cl))
                claim = acts == 1
                u = graded_step(u, claim, c_ep)
                fstate = flag_update(fstate, claim)
                return (u, cc + claim, fstate, key), (tok, acts, u)
            z = jnp.zeros((B, N))
            (_, _, _, _), out = jax.lax.scan(
                step, (z, z, fstate0(B), key), jnp.arange(T))
            return out

        @jax.jit
        def upd(coop_p, opt, key, ec):
            key, ks, kp, kr, kcc, kt = jax.random.split(key, 6)
            c_ep = C_GRID[jax.random.randint(kcc, (B,), 0, len(C_GRID))]
            Dn = jax.random.randint(ks, (B,), 0, dmax + 1)
            rank = jnp.argsort(jnp.argsort(
                jax.random.uniform(jax.random.fold_in(ks, 1), (B, N)), -1), -1)
            def_mask = (rank < Dn[:, None]).astype(jnp.float32)
            pidx = jax.random.randint(kp, (B,), 0, P)
            coop = 1.0 - def_mask
            toks, acts, useq = rollout(coop_p, c_ep, def_mask, pidx, kr)
            Wc = jax.vmap(coop_welfare, in_axes=(0, None))(useq, def_mask)
            ac = disc_rtg(Wc - jnp.concatenate(
                [jnp.zeros((1, B)), Wc[:-1]], 0))
            ac = ac - ac.mean(1, keepdims=True)
            if arm == "uni-joint":
                ttoks, tlabels, tcoop = _teacher_batch(B // 2, kt)

            def loss(p):
                lsm = jax.nn.log_softmax(pol.apply(p, toks))
                lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
                ent = -(jnp.exp(lsm) * lsm).sum(-1)
                L = -(ac[:, :, None] * lp * coop[None]).sum(-1).mean() \
                    - ec * (ent * coop[None]).sum(-1).mean()
                if arm == "uni-joint":
                    tlsm = jax.nn.log_softmax(pol.apply(p, ttoks))
                    tlp = jnp.take_along_axis(
                        tlsm, tlabels[..., None], -1)[..., 0]
                    L = L - bc_lambda * (tlp * tcoop[None]).mean()
                return L
            g = jax.grad(loss)(coop_p); u, opt = tx.update(g, opt)
            return optax.apply_updates(coop_p, u), opt, key

        ent_hi, ent_lo = 0.05, 0.003
        for it in range(iters):
            ec = ent_hi * (1 - it / iters) + ent_lo * (it / iters)
            coop_p, opt, key = upd(coop_p, opt, key, ec)
        return coop_p

    for gen in range(generations):
        pool_stk = jax.tree.map(lambda *xs: jnp.stack(xs), *pool)
        key, k1 = jax.random.split(key)
        coop_p = coop_vs_pool(coop_p, pool_stk, coop_iters, k1)
        # fresh BR defector trained on the c-mix (it sees the c channel)
        pool.append(_br_defector_mix(coop_p, br_iters, lr, seed * 100 + gen,
                                     B=B))
    return coop_p


def _br_defector_mix(coop_p, iters, lr, seed, B=512):
    """BR defector against the frozen unified cooperator, trained across the
    same per-episode c mix it will meet in the league."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    dp = pol.init(ki, jnp.zeros((1, N, 9)))
    tx = optax.adam(lr); opt = tx.init(dp)

    @jax.jit
    def upd(dp, opt, key):
        key, kd, kr, kcc = jax.random.split(key, 4)
        c_ep = C_GRID[jax.random.randint(kcc, (B,), 0, len(C_GRID))]
        d_oh = jax.nn.one_hot(jax.random.randint(kd, (B,), 0, N), N)
        dm = d_oh.astype(bool)

        def step(carry, t):
            u, cc, fstate, key = carry
            tok = features9(u, cc, fstate, c_ep, t, T)
            key, ka, kb = jax.random.split(key, 3)
            ca = jax.random.categorical(ka, pol.apply(coop_p, tok))
            da = jax.random.categorical(kb, pol.apply(dp, tok))
            acts = jnp.where(dm, da, ca)
            claim = acts == 1
            u = graded_step(u, claim, c_ep)
            fstate = flag_update(fstate, claim)
            return (u, cc + claim, fstate, key), (tok, acts)
        z = jnp.zeros((B, N))
        (uF, _, _, _), (toks, acts) = jax.lax.scan(
            step, (z, z, fstate0(B), kr), jnp.arange(T))
        R = (uF * d_oh).sum(-1); adv = R - R.mean()

        def loss(p):
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
            return -(adv[None, :, None] * lp * d_oh[None]).sum(-1).mean()
        g = jax.grad(loss)(dp); u, opt = tx.update(g, opt)
        return optax.apply_updates(dp, u), opt, key

    for _ in range(iters):
        dp, opt, key = upd(dp, opt, key)
    return dp


# ------------------------- audits ------------------------------------------------
def audit_uni(params, c, B=512, br_iters=1500, seed=7, patient=False):
    rf = make_rollout9(params, c, B)
    out = {}
    m0 = jnp.zeros((B, N))
    uF, ms = rf(always_claim_act, m0, jax.random.PRNGKey(seed + 1))
    out["d0_eff"] = float(np.asarray(uF).sum(1).mean()) / T
    out["d0_fp"] = float((np.asarray(ms) >= 2).mean())
    kd, kr = jax.random.split(jax.random.PRNGKey(seed + 2))
    d_oh = jax.nn.one_hot(jax.random.randint(kd, (B,), 0, N), N)
    uF, _ = rf(always_claim_act, d_oh, kr)
    out["ac_rho"] = evaluate(uF, d_oh, T)[0]
    classes = [("br", 1)] + ([("patient", 3)] if patient else [])
    for name, mult in classes:
        def_act = br_defector(rf, c, br_iters * mult, seed=seed + 700 + mult,
                              return_act=True, feat=9)
        kd, kr = jax.random.split(jax.random.PRNGKey(seed + 3 + mult))
        d_oh = jax.nn.one_hot(jax.random.randint(kd, (B,), 0, N), N)
        uF, _ = rf(def_act, d_oh, kr)
        out[f"{name}_rho"] = evaluate(uF, d_oh, T)[0]
    return out


# ------------------------- figure ------------------------------------------------
def make_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(EXP_DIR, exist_ok=True)
    rows = list(csv.DictReader(open(OUT)))
    # per-c specialist references
    can = list(csv.DictReader(open("results/m0_hardened_CAN.csv")))
    m2b = list(csv.DictReader(open("results/m2b_commitment.csv")))
    ref = {0.5: np.mean([float(r["br_rho"]) for r in can
                         if abs(float(r["c"]) - 0.5) < 1e-9]),
           0.9: np.mean([float(r["br_rho"]) for r in can
                         if abs(float(r["c"]) - 0.9) < 1e-9]),
           1.0: np.mean([float(r["br_rho"]) for r in m2b
                         if r["policy"] == "distill8"
                         and abs(float(r["c"]) - 1.0) < 1e-9])}
    arms = ["uni-rl", "uni-bcft", "uni-joint"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for ax, c in zip(axes, CS_EVAL):
        xs = np.arange(len(arms))
        for i, (col, lbl, cl) in enumerate(
                [("br_rho", "BR learner", "#2f9e44"),
                 ("ac_rho", "always-claim", "#d05c5c")]):
            ms, sd = [], []
            for a in arms:
                v = [float(r[col]) for r in rows
                     if r["policy"] == a and abs(float(r["c"]) - c) < 1e-9]
                ms.append(np.mean(v) if v else np.nan)
                sd.append(np.std(v) if v else 0)
            ax.bar(xs + (i - 0.5) * 0.35, ms, 0.35, yerr=sd, capsize=3,
                   label=lbl, color=cl, alpha=0.9)
        ax.axhline(ref[c], ls="--", c="#444", lw=1.2,
                   label="per-$c$ specialist (BR)")
        ax.axhline(1.0, ls="--", c="#bbb", lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(arms, fontsize=9)
        ax.set_title(f"c = {c}", fontsize=10)
    axes[0].set_ylabel(r"free-ride $\rho$ (single policy across all $c$)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("M2c: one policy across regimes vs per-$c$ specialists "
                 "(5 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{EXP_DIR}/fig_m2c_unified.png", dpi=160)
    plt.close(fig)
    lines = ["| arm | c | D=0 eff | FP | BR rho | always-claim rho | "
             "patient rho |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        for c in CS_EVAL:
            sel = [r for r in rows
                   if r["policy"] == a and abs(float(r["c"]) - c) < 1e-9]
            if not sel:
                continue
            g = lambda k: np.mean([float(r[k]) for r in sel if r[k]])
            pat = f"{g('patient_rho'):.2f}" if c == 1.0 else "---"
            lines.append(f"| {a} | {c} | {g('d0_eff'):.3f} | {g('d0_fp'):.3f} "
                         f"| {g('br_rho'):.2f} | {g('ac_rho'):.2f} | {pat} |")
    with open(f"{EXP_DIR}/results_table.md", "w") as f:
        f.write("# M2c unified policy --- audits\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {EXP_DIR}/fig_m2c_unified.png and results_table.md")


# ------------------------- driver ------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["uni-rl", "uni-bcft", "uni-joint"],
                    default="uni-rl")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--coop_iters", type=int, default=1200)
    ap.add_argument("--br_iters", type=int, default=1000)
    ap.add_argument("--audit_iters", type=int, default=1500)
    a = ap.parse_args()
    if a.figs:
        make_figs(); return
    seeds = SEEDS
    if a.smoke:
        seeds = [0]
        a.generations, a.coop_iters, a.br_iters, a.audit_iters = 2, 60, 50, 80
    os.makedirs("results", exist_ok=True)
    os.makedirs(PDIR, exist_ok=True)
    header = ["policy", "c", "seed", "d0_eff", "d0_fp", "br_rho", "ac_rho",
              "patient_rho"]
    done = set()
    if not a.smoke and os.path.exists(OUT):
        done = {(r["policy"], float(r["c"]), int(r["seed"]))
                for r in csv.DictReader(open(OUT))}
    for s in seeds:
        if all((a.arm, c, s) in done for c in CS_EVAL):
            print(f"[{a.arm}] s={s}: done, skip", flush=True)
            continue
        p = league_uni(a.arm, a.generations, a.coop_iters, a.br_iters, seed=s)
        if not a.smoke:
            with open(f"{PDIR}/m2c_{a.arm}_s{s}.msgpack", "wb") as fh:
                fh.write(flax.serialization.to_bytes(p))
        for c in CS_EVAL:
            if (a.arm, c, s) in done:
                continue
            met = audit_uni(p, c, br_iters=a.audit_iters, seed=7 + s,
                            patient=(c == 1.0))
            row = dict(policy=a.arm, c=c, seed=s,
                       d0_eff=float(met["d0_eff"]), d0_fp=float(met["d0_fp"]),
                       br_rho=float(met["br_rho"]),
                       ac_rho=float(met["ac_rho"]),
                       patient_rho=float(met.get("patient_rho", float("nan"))))
            if not a.smoke:
                new = not os.path.exists(OUT)
                with open(OUT, "a", newline="") as f:
                    w = csv.DictWriter(f, header)
                    if new:
                        w.writeheader()
                    w.writerow(row); f.flush()
            print(f"[{a.arm}] s={s} c={c}: d0_eff={met['d0_eff']:.3f} "
                  f"fp={met['d0_fp']:.3f} br={met['br_rho']:.2f} "
                  f"ac={met['ac_rho']:.2f} "
                  f"patient={met.get('patient_rho', float('nan')):.2f}",
                  flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
