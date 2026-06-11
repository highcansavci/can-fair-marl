"""M1c --- FAST EVIDENCE + COUNT-PRESERVING AGGREGATION (the combined fix).

M1: token scale is not the constraint. M1b: max-pooled suspicion halves the
lone-defector transfer gap (aggregation fixed) but the residual is
EVIDENCE-limited --- the v1 claim-rate channel cc/t accumulates slowly, so
early-episode ambiguity costs ~latency banked units regardless of pooling.

M1c adds the missing fast-evidence channel and completes the 2x2 factorial:

    tokens \\ aggregation | mean-only (XAttn) | max-suspicion (XAttnMax)
    v1 (6ch)             | M1 v1tok control  | M1b maxagg
    v1 + claimed-last-step (7ch) | arm "last" | arm "lastmax"  <- M1c

The 7th channel is the per-agent indicator "claimed at t-1": instantaneous
evidence (a persistent claimer shows a solid 1-streak immediately) instead of
the cumulative average cc/t. Everything else is the M1/M1b protocol verbatim
(vanilla train at N=6, 2500 updates, dmax=2; lone-defector always-claim
transfer at N in {6,12,24,48}; BR audit at N=6; 5 seeds).

    python -m can.fair_m1c_evidence --arm lastmax       # the combined fix
    python -m can.fair_m1c_evidence --arm last          # channel-only control
    python -m can.fair_m1c_evidence --smoke
    python -m can.fair_m1c_evidence --figs              # 2x2 factorial figure
    python -m can.fair_m1c_evidence --probe             # onset-latency deciles
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
from .fair_xattn import XAttn, features, evaluate, disc_rtg, coop_welfare
from .fair_m1b_count_agg import XAttnMax

T = 100
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]
NS_EVAL = [6, 12, 24, 48]
F = 7
OUT = "results/m1c_evidence.csv"
PDIR = "results/m1c_params"
EXP_DIR = "experiments/m1c_evidence"
NETS = {"last": XAttn, "lastmax": XAttnMax}


def features7(u, cc, cl_prev, t, T):
    """v1 tokens + per-agent claimed-last-step indicator. (B,N)->(B,N,7)."""
    return jnp.concatenate(
        [features(u, cc, t, T), cl_prev.astype(jnp.float32)[..., None]], -1)


def make_rollout(net, N, T, B, c):
    pol = net()

    def rollout(params, def_mask, key):
        dm = def_mask.astype(bool)

        def step(carry, t):
            u, cc, clp, key = carry
            tok = features7(u, cc, clp, t, T)
            key, ka = jax.random.split(key)
            acts = jax.random.categorical(ka, pol.apply(params, tok))
            acts = jnp.where(dm, 1, acts)
            claim = acts == 1
            u = graded_step(u, claim, c)
            return (u, cc + claim, claim, key), (tok, acts, u)
        z = jnp.zeros((B, N))
        (uF, _, _, _), (toks, acts, useq) = jax.lax.scan(
            step, (z, z, jnp.zeros((B, N), bool), key), jnp.arange(T))
        return uF, toks, acts, useq

    return pol, rollout


def train(arm, N=6, B=512, c=0.5, iters=2500, lr=3e-3, seed=0, dmax=2):
    pol, rollout = make_rollout(NETS[arm], N, T, B, c)
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    params = pol.init(ki, jnp.zeros((1, N, F)))
    tx = optax.adam(lr); opt = tx.init(params)

    @jax.jit
    def upd(params, opt, key, ec):
        key, kd, ks, kr = jax.random.split(key, 4)
        Dn = jax.random.randint(kd, (B,), 0, dmax + 1)
        rank = jnp.argsort(jnp.argsort(jax.random.uniform(ks, (B, N)), -1), -1)
        def_mask = (rank < Dn[:, None]).astype(jnp.float32)
        uF, toks, acts, useq = rollout(params, def_mask, kr)
        coop = 1.0 - def_mask
        W = jax.vmap(coop_welfare, in_axes=(0, None))(useq, def_mask)
        adv_t = disc_rtg(W - jnp.concatenate([jnp.zeros((1, B)), W[:-1]], 0))
        adv_t = adv_t - adv_t.mean(1, keepdims=True)

        def loss(prm):
            lsm = jax.nn.log_softmax(pol.apply(prm, toks))
            lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
            ent = -(jnp.exp(lsm) * lsm).sum(-1)
            return -(adv_t[:, :, None] * lp * coop[None]).sum(-1).mean() \
                - ec * (ent * coop[None]).sum(-1).mean()
        g = jax.grad(loss)(params); u, opt = tx.update(g, opt)
        return optax.apply_updates(params, u), opt, key

    ent_hi, ent_lo = 0.05, 0.003
    for it in range(iters):
        ec = ent_hi * (1 - it / iters) + ent_lo * (it / iters)
        params, opt, key = upd(params, opt, key, ec)
    return params


def eval_at(arm, params, N, c, B=512, seed=123):
    _, rollout = make_rollout(NETS[arm], N, T, B, c)
    k0, k1, kr0, kr1 = jax.random.split(jax.random.PRNGKey(seed), 4)
    m0 = jnp.zeros((B, N))
    m1 = jax.nn.one_hot(jax.random.randint(k1, (B,), 0, N), N)
    e0 = evaluate(rollout(params, m0, kr0)[0], m0, T)
    e1 = evaluate(rollout(params, m1, kr1)[0], m1, T)
    return e0, e1


def br_audit(arm, coop_params, N, c, B=512, iters=1500, lr=3e-3, seed=7):
    net = NETS[arm]
    pol = net()
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    dp = pol.init(ki, jnp.zeros((1, N, F)))
    tx = optax.adam(lr); opt = tx.init(dp)

    def rollout(dp, d_idx, key):
        d_oh = jax.nn.one_hot(d_idx, N)

        def step(carry, t):
            u, cc, clp, key = carry
            tok = features7(u, cc, clp, t, T)
            key, ka, kb = jax.random.split(key, 3)
            ca = jax.random.categorical(ka, pol.apply(coop_params, tok))
            da = jax.random.categorical(kb, pol.apply(dp, tok))
            acts = jnp.where(d_oh.astype(bool), da, ca)
            claim = acts == 1
            u = graded_step(u, claim, c)
            return (u, cc + claim, claim, key), (tok, acts)
        z = jnp.zeros((B, N))
        (uF, _, _, _), (toks, acts) = jax.lax.scan(
            step, (z, z, jnp.zeros((B, N), bool), key), jnp.arange(T))
        return uF, toks, acts, d_oh

    @jax.jit
    def upd(dp, opt, key):
        key, kd, kr = jax.random.split(key, 3)
        d_idx = jax.random.randint(kd, (B,), 0, N)
        uF, toks, acts, d_oh = rollout(dp, d_idx, kr)
        R = (uF * d_oh).sum(-1); adv = R - R.mean()

        def loss(p):
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
            return -(adv[None, :, None] * lp * d_oh[None]).sum(-1).mean()
        g = jax.grad(loss)(dp); u, opt = tx.update(g, opt)
        return optax.apply_updates(dp, u), opt, key

    for _ in range(iters):
        dp, opt, key = upd(dp, opt, key)
    key, kd, kr = jax.random.split(key, 3)
    d_idx = jax.random.randint(kd, (B,), 0, N)
    uF, _, _, _ = rollout(dp, d_idx, kr)
    return evaluate(uF, jax.nn.one_hot(d_idx, N), T)[0]


def _load_params(arm, c, s):
    pol = NETS[arm]()
    init = pol.init(jax.random.PRNGKey(0), jnp.zeros((1, 6, F)))
    with open(f"{PDIR}/m1c_{arm}_c{c}_s{s}.msgpack", "rb") as fh:
        return flax.serialization.from_bytes(init, fh.read())


def probe(c=0.9, s=0, B=512):
    for arm in ["last", "lastmax"]:
        try:
            p = _load_params(arm, c, s)
        except FileNotFoundError:
            continue
        for Ne in NS_EVAL:
            _, rollout = make_rollout(NETS[arm], Ne, T, B, c)
            d_oh = jax.nn.one_hot(
                jax.random.randint(jax.random.PRNGKey(1), (B,), 0, Ne), Ne)
            _, _, acts, _ = rollout(p, d_oh, jax.random.PRNGKey(2))
            cl = np.asarray(acts == 1, float)
            coop = 1 - np.asarray(d_oh)
            qt = (cl * coop[None]).sum((1, 2)) / coop.sum()
            dec = [f"{qt[i*10:(i+1)*10].mean():.2f}" for i in range(10)]
            print(f"[probe] {arm} N={Ne} c={c}: claim rate by decile: {dec}")


def make_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(EXP_DIR, exist_ok=True)
    cells = {
        ("v1 tokens", "mean-only"): (
            [r for r in csv.DictReader(open("results/m1_fraction.csv"))
             if r["arm"] == "v1tok"], "tab:gray", "o", 0.5),
        ("v1 tokens", "max-susp."): (
            list(csv.DictReader(open("results/m1b_count_agg.csv"))),
            "#5277bd", "s", 0.8),
        ("+last-step", "mean-only"): (
            [r for r in csv.DictReader(open(OUT)) if r["arm"] == "last"],
            "#df8a2c", "^", 0.8),
        ("+last-step", "max-susp."): (
            [r for r in csv.DictReader(open(OUT)) if r["arm"] == "lastmax"],
            "#2f9e44", "*", 1.0),
    }
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=True)
    for ax, c in zip(axes, CS):
        for (tok, agg), (rws, col, mk, al) in cells.items():
            sel = [r for r in rws if abs(float(r["c"]) - c) < 1e-9]
            if not sel:
                continue
            ms = [np.mean([float(r[f"t{n}_rho"]) for r in sel])
                  for n in NS_EVAL]
            sd = [np.std([float(r[f"t{n}_rho"]) for r in sel])
                  for n in NS_EVAL]
            ax.errorbar(NS_EVAL, ms, yerr=sd, marker=mk, color=col,
                        label=f"{tok} / {agg}", alpha=al,
                        ms=10 if mk == "*" else 6, capsize=3)
        ax.plot(NS_EVAL, NS_EVAL, ls=":", c="grey", lw=1)
        ax.axhline(1.0, ls="--", c="tab:blue", lw=1)
        ax.axhline(2.5, ls="-.", c="#aa3333", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(NS_EVAL); ax.set_xticklabels(NS_EVAL)
        ax.set_xticks([], minor=True)
        ax.set_title(f"c = {c}", fontsize=10)
        ax.set_xlabel("eval team size $N$")
    axes[0].set_ylabel(r"lone-defector transfer $\rho$ (log)")
    axes[0].legend(fontsize=7)
    fig.suptitle("M1c: 2$\\times$2 factorial --- fast evidence channel "
                 "$\\times$ count-preserving aggregation (5 seeds; dash-dot = "
                 "target 2.5)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{EXP_DIR}/fig_m1c_factorial.png", dpi=160)
    plt.close(fig)
    print(f"wrote {EXP_DIR}/fig_m1c_factorial.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["last", "lastmax"], default="lastmax")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--br_iters", type=int, default=1500)
    a = ap.parse_args()
    if a.figs:
        make_figs(); return
    if a.probe:
        probe(); return
    cs, seeds = CS, SEEDS
    if a.smoke:
        cs, seeds, a.iters, a.br_iters = [0.9], [0], 150, 100
    os.makedirs("results", exist_ok=True)
    os.makedirs(PDIR, exist_ok=True)
    header = ["arm", "c", "seed", "br6_rho"] + \
        [f"t{n}_{k}" for n in NS_EVAL for k in ("rho", "eff")]
    done = set()
    if not a.smoke and os.path.exists(OUT):
        done = {(r["arm"], float(r["c"]), int(r["seed"]))
                for r in csv.DictReader(open(OUT))}
    for c in cs:
        for s in seeds:
            if (a.arm, c, s) in done:
                print(f"[{a.arm}] c={c} s={s}: done, skip", flush=True)
                continue
            p = train(a.arm, c=c, iters=a.iters, seed=s)
            br = br_audit(a.arm, p, 6, c, iters=a.br_iters, seed=700 + s)
            tr = {n: eval_at(a.arm, p, n, c) for n in NS_EVAL}
            if not a.smoke:
                with open(f"{PDIR}/m1c_{a.arm}_c{c}_s{s}.msgpack", "wb") as fh:
                    fh.write(flax.serialization.to_bytes(p))
            row = dict(arm=a.arm, c=c, seed=s, br6_rho=float(br),
                       **{f"t{n}_rho": float(tr[n][1][0]) for n in NS_EVAL},
                       **{f"t{n}_eff": float(tr[n][0][2]) for n in NS_EVAL})
            if not a.smoke:
                new = not os.path.exists(OUT)
                with open(OUT, "a", newline="") as f:
                    w = csv.DictWriter(f, header)
                    if new:
                        w.writeheader()
                    w.writerow(row); f.flush()
            print(f"[{a.arm}] c={c} s={s}: br6={br:.2f} rho@N=" +
                  "/".join(f"{tr[n][1][0]:.2f}" for n in NS_EVAL) +
                  " eff@N=" + "/".join(f"{tr[n][0][2]:.2f}" for n in NS_EVAL),
                  flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
