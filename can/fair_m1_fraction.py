"""M1 --- FRACTION-PARAMETERIZED (mean-field) tokens: kill the transfer fragility.

The game is anonymous/aggregative (payoffs depend on the claimer count m only),
so the mean-field-correct policy responds to FRACTIONS. The v1 tokens drift with
N (audit: notes/m1_n_dependence_audit.md): channels u/T, (u-mean)/T, (u-min)/T
shrink ~1/N, so an N=6-trained policy sees out-of-distribution inputs at N=24
and transfer collapses at high c (v1: rho(N=24, c=0.9) ~ 8.9).

ONE variable changed vs the v1 transfer experiment (fair_rerun (A), train(),
2500 updates, dmax=2, same net/hypers): the token parameterization. 8 channels,
all N-invariant in scale (clips noted in the audit):
  1. share      clip(N*u_i/sum(u), 0, 4)      (=1 under fairness at every N)
  2. zdev       clip((u_i-mean)/std, -3, 3)
  3. zmin       clip((u_i-min)/std, 0, 4)
  4. is_min     1[u_i = min]
  5. claim rate cc_i/t                        (already N-invariant)
  6. m_frac     m_{t-1}/N                     (claimed FRACTION, aggregate)
  7. crbar      mean_j cc_j/t                 (running claimed fraction)
  8. time       t/T

Evaluation: train at N=6 ONLY, zero-shot audit at N in {6,12,24,48}, all c,
5 seeds: D=0 efficiency + always-claim transfer rho (the v1 metric, and per M0
the committed adversary is the stronger audit) at every N; trained
best-response audit at N=6 (regression check vs v1 CIs). The v1-token control
arm runs the identical protocol. Trained params are saved for the --attn and
--mf probes.

    python -m can.fair_m1_fraction --arm frac          # sweep -> CSV (+params)
    python -m can.fair_m1_fraction --arm v1tok         # control arm
    python -m can.fair_m1_fraction --smoke
    python -m can.fair_m1_fraction --figs              # transfer figure
    python -m can.fair_m1_fraction --attn              # attention N-invariance
    python -m can.fair_m1_fraction --mf                # mean-field reference
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
from .fair_xattn import (XAttn, evaluate, disc_rtg, coop_welfare,
                         train as v1_train, eval_at as v1_eval_at,
                         best_response_audit as v1_br)

T = 100
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]
NS_EVAL = [6, 12, 24, 48]
F = 8
OUT = "results/m1_fraction.csv"
PDIR = "results/m1_params"
EXP_DIR = "experiments/m1_fraction_mf"


def features_frac(u, cc, m_prev, t, T):
    """(B,N)->(B,N,8) N-invariant fraction tokens (see module docstring)."""
    n = u.shape[-1]
    eps = 1e-6
    f32 = jnp.float32
    share = jnp.clip(n * u / (u.sum(-1, keepdims=True) + eps), 0.0, 4.0)
    std = u.std(-1, keepdims=True)
    zdev = jnp.clip((u - u.mean(-1, keepdims=True)) / (std + eps), -3.0, 3.0)
    zmin = jnp.clip((u - u.min(-1, keepdims=True)) / (std + eps), 0.0, 4.0)
    ismin = (u == u.min(-1, keepdims=True)).astype(f32)
    cr = cc / jnp.clip(t, 1.0, None)
    mfrac = jnp.broadcast_to((m_prev / n)[..., None], u.shape)
    crbar = jnp.broadcast_to(cr.mean(-1, keepdims=True), u.shape)
    tt = jnp.full_like(u, t / T)
    return jnp.stack([share, zdev, zmin, ismin, cr, mfrac, crbar, tt], -1)


def make_rollout_frac(N, T, B, c):
    """Mirror of fair_xattn.make_rollout (defectors always claim) on fraction
    tokens; carry tracks the previous step's claimer count for m_frac."""
    pol = XAttn()

    def rollout(params, def_mask, key):
        dm = def_mask.astype(bool)

        def step(carry, t):
            u, cc, mp, key = carry
            tok = features_frac(u, cc, mp, t, T)
            key, ka = jax.random.split(key)
            acts = jax.random.categorical(ka, pol.apply(params, tok))
            acts = jnp.where(dm, 1, acts)
            claim = acts == 1
            u = graded_step(u, claim, c)
            return (u, cc + claim, claim.sum(-1).astype(jnp.float32), key), \
                (tok, acts, u)
        z = jnp.zeros((B, N))
        (uF, _, _, _), (toks, acts, useq) = jax.lax.scan(
            step, (z, z, jnp.zeros(B), key), jnp.arange(T))
        return uF, toks, acts, useq

    return pol, rollout


def train_frac(N=6, T=100, B=512, c=0.5, iters=2500, lr=3e-3, seed=0, dmax=2):
    """fair_xattn.train verbatim, on fraction tokens."""
    pol, rollout = make_rollout_frac(N, T, B, c)
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


def eval_frac(params, N, c, B=512, seed=123):
    """Zero-shot eval at team size N: (D=0 rho/jain/eff, D=1 always-claim ...)."""
    _, rollout = make_rollout_frac(N, T, B, c)
    k0, k1, kr0, kr1 = jax.random.split(jax.random.PRNGKey(seed), 4)
    m0 = jnp.zeros((B, N))
    m1 = jax.nn.one_hot(jax.random.randint(k1, (B,), 0, N), N)
    e0 = evaluate(rollout(params, m0, kr0)[0], m0, T)
    e1 = evaluate(rollout(params, m1, kr1)[0], m1, T)
    return e0, e1


def br_audit_frac(coop_params, N, c, B=512, iters=1500, lr=3e-3, seed=7):
    """fair_xattn.best_response_audit on fraction tokens (XAttn defector at a
    random index vs the frozen cooperators)."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    dp = pol.init(ki, jnp.zeros((1, N, F)))
    tx = optax.adam(lr); opt = tx.init(dp)

    def rollout(dp, d_idx, key):
        d_oh = jax.nn.one_hot(d_idx, N)

        def step(carry, t):
            u, cc, mp, key = carry
            tok = features_frac(u, cc, mp, t, T)
            key, ka, kb = jax.random.split(key, 3)
            ca = jax.random.categorical(ka, pol.apply(coop_params, tok))
            da = jax.random.categorical(kb, pol.apply(dp, tok))
            acts = jnp.where(d_oh.astype(bool), da, ca)
            claim = acts == 1
            u = graded_step(u, claim, c)
            return (u, cc + claim, claim.sum(-1).astype(jnp.float32), key), \
                (tok, acts)
        z = jnp.zeros((B, N))
        (uF, _, _, _), (toks, acts) = jax.lax.scan(
            step, (z, z, jnp.zeros(B), key), jnp.arange(T))
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


# ------------------------- probes ------------------------------------------------
def _load_params(arm, c, s):
    pol = XAttn()
    f = F if arm == "frac" else 6
    init = pol.init(jax.random.PRNGKey(0), jnp.zeros((1, 6, f)))
    with open(f"{PDIR}/m1_{arm}_c{c}_s{s}.msgpack", "rb") as fh:
        return flax.serialization.from_bytes(init, fh.read())


def attn_stats(arm, c=0.9, s=0, B=256):
    """Verify (don't assume) attention N-invariance: row entropy and ctx norm of
    the trained policy on D=1 rollout states at each eval N."""
    from .fair_xattn import features as features_v1, make_rollout as mk_v1
    params = _load_params(arm, c, s)
    d = 64

    def attn(tok):                                              # manual XAttn fwd
        p = params["params"]
        def dense(name, x):
            return x @ p[name]["kernel"] + p[name]["bias"]
        Q, K, V = dense("Dense_0", tok), dense("Dense_1", tok), dense("Dense_2", tok)
        a = jax.nn.softmax(Q @ jnp.swapaxes(K, -1, -2) / jnp.sqrt(d), -1)
        return a, a @ V

    out = {}
    for Ne in NS_EVAL:
        if arm == "frac":
            _, rollout = make_rollout_frac(Ne, T, B, c)
        else:
            _, rollout = mk_v1(Ne, T, B, c)
        m1 = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(1), (B,), 0, Ne), Ne)
        _, toks, _, _ = rollout(params, m1, jax.random.PRNGKey(2))
        a, ctx = attn(toks)                                     # (T,B,N,N),(T,B,N,d)
        ent = -(a * jnp.log(a + 1e-12)).sum(-1).mean()          # nats
        out[Ne] = (float(ent), float(jnp.log(Ne)),
                   float(jnp.linalg.norm(ctx, axis=-1).mean()))
    print(f"[attn] arm={arm} c={c} s={s}: row entropy (vs ln N = uniform) and "
          f"|ctx|")
    for Ne, (e, u, cn) in out.items():
        print(f"  N={Ne:>2}: H={e:.3f} (uniform={u:.3f}, sharpness={1-e/u:.2f}) "
              f"|ctx|={cn:.3f}")


def mf_reference(c, D, N, qs=None):
    """Numeric finite-N mean-field reference: expected cooperator welfare per
    step when D scripted defectors always claim and each of the N-D cooperators
    contests i.i.d. with prob q. Returns (q*, welfare(q), rho(q))."""
    from math import comb
    nc = N - D
    qs = np.linspace(0, 1, 101) if qs is None else qs
    W, R = [], []
    for q in qs:
        pk = np.array([comb(nc, k) * q**k * (1-q)**(nc-k) for k in range(nc+1)])
        w = rho = 0.0
        for k, p in enumerate(pk):
            m = D + k
            if m == 0:
                w += p * 1.0 / nc                  # routed to worst-off (a coop)
            else:
                g = 1.0 if m == 1 else (1.0 - c) / m
                tot = 1.0 if m == 1 else (1.0 - c)  # delivered this step
                w += p * (k * g) / nc              # mean coop share
                rho += p * (g * N / tot) if D > 0 else 0.0
        W.append(w); R.append(rho)
    W, R = np.array(W), np.array(R)
    return qs[W.argmax()], (qs, W, R)


def mf_probe(arms=("frac",), c_list=(0.5, 0.9), s=0, B=512):
    """Compare the MF-optimal contest fraction q*(c, D, N) with the trained
    policy's empirical cooperator claim rate at forced D, per eval N."""
    print("[mf] numeric reference q* vs empirical cooperator claim rate")
    for c in c_list:
        for arm in arms:
            params = _load_params(arm, c, s)
            for Ne in NS_EVAL:
                _, rollout = make_rollout_frac(Ne, T, B, c)
                for D in [0, 1, 2]:
                    qstar, _ = mf_reference(c, D, Ne)
                    dm = (jnp.arange(Ne)[None, :] < D).astype(jnp.float32)
                    dm = jnp.broadcast_to(dm, (B, Ne))
                    _, _, acts, _ = rollout(params, dm, jax.random.PRNGKey(3))
                    cl = np.asarray(acts == 1, float)
                    coop = 1 - np.asarray(dm)
                    q_emp = float((cl * coop[None]).sum() / (T * coop.sum()))
                    print(f"  c={c} {arm} N={Ne:>2} D={D}: q*={qstar:.2f} "
                          f"q_emp={q_emp:.2f}")


def constfrac_eval(B=512):
    """Constant defector-FRACTION transfer: D = N/6 always-claim coalition at
    each eval N (the mean-field-preserving scaling, vs the lone-defector D=1 of
    the main sweep). Writes results/m1_constfrac.csv from the saved params."""
    out = "results/m1_constfrac.csv"
    rows = []
    for arm in ["frac", "v1tok"]:
        mk = make_rollout_frac if arm == "frac" else None
        for c in CS:
            for s in SEEDS:
                p = _load_params(arm, c, s)
                r = {"arm": arm, "c": c, "seed": s}
                for Ne in NS_EVAL:
                    D = Ne // 6
                    if arm == "frac":
                        rollout = make_rollout_frac(Ne, T, B, c)[1]
                    else:
                        from .fair_xattn import make_rollout as mk_v1
                        rollout = mk_v1(Ne, T, B, c)[1]
                    k1, k2 = jax.random.split(jax.random.PRNGKey(10 + s))
                    rank = jnp.argsort(jnp.argsort(
                        jax.random.uniform(k1, (B, Ne)), -1), -1)
                    dm = (rank < D).astype(jnp.float32)
                    uF = rollout(p, dm, k2)[0]
                    r[f"t{Ne}_rho"] = float(evaluate(uF, dm, T)[0])
                rows.append(r)
                print(f"[constfrac] {arm} c={c} s={s}: " +
                      "/".join(f"{r[f't{n}_rho']:.2f}" for n in NS_EVAL),
                      flush=True)
    hdr = ["arm", "c", "seed"] + [f"t{n}_rho" for n in NS_EVAL]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, hdr); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out}")


# ------------------------- figure ------------------------------------------------
def make_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(EXP_DIR, exist_ok=True)
    rows = list(csv.DictReader(open(OUT)))

    def series(arm, c, col):
        out = []
        for n in NS_EVAL:
            v = [float(r[f"t{n}_{col}"]) for r in rows
                 if r["arm"] == arm and abs(float(r["c"]) - c) < 1e-9]
            out.append((np.mean(v), np.std(v)))
        return out

    cf = None
    if os.path.exists("results/m1_constfrac.csv"):
        cf = list(csv.DictReader(open("results/m1_constfrac.csv")))

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=True)
    for ax, c in zip(axes, CS):
        for arm, col, lbl, mk in [("v1tok", "tab:gray", "v1 tokens", "o"),
                                  ("frac", "#2f9e44", "fraction tokens", "*")]:
            sr = series(arm, c, "rho")
            if not any(np.isfinite(m) for m, _ in sr):
                continue
            ms, sd = zip(*sr)
            ax.errorbar(NS_EVAL, ms, yerr=sd, marker=mk, color=col, label=lbl,
                        ms=10 if mk == "*" else 6, capsize=3)
            if cf is not None:
                ms2 = [np.mean([float(r[f"t{n}_rho"]) for r in cf
                                if r["arm"] == arm
                                and abs(float(r["c"]) - c) < 1e-9])
                       for n in NS_EVAL]
                ax.plot(NS_EVAL, ms2, ls="--", marker=mk, color=col, alpha=0.45,
                        ms=7 if mk == "*" else 4,
                        label=f"{lbl}, $D{{=}}N/6$ (const.\\ fraction)")
        ax.plot(NS_EVAL, NS_EVAL, ls=":", c="grey", lw=1, label="yield (=N)")
        ax.axhline(1.0, ls="--", c="tab:blue", lw=1)
        ax.axhline(2.5, ls="-.", c="#aa3333", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(NS_EVAL); ax.set_xticklabels(NS_EVAL)
        ax.set_xticks([], minor=True)
        ax.set_title(f"c = {c}", fontsize=10)
        ax.set_xlabel("eval team size $N$")
    axes[0].set_ylabel(r"always-claim transfer $\rho$ (log)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Zero-shot transfer of the $N{=}6$ policy: fraction vs v1 "
                 "tokens (5 seeds; dash-dot = M1 target 2.5)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{EXP_DIR}/fig_m1_transfer.png", dpi=160)
    plt.close(fig)
    print(f"wrote {EXP_DIR}/fig_m1_transfer.png")


# ------------------------- driver ------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["frac", "v1tok"], default="frac")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--attn", action="store_true")
    ap.add_argument("--mf", action="store_true")
    ap.add_argument("--constfrac", action="store_true")
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--br_iters", type=int, default=1500)
    a = ap.parse_args()
    if a.figs:
        make_figs(); return
    if a.attn:
        for arm in ["frac", "v1tok"]:
            attn_stats(arm)
        return
    if a.mf:
        mf_probe(); return
    if a.constfrac:
        constfrac_eval(); return
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
                print(f"[{a.arm}] c={c} s={s}: already done, skip", flush=True)
                continue
            if a.arm == "frac":
                p = train_frac(c=c, iters=a.iters, seed=s)
                br = br_audit_frac(p, 6, c, iters=a.br_iters, seed=700 + s)
                tr = {n: eval_frac(p, n, c) for n in NS_EVAL}
            else:
                _, _, p = v1_train(N=6, T=T, c=c, iters=a.iters, seed=s, dmax=2)
                br = v1_br(p, 6, T, c, iters=a.br_iters, seed=700 + s)
                tr = {n: v1_eval_at(p, n, T, c) for n in NS_EVAL}
            if not a.smoke:
                with open(f"{PDIR}/m1_{a.arm}_c{c}_s{s}.msgpack", "wb") as fh:
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
