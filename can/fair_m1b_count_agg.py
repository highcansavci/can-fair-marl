"""M1b --- COUNT-PRESERVING AGGREGATION: break the 1/N evidence scaling.

M1 located the lone-defector transfer failure: every statistic CAN forms is a
convex combination over N tokens, so one deviating agent's footprint is ~1/N
and detection latency grows with N (silent first ~30 steps at N=48, c=0.9,
during which the sole-claiming defector banks full units). Tokens are not the
constraint (M1); the AGGREGATION is. Design: notes/m1b_count_aggregation_design.md.

ONE variable changed vs the M1 v1tok control (same v1 tokens, same vanilla
training, 2500 updates, dmax=2): the policy adds a max-pooled per-agent
"suspicion" branch ---

    s_i = tanh(Dense(tanh(Dense(x_i))))          # (N, ds) per-agent embedding
    g   = [max_i s_i ; mean_i s_i]               # count-preserving global stats
    logits_i = head([x_i ; ctx_i ; g])

max_i responds Theta(1) to ONE outlier at any N (a max is exactly as sensitive
to one defector-like token at N=48 as at N=6); mean_i keeps the D=0 statistics.
The attention block is untouched.

Evaluation = M1 protocol verbatim: train at N=6 only, lone-defector
always-claim transfer rho + D=0 eff at N in {6,12,24,48}, trained-BR audit at
N=6, 5 seeds, c in {0.3..0.9}. Control = arm v1tok in results/m1_fraction.csv.
--probe reruns the onset-latency (claim rate by decile) mechanism check.

    python -m can.fair_m1b_count_agg               # sweep -> CSV (+params)
    python -m can.fair_m1b_count_agg --smoke
    python -m can.fair_m1b_count_agg --figs
    python -m can.fair_m1b_count_agg --probe
"""
import os
import csv
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import flax.serialization

from .fair_graded import graded_step
from .fair_xattn import features, evaluate, disc_rtg, coop_welfare

T = 100
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]
NS_EVAL = [6, 12, 24, 48]
OUT = "results/m1b_count_agg.csv"
PDIR = "results/m1b_params"
EXP_DIR = "experiments/m1b_count_agg"


class XAttnMax(nn.Module):
    """v1 cross-attention + a count-preserving (max-pooled) suspicion branch."""
    hidden: int = 64
    ds: int = 16

    @nn.compact
    def __call__(self, tok):                                    # (...,N,F)
        d = self.hidden
        Q, K, V = nn.Dense(d)(tok), nn.Dense(d)(tok), nn.Dense(d)(tok)
        a = nn.softmax(Q @ jnp.swapaxes(K, -1, -2) / jnp.sqrt(d), -1)
        ctx = a @ V
        s = nn.tanh(nn.Dense(self.ds)(nn.tanh(nn.Dense(self.ds)(tok))))
        g = jnp.concatenate([s.max(-2), s.mean(-2)], -1)        # (...,2ds)
        g = jnp.broadcast_to(g[..., None, :], tok.shape[:-1] + (2 * self.ds,))
        h = nn.tanh(nn.Dense(d)(jnp.concatenate([tok, ctx, g], -1)))
        return nn.Dense(2)(h)                                   # (...,N,2)


def make_rollout(N, T, B, c):
    pol = XAttnMax()

    def rollout(params, def_mask, key):
        dm = def_mask.astype(bool)

        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            key, ka = jax.random.split(key)
            acts = jax.random.categorical(ka, pol.apply(params, tok))
            acts = jnp.where(dm, 1, acts)
            claim = acts == 1
            u = graded_step(u, claim, c)
            return (u, cc + claim, key), (tok, acts, u)
        z = jnp.zeros((B, N))
        (uF, _, _), (toks, acts, useq) = jax.lax.scan(
            step, (z, z, key), jnp.arange(T))
        return uF, toks, acts, useq

    return pol, rollout


def train(N=6, B=512, c=0.5, iters=2500, lr=3e-3, seed=0, dmax=2):
    """fair_xattn.train verbatim on the XAttnMax policy."""
    pol, rollout = make_rollout(N, T, B, c)
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    params = pol.init(ki, jnp.zeros((1, N, 6)))
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


def eval_at(params, N, c, B=512, seed=123):
    _, rollout = make_rollout(N, T, B, c)
    k0, k1, kr0, kr1 = jax.random.split(jax.random.PRNGKey(seed), 4)
    m0 = jnp.zeros((B, N))
    m1 = jax.nn.one_hot(jax.random.randint(k1, (B,), 0, N), N)
    e0 = evaluate(rollout(params, m0, kr0)[0], m0, T)
    e1 = evaluate(rollout(params, m1, kr1)[0], m1, T)
    return e0, e1


def br_audit(coop_params, N, c, B=512, iters=1500, lr=3e-3, seed=7):
    """Best-response XAttnMax defector vs the frozen team (v1 protocol)."""
    pol = XAttnMax()
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    dp = pol.init(ki, jnp.zeros((1, N, 6)))
    tx = optax.adam(lr); opt = tx.init(dp)

    def rollout(dp, d_idx, key):
        d_oh = jax.nn.one_hot(d_idx, N)

        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            key, ka, kb = jax.random.split(key, 3)
            ca = jax.random.categorical(ka, pol.apply(coop_params, tok))
            da = jax.random.categorical(kb, pol.apply(dp, tok))
            acts = jnp.where(d_oh.astype(bool), da, ca)
            claim = acts == 1
            u = graded_step(u, claim, c)
            return (u, cc + claim, key), (tok, acts)
        z = jnp.zeros((B, N))
        (uF, _, _), (toks, acts) = jax.lax.scan(step, (z, z, key),
                                                jnp.arange(T))
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


def _load_params(c, s):
    pol = XAttnMax()
    init = pol.init(jax.random.PRNGKey(0), jnp.zeros((1, 6, 6)))
    with open(f"{PDIR}/m1b_c{c}_s{s}.msgpack", "rb") as fh:
        return flax.serialization.from_bytes(init, fh.read())


def probe(c=0.9, s=0, B=512):
    """Onset-latency mechanism check: coop claim rate by episode decile."""
    p = _load_params(c, s)
    for Ne in NS_EVAL:
        _, rollout = make_rollout(Ne, T, B, c)
        d_oh = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(1), (B,), 0, Ne), Ne)
        _, _, acts, _ = rollout(p, d_oh, jax.random.PRNGKey(2))
        cl = np.asarray(acts == 1, float)
        coop = 1 - np.asarray(d_oh)
        qt = (cl * coop[None]).sum((1, 2)) / coop.sum()
        dec = [f"{qt[i*10:(i+1)*10].mean():.2f}" for i in range(10)]
        print(f"[probe] maxagg N={Ne} c={c}: coop claim rate by decile: {dec}")


def make_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(EXP_DIR, exist_ok=True)
    rows = list(csv.DictReader(open(OUT)))
    ctrl = [r for r in csv.DictReader(open("results/m1_fraction.csv"))
            if r["arm"] == "v1tok"]

    def series(rws, c):
        out = []
        for n in NS_EVAL:
            v = [float(r[f"t{n}_rho"]) for r in rws
                 if abs(float(r["c"]) - c) < 1e-9]
            out.append((np.mean(v), np.std(v)))
        return out

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=True)
    for ax, c in zip(axes, CS):
        for rws, col, lbl, mk in [(ctrl, "tab:gray", "v1 (mean-only agg.)", "o"),
                                  (rows, "#2f9e44", "max-suspicion agg.", "*")]:
            ms, sd = zip(*series(rws, c))
            ax.errorbar(NS_EVAL, ms, yerr=sd, marker=mk, color=col, label=lbl,
                        ms=10 if mk == "*" else 6, capsize=3)
        ax.plot(NS_EVAL, NS_EVAL, ls=":", c="grey", lw=1, label="yield (=N)")
        ax.axhline(1.0, ls="--", c="tab:blue", lw=1)
        ax.axhline(2.5, ls="-.", c="#aa3333", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(NS_EVAL); ax.set_xticklabels(NS_EVAL)
        ax.set_xticks([], minor=True)
        ax.set_title(f"c = {c}", fontsize=10)
        ax.set_xlabel("eval team size $N$")
    axes[0].set_ylabel(r"lone-defector transfer $\rho$ (log)")
    axes[0].legend(fontsize=7)
    fig.suptitle("M1b: count-preserving (max) aggregation vs mean-only "
                 "aggregation, lone-defector transfer (5 seeds; dash-dot = "
                 "target 2.5)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{EXP_DIR}/fig_m1b_transfer.png", dpi=160)
    plt.close(fig)
    print(f"wrote {EXP_DIR}/fig_m1b_transfer.png")


def main():
    ap = argparse.ArgumentParser()
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
        done = {(float(r["c"]), int(r["seed"]))
                for r in csv.DictReader(open(OUT))}
    for c in cs:
        for s in seeds:
            if (c, s) in done:
                print(f"[maxagg] c={c} s={s}: done, skip", flush=True)
                continue
            p = train(c=c, iters=a.iters, seed=s)
            br = br_audit(p, 6, c, iters=a.br_iters, seed=700 + s)
            tr = {n: eval_at(p, n, c) for n in NS_EVAL}
            if not a.smoke:
                with open(f"{PDIR}/m1b_c{c}_s{s}.msgpack", "wb") as fh:
                    fh.write(flax.serialization.to_bytes(p))
            row = dict(arm="maxagg", c=c, seed=s, br6_rho=float(br),
                       **{f"t{n}_rho": float(tr[n][1][0]) for n in NS_EVAL},
                       **{f"t{n}_eff": float(tr[n][0][2]) for n in NS_EVAL})
            if not a.smoke:
                new = not os.path.exists(OUT)
                with open(OUT, "a", newline="") as f:
                    w = csv.DictWriter(f, header)
                    if new:
                        w.writeheader()
                    w.writerow(row); f.flush()
            print(f"[maxagg] c={c} s={s}: br6={br:.2f} rho@N=" +
                  "/".join(f"{tr[n][1][0]:.2f}" for n in NS_EVAL) +
                  " eff@N=" + "/".join(f"{tr[n][0][2]:.2f}" for n in NS_EVAL),
                  flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
