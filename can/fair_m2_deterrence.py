"""M2 --- c=1 DETERRENCE: punishment, not surplus capture.

At c=1 (all-or-nothing), contesting returns (1-c)/2 = 0 to the contester, so
Prop. 1's leverage is gone. But contesting is COSTLESS to cooperators (their
welfare excludes the defector; the destroyed unit was going to the defector
anyway) and MAXIMALLY PUNITIVE to the free-rider (1 -> 0). Against an adaptive
(learning) defector, credible punishment can deter; against a committed
always-claim script, c=1 remains hopeless by Prop. 1 --- that is the boundary,
not a failure. The payoff for successful deterrence arrives through the m=0
branch: a deterred defector stops claiming, and unclaimed units route
losslessly to the worst-off cooperator.

Policies (game/adversary + credit attribution only; v1 net + tokens unchanged):
  CAN-v1    : league exactly as v1 (gamma=0.99) at c in {0.95, 1.0} (control).
  CAN-deter : identical league with FULL-HORIZON credit gamma=1.0, so the
              within-episode return of punishing now (claims deterred + m=0
              routing later) is not discounted away.
  grim      : scripted bracket. Cooperators yield by default (at c=1 all-yield
              is D=0-optimal: every unit routes to the worst-off). An agent is
              FLAGGED when its OFFENSIVE claim-rate (claims made on
              non-punishment steps) exceeds tau --- punishment-phase claims do
              not count, so punishers never flag each other. Once anyone is
              flagged: all cooperators contest every step, forever.
  tft-k     : as grim, but contest only while some flagged agent has claimed
              within the last k steps (forgiving; a flagged agent that keeps
              claiming keeps the punishment alive).
  Note the evasion cap: a defector claiming just under tau is never flagged
  and banks ~ tau*T sole-claim units, so scripted deterrence concedes
  rho <= N*tau by construction (tau=0.15 -> 0.9).

Audits (three adversary classes + the M0 dual-audit rule):
  (a) fresh best-response learner, 1500 updates  (deterrence should bind here)
  (b) scripted always-claim                       (Prop. 1 boundary; at c=1 a
      grim team forces rho -> N WITH eff -> 0: levelling-down, by design)
  (c) PATIENT best-response learner, 3x budget    (can patience wait out the
                                                   punishment?)
Cost axis: D=0 efficiency and false-positive contest rate P(m >= 2 | D=0)
--- at c=1 every contested step is pure loss.

    python -m can.fair_m2_deterrence --arm v1            # league control
    python -m can.fair_m2_deterrence --arm deter         # gamma=1 league
    python -m can.fair_m2_deterrence --arm scripted      # grim + tft-k audits
    python -m can.fair_m2_deterrence --smoke
    python -m can.fair_m2_deterrence --figs
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
from .fair_xattn import XAttn, features, evaluate, coop_welfare

N, T = 6, 100
CS = [0.95, 1.0]
SEEDS = [0, 1, 2, 3, 4]
TAU = 0.15                     # flag threshold on offensive claim-rate
K_TFT = 10                     # tit-for-tat punishment window
OUT = "results/m2_deterrence.csv"
PDIR = "results/m2_params"
EXP_DIR = "experiments/m2_deterrence"


def disc_rtg_g(r, gamma):
    def f(carry, x):
        g = x + gamma * carry
        return g, g
    _, G = jax.lax.scan(f, jnp.zeros(r.shape[1:]), r, reverse=True)
    return G


# ------------------------- rollout factories -------------------------------------
# A rollout function rf(def_act, def_mask, key, ret_traj) rolls cooperators of a
# fixed policy against defector rows acting via def_act(tok, key) -> acts.
def make_can_rollout(params, c, B):
    pol = XAttn()

    def rf(def_act, def_mask, key, ret_traj=False):
        dm = def_mask.astype(bool)

        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            key, ka, kb = jax.random.split(key, 3)
            ca = jax.random.categorical(ka, pol.apply(params, tok))
            acts = jnp.where(dm, def_act(tok, kb), ca)
            claim = acts == 1
            u = graded_step(u, claim, c)
            return (u, cc + claim, key), (tok, acts, u, claim.sum(-1))
        z = jnp.zeros((B, N))
        (uF, _, _), (toks, acts, useq, ms) = jax.lax.scan(
            step, (z, z, key), jnp.arange(T))
        return (uF, toks, acts, useq, ms) if ret_traj else (uF, ms)
    return rf


def make_scripted_rollout(kind, c, B, tau=TAU, k=K_TFT):
    """grim / tft-k trigger team. Flags from OFFENSIVE claims only (claims on
    steps where the trigger was off), so punishers never flag each other; a
    flagged agent's claims during punishment extend the tft window."""

    def rf(def_act, def_mask, key, ret_traj=False):
        dm = def_mask.astype(bool)

        def step(carry, t):
            u, cc, cc_off, n_off, last_cl, key = carry
            tok = features(u, cc, t, T)
            cr_off = cc_off / jnp.clip(n_off, 1.0, None)[:, None]
            flagged = cr_off > tau                          # (B,N)
            if kind == "grim":
                trig = flagged.any(-1)                      # (B,) absorbing
            else:
                recent = (t - last_cl) <= k
                trig = (flagged & recent).any(-1)
            ca = jnp.broadcast_to(trig[:, None], (B, N)).astype(jnp.int32)
            key, kb = jax.random.split(key)
            acts = jnp.where(dm, def_act(tok, kb), ca)
            claim = acts == 1
            u = graded_step(u, claim, c)
            off = ~trig                                     # non-punish step
            cc_off = cc_off + claim * off[:, None]
            n_off = n_off + off
            last_cl = jnp.where(claim, t, last_cl)
            return (u, cc + claim, cc_off, n_off, last_cl, key), \
                (tok, acts, u, claim.sum(-1))
        z = jnp.zeros((B, N))
        init = (z, z, z, jnp.zeros(B), jnp.full((B, N), -1e9), key)
        (uF, _, _, _, _, _), (toks, acts, useq, ms) = jax.lax.scan(
            step, init, jnp.arange(T))
        return (uF, toks, acts, useq, ms) if ret_traj else (uF, ms)
    return rf


def always_claim_act(tok, key):
    return jnp.ones(tok.shape[:-1], jnp.int32)


# ------------------------- best-response defector --------------------------------
def br_defector(rf, c, iters, lr=3e-3, seed=7, B=512, return_act=False):
    """Best-response XAttn defector at a random index vs the frozen team rf."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    dp = pol.init(ki, jnp.zeros((1, N, 6)))
    tx = optax.adam(lr); opt = tx.init(dp)

    @jax.jit
    def upd(dp, opt, key):
        key, kd, kr = jax.random.split(key, 3)
        d_oh = jax.nn.one_hot(jax.random.randint(kd, (B,), 0, N), N)

        def def_act(tok, kk):
            return jax.random.categorical(kk, pol.apply(dp, tok))
        uF, toks, acts, _, _ = rf(def_act, d_oh, kr, ret_traj=True)
        R = (uF * d_oh).sum(-1); adv = R - R.mean()

        def loss(p):
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            lp = jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]
            return -(adv[None, :, None] * lp * d_oh[None]).sum(-1).mean()
        g = jax.grad(loss)(dp); u, opt = tx.update(g, opt)
        return optax.apply_updates(dp, u), opt, key

    for _ in range(iters):
        dp, opt, key = upd(dp, opt, key)
    if not return_act:
        return dp
    pol2 = XAttn()

    def act(tok, kk):
        return jax.random.categorical(kk, pol2.apply(dp, tok))
    return act


# ------------------------- league training (gamma knob) --------------------------
def league_train_g(c, gamma, generations=6, coop_iters=1200, br_iters=1000,
                   B=512, lr=3e-3, seed=0, dmax=2):
    """fair_xattn.league_train with a discount knob for the cooperator credit."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    key, kc, kd = jax.random.split(key, 3)
    coop_p = pol.init(kc, jnp.zeros((1, N, 6)))
    pool = [pol.init(kd, jnp.zeros((1, N, 6)))]

    def coop_vs_pool(coop_p, pool_stk, iters, key):
        P = jax.tree_util.tree_leaves(pool_stk)[0].shape[0]
        tx = optax.adam(lr); opt = tx.init(coop_p)

        def rollout(coop_p, def_mask, pidx, key):
            dm = def_mask.astype(bool)

            def step(carry, t):
                u, cc, key = carry
                tok = features(u, cc, t, T)
                cl = pol.apply(coop_p, tok)
                dl_all = jax.vmap(lambda dp: pol.apply(dp, tok))(pool_stk)
                dl = dl_all[pidx, jnp.arange(B)]
                key, ka, kb = jax.random.split(key, 3)
                acts = jnp.where(dm, jax.random.categorical(kb, dl),
                                 jax.random.categorical(ka, cl))
                u = graded_step(u, acts == 1, c)
                return (u, cc + (acts == 1), key), (tok, acts, u)
            (uF, _, _), out = jax.lax.scan(
                step, (jnp.zeros((B, N)), jnp.zeros((B, N)), key),
                jnp.arange(T))
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
            ac = disc_rtg_g(
                Wc - jnp.concatenate([jnp.zeros((1, B)), Wc[:-1]], 0), gamma)
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

    for gen in range(generations):
        pool_stk = jax.tree.map(lambda *xs: jnp.stack(xs), *pool)
        key, k1 = jax.random.split(key)
        coop_p = coop_vs_pool(coop_p, pool_stk, coop_iters, k1)
        rf = make_can_rollout(coop_p, c, B)
        pool.append(br_defector(rf, c, br_iters, lr, seed * 100 + gen))
    return coop_p


# ------------------------- audits ------------------------------------------------
def audit(rf, c, B=512, br_iters=1500, seed=7):
    """Three adversary classes + cost axis. Returns dict of metrics."""
    out = {}
    m0 = jnp.zeros((B, N))
    uF, ms = rf(always_claim_act, m0, jax.random.PRNGKey(seed + 1))
    out["d0_eff"] = float(np.asarray(uF).sum(1).mean()) / T
    out["d0_fp"] = float((np.asarray(ms) >= 2).mean())
    kd, kr = jax.random.split(jax.random.PRNGKey(seed + 2))
    d_oh = jax.nn.one_hot(jax.random.randint(kd, (B,), 0, N), N)
    uF, _ = rf(always_claim_act, d_oh, kr)
    out["ac_rho"], _, out["ac_d1eff"] = evaluate(uF, d_oh, T)
    for name, mult in [("br", 1), ("patient", 3)]:
        def_act = br_defector(rf, c, br_iters * mult, seed=seed + 700 + mult,
                              return_act=True)
        kd, kr = jax.random.split(jax.random.PRNGKey(seed + 3 + mult))
        d_oh = jax.nn.one_hot(jax.random.randint(kd, (B,), 0, N), N)
        uF, _ = rf(def_act, d_oh, kr)
        rho, _, eff = evaluate(uF, d_oh, T)
        out[f"{name}_rho"] = rho
        out[f"{name}_d1eff"] = eff
    return out


# ------------------------- figure ------------------------------------------------
def make_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(EXP_DIR, exist_ok=True)
    rows = list(csv.DictReader(open(OUT)))
    pols = ["CAN-v1", "CAN-deter", "grim", "tft-k"]
    advs = [("br_rho", "(a) learner"), ("ac_rho", "(b) always-claim"),
            ("patient_rho", "(c) patient learner")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, c in zip(axes, CS):
        xs = np.arange(len(pols))
        w = 0.25
        for i, (col, lbl) in enumerate(advs):
            ms, sd = [], []
            for p in pols:
                v = [float(r[col]) for r in rows
                     if r["policy"] == p and abs(float(r["c"]) - c) < 1e-9]
                ms.append(np.mean(v) if v else np.nan)
                sd.append(np.std(v) if v else 0)
            ax.bar(xs + (i - 1) * w, ms, w, yerr=sd, capsize=3,
                   label=lbl, color=["#2f9e44", "#d05c5c", "#5277bd"][i],
                   alpha=0.9)
        ax.axhline(1.0, ls="--", c="#888", lw=1)
        ax.axhline(1.5, ls="-.", c="#aa3333", lw=0.8)
        ax.axhline(N, ls=":", c="#888", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(pols, fontsize=9)
        ax.set_title(f"c = {c}", fontsize=10)
    axes[0].set_ylabel(r"free-ride $\rho$  (target $\leq$1.5; $N$=6 = unprotected)")
    axes[0].legend(fontsize=8)
    fig.suptitle("M2: deterrence at the all-or-nothing boundary, three adversary "
                 "classes (5 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{EXP_DIR}/fig_m2_threeway.png", dpi=160)
    plt.close(fig)
    lines = ["| policy | c | D=0 eff | FP contest rate | rho learner | "
             "rho always-claim (D=1 eff) | rho patient |",
             "|---|---|---|---|---|---|---|"]
    for p in pols:
        for c in CS:
            sel = [r for r in rows
                   if r["policy"] == p and abs(float(r["c"]) - c) < 1e-9]
            if not sel:
                continue
            g = lambda kk: np.mean([float(r[kk]) for r in sel])
            lines.append(
                f"| {p} | {c} | {g('d0_eff'):.3f} | {g('d0_fp'):.3f} | "
                f"{g('br_rho'):.2f} | {g('ac_rho'):.2f} ({g('ac_d1eff'):.2f}) | "
                f"{g('patient_rho'):.2f} |")
    with open(f"{EXP_DIR}/results_table.md", "w") as f:
        f.write("# M2 deterrence --- three-adversary audit\n\n"
                + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {EXP_DIR}/fig_m2_threeway.png and results_table.md")


# ------------------------- driver ------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["v1", "deter", "scripted"], default="v1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--coop_iters", type=int, default=1200)
    ap.add_argument("--br_iters", type=int, default=1000)
    ap.add_argument("--audit_iters", type=int, default=1500)
    a = ap.parse_args()
    if a.figs:
        make_figs(); return
    cs, seeds = CS, SEEDS
    if a.smoke:
        cs, seeds = [1.0], [0]
        a.generations, a.coop_iters, a.br_iters, a.audit_iters = 2, 60, 50, 80
    os.makedirs("results", exist_ok=True)
    os.makedirs(PDIR, exist_ok=True)
    header = ["policy", "c", "seed", "d0_eff", "d0_fp", "br_rho", "br_d1eff",
              "ac_rho", "ac_d1eff", "patient_rho", "patient_d1eff"]
    done = set()
    if not a.smoke and os.path.exists(OUT):
        done = {(r["policy"], float(r["c"]), int(r["seed"]))
                for r in csv.DictReader(open(OUT))}
    arms = {"v1": [("CAN-v1", 0.99)], "deter": [("CAN-deter", 1.0)],
            "scripted": [("grim", None), ("tft-k", None)]}[a.arm]
    B = 512
    for pname, gamma in arms:
        for c in cs:
            for s in seeds:
                if (pname, c, s) in done:
                    print(f"[{pname}] c={c} s={s}: done, skip", flush=True)
                    continue
                if gamma is not None:
                    p = league_train_g(c, gamma, a.generations, a.coop_iters,
                                       a.br_iters, seed=s)
                    if not a.smoke:
                        with open(f"{PDIR}/m2_{pname}_c{c}_s{s}.msgpack",
                                  "wb") as fh:
                            fh.write(flax.serialization.to_bytes(p))
                    rf = make_can_rollout(p, c, B)
                else:
                    rf = make_scripted_rollout(
                        "grim" if pname == "grim" else "tft", c, B)
                met = audit(rf, c, br_iters=a.audit_iters, seed=7 + s)
                row = dict(policy=pname, c=c, seed=s,
                           **{k: float(v) for k, v in met.items()})
                if not a.smoke:
                    new = not os.path.exists(OUT)
                    with open(OUT, "a", newline="") as f:
                        w = csv.DictWriter(f, header)
                        if new:
                            w.writeheader()
                        w.writerow(row); f.flush()
                print(f"[{pname}] c={c} s={s}: d0_eff={met['d0_eff']:.3f} "
                      f"fp={met['d0_fp']:.3f} br={met['br_rho']:.2f} "
                      f"ac={met['ac_rho']:.2f} patient={met['patient_rho']:.2f}",
                      flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
