"""M2b --- LEARNED COMMITMENT: close M2's credibility gap.

M2: scripted grim/tft hold learning defectors at rho = 1.00 at c = 1; league-
learned policies leave rho ~ 1.8. Hypothesis refinement: the v1 tokens cannot
REPRESENT an absorbing trigger --- their only memory is the claim-rate cc/t,
which forgives by dilution (a defector that claims then stops watches its own
flag decay), so no amount of training can make punishment absorbing. Grim's
bookkeeping, however, is a deterministic recursion over PUBLIC claim history,
so it can live in the feature extractor of a decentralized policy:

  off-step   : trig was off at t-1 (punishment steps don't count as offense)
  cc_off_i   : claims accumulated on off-steps;  cr_off = cc_off / n_off
  flag_i     : ABSORBING bit, set once cr_off_i > tau (tau = 0.15 < 1/N)
  trig       : any(flag)   (grim semantics)

Two extra channels (flag_i, trig) -> 8-channel tokens. Three arms separate
representation from optimization, plus a determinism probe:

  distill6 : behaviour-clone grim into XAttn on v1 6ch tokens. Expected to
             fail against stop-and-wait exploits (no absorbing memory) ---
             the representation diagnosis.
  distill8 : behaviour-clone grim into XAttn on 8ch tokens. Should reach
             grim parity (rho ~ 1.0) --- representation sufficient.
  league8  : league-train (M2 protocol, gamma=0.99) on 8ch tokens. Does RL
             LEARN commitment when absorption is representable? Also run at
             c=0.5 to check the flag channels do not break proportional
             contesting in the graded regime (one policy, both regimes).
  --probe  : argmax-evaluate the saved M2 CAN-deter params --- how much of
             the gap was sampling softness?

Audits: identical to M2 (fresh BR learner 1500 / always-claim / patient 3x +
D=0 efficiency and false-positive contest rate). BC teacher data mixes
adversary types (none / always / rate-p / burst-then-stop) so the stop-trick
is in-distribution for the distillation.

    python -m can.fair_m2b_commitment --arm distill6
    python -m can.fair_m2b_commitment --arm distill8
    python -m can.fair_m2b_commitment --arm league8
    python -m can.fair_m2b_commitment --probe
    python -m can.fair_m2b_commitment --smoke
    python -m can.fair_m2b_commitment --figs
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
from .fair_m2_deterrence import (audit, br_defector, always_claim_act,
                                 make_can_rollout, league_train_g)

N, T = 6, 100
TAU = 0.15
SEEDS = [0, 1, 2, 3, 4]
OUT = "results/m2b_commitment.csv"
PDIR = "results/m2b_params"
EXP_DIR = "experiments/m2b_commitment"
CS_ARM = {"distill6": [0.95, 1.0], "distill8": [0.95, 1.0],
          "league8": [0.5, 0.95, 1.0]}


def flag_update(state, claim):
    """Absorbing-trigger recursion over public claims. state = (cc_off, n_off,
    flag, trig); claim (B,N) this step's claims. Returns updated state."""
    cc_off, n_off, flag, trig = state
    off = ~trig                                       # prev-step trigger off
    cc_off = cc_off + claim * off[:, None]
    n_off = n_off + off
    cr_off = cc_off / jnp.clip(n_off, 1.0, None)[:, None]
    flag = flag | (cr_off > TAU)                      # absorbing
    return (cc_off, n_off, flag, flag.any(-1))


def features8(u, cc, fstate, t, T):
    flag, trig = fstate[2], fstate[3]
    return jnp.concatenate(
        [features(u, cc, t, T), flag.astype(jnp.float32)[..., None],
         jnp.broadcast_to(trig.astype(jnp.float32)[:, None, None],
                          u.shape + (1,))], -1)


def fstate0(B):
    return (jnp.zeros((B, N)), jnp.zeros(B), jnp.zeros((B, N), bool),
            jnp.zeros(B, bool))


def make_rollout8(params, c, B):
    """rf(def_act, def_mask, key, ret_traj) for an 8ch XAttn cooperator."""
    pol = XAttn()

    def rf(def_act, def_mask, key, ret_traj=False):
        dm = def_mask.astype(bool)

        def step(carry, t):
            u, cc, fstate, key = carry
            tok = features8(u, cc, fstate, t, T)
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


# ------------------------- BC distillation ---------------------------------------
def _teacher_rollout(channels, c, B, key):
    """Grim team (teacher) vs a per-episode adversary mix. Returns toks
    (T,B,N,ch), teacher labels (T,B,N), coop mask (B,N)."""
    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    typ = jax.random.randint(k1, (B,), 0, 4)          # none/always/rate/burst
    d_oh = jax.nn.one_hot(jax.random.randint(k2, (B,), 0, N), N)
    d_oh = d_oh * (typ != 0)[:, None]                 # type 0: D=0
    p_rate = jax.random.uniform(k3, (B,), minval=0.05, maxval=1.0)
    k_burst = jax.random.randint(k4, (B,), 5, 51)
    dm = d_oh.astype(bool)

    def step(carry, t):
        u, cc, fstate, key = carry
        flag, trig = fstate[2], fstate[3]
        if channels == 8:
            tok = features8(u, cc, fstate, t, T)
        else:
            tok = features(u, cc, t, T)
        teacher = jnp.broadcast_to(trig[:, None], (B, N)).astype(jnp.int32)
        key, kb = jax.random.split(key)
        p = jnp.where(typ == 1, 1.0,
                      jnp.where(typ == 2, p_rate,
                                (t < k_burst).astype(jnp.float32)))
        da = jax.random.bernoulli(kb, p[:, None], (B, N)).astype(jnp.int32)
        acts = jnp.where(dm, da, teacher)
        claim = acts == 1
        u = graded_step(u, claim, c)
        fstate = flag_update(fstate, claim)
        return (u, cc + claim, fstate, key), (tok, teacher)
    z = jnp.zeros((B, N))
    (_, _, _, _), (toks, labels) = jax.lax.scan(
        step, (z, z, fstate0(B), key), jnp.arange(T))
    return toks, labels, 1.0 - d_oh


def distill(channels, c, iters=2000, B=256, lr=1e-3, seed=0):
    """Behaviour-clone the grim teacher into XAttn on `channels` tokens."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    params = pol.init(ki, jnp.zeros((1, N, channels)))
    tx = optax.adam(lr); opt = tx.init(params)

    @jax.jit
    def upd(params, opt, key):
        key, kr = jax.random.split(key)
        toks, labels, coop = _teacher_rollout(channels, c, B, kr)

        def loss(p):
            lsm = jax.nn.log_softmax(pol.apply(p, toks))
            lp = jnp.take_along_axis(lsm, labels[..., None], -1)[..., 0]
            return -(lp * coop[None]).mean()
        g = jax.grad(loss)(params); u, opt = tx.update(g, opt)
        return optax.apply_updates(params, u), opt, key

    for _ in range(iters):
        params, opt, key = upd(params, opt, key)
    return params


# ------------------------- league on 8ch tokens ----------------------------------
def league8_train(c, generations=6, coop_iters=1200, br_iters=1000, B=512,
                  lr=3e-3, seed=0, dmax=2):
    """M2 league protocol (gamma=0.99) with 8ch absorbing-flag tokens for
    cooperators AND pool defectors."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed)
    key, kc, kd = jax.random.split(key, 3)
    coop_p = pol.init(kc, jnp.zeros((1, N, 8)))
    pool = [pol.init(kd, jnp.zeros((1, N, 8)))]

    def coop_vs_pool(coop_p, pool_stk, iters, key):
        P = jax.tree_util.tree_leaves(pool_stk)[0].shape[0]
        tx = optax.adam(lr); opt = tx.init(coop_p)

        def rollout(coop_p, def_mask, pidx, key):
            dm = def_mask.astype(bool)

            def step(carry, t):
                u, cc, fstate, key = carry
                tok = features8(u, cc, fstate, t, T)
                cl = pol.apply(coop_p, tok)
                dl_all = jax.vmap(lambda dp: pol.apply(dp, tok))(pool_stk)
                dl = dl_all[pidx, jnp.arange(B)]
                key, ka, kb = jax.random.split(key, 3)
                acts = jnp.where(dm, jax.random.categorical(kb, dl),
                                 jax.random.categorical(ka, cl))
                claim = acts == 1
                u = graded_step(u, claim, c)
                fstate = flag_update(fstate, claim)
                return (u, cc + claim, fstate, key), (tok, acts, u)
            z = jnp.zeros((B, N))
            (_, _, _, _), out = jax.lax.scan(
                step, (z, z, fstate0(B), key), jnp.arange(T))
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
            ac = disc_rtg(Wc - jnp.concatenate(
                [jnp.zeros((1, B)), Wc[:-1]], 0))
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
        rf = make_rollout8(coop_p, c, B)
        pool.append(br_defector(rf, c, br_iters, lr, seed * 100 + gen,
                                feat=8))
    return coop_p


# ------------------------- argmax probe ------------------------------------------
def make_can_rollout_greedy(params, c, B):
    pol = XAttn()

    def rf(def_act, def_mask, key, ret_traj=False):
        dm = def_mask.astype(bool)

        def step(carry, t):
            u, cc, key = carry
            tok = features(u, cc, t, T)
            key, kb = jax.random.split(key)
            ca = jnp.argmax(pol.apply(params, tok), -1)
            acts = jnp.where(dm, def_act(tok, kb), ca)
            claim = acts == 1
            u = graded_step(u, claim, c)
            return (u, cc + claim, key), (tok, acts, u, claim.sum(-1))
        z = jnp.zeros((B, N))
        (uF, _, _), (toks, acts, useq, ms) = jax.lax.scan(
            step, (z, z, key), jnp.arange(T))
        return (uF, toks, acts, useq, ms) if ret_traj else (uF, ms)
    return rf


def probe(B=512):
    """Argmax-evaluate the saved M2 CAN-deter params (br + always-claim only)."""
    pol = XAttn()
    init = pol.init(jax.random.PRNGKey(0), jnp.zeros((1, N, 6)))
    for c in [0.95, 1.0]:
        for s in SEEDS:
            f = f"results/m2_params/m2_CAN-deter_c{c}_s{s}.msgpack"
            if not os.path.exists(f):
                continue
            with open(f, "rb") as fh:
                p = flax.serialization.from_bytes(init, fh.read())
            for nm, rfac in [("sampled", make_can_rollout),
                             ("greedy", make_can_rollout_greedy)]:
                rf = rfac(p, c, B)
                m0 = jnp.zeros((B, N))
                uF, ms = rf(always_claim_act, m0, jax.random.PRNGKey(8))
                d0 = float(np.asarray(uF).sum(1).mean()) / T
                def_act = br_defector(rf, c, 1500, seed=900 + s,
                                      return_act=True)
                kd, kr = jax.random.split(jax.random.PRNGKey(9 + s))
                d_oh = jax.nn.one_hot(
                    jax.random.randint(kd, (B,), 0, N), N)
                uF, _ = rf(def_act, d_oh, kr)
                rho, _, _ = evaluate(uF, d_oh, T)
                print(f"[probe] CAN-deter c={c} s={s} {nm}: d0_eff={d0:.3f} "
                      f"br_rho={rho:.2f}", flush=True)


# ------------------------- figure ------------------------------------------------
def make_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(EXP_DIR, exist_ok=True)
    rows = list(csv.DictReader(open(OUT)))
    m2 = list(csv.DictReader(open("results/m2_deterrence.csv")))
    pols = [("CAN-deter (M2, 6ch)", m2, "CAN-deter"),
            ("distill6", rows, "distill6"), ("distill8", rows, "distill8"),
            ("league8", rows, "league8"), ("grim (script)", m2, "grim")]
    advs = [("br_rho", "(a) learner", "#2f9e44"),
            ("patient_rho", "(c) patient", "#5277bd")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, c in zip(axes, [0.95, 1.0]):
        xs = np.arange(len(pols))
        for i, (col, lbl, cl) in enumerate(advs):
            ms, sd = [], []
            for _, src, key in pols:
                v = [float(r[col]) for r in src
                     if r["policy"] == key and abs(float(r["c"]) - c) < 1e-9]
                ms.append(np.mean(v) if v else np.nan)
                sd.append(np.std(v) if v else 0)
            ax.bar(xs + (i - 0.5) * 0.35, ms, 0.35, yerr=sd, capsize=3,
                   label=lbl, color=cl, alpha=0.9)
        ax.axhline(1.0, ls="--", c="#888", lw=1)
        ax.axhline(1.5, ls="-.", c="#aa3333", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([p[0] for p in pols], fontsize=7.5, rotation=12)
        ax.set_title(f"c = {c}", fontsize=10)
    axes[0].set_ylabel(r"free-ride $\rho$ vs learning defectors")
    axes[0].legend(fontsize=8)
    fig.suptitle("M2b: closing the credibility gap --- representation "
                 "(distill) vs optimization (league) on absorbing-flag tokens",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{EXP_DIR}/fig_m2b_gap.png", dpi=160)
    plt.close(fig)
    lines = ["| policy | c | D=0 eff | FP | rho learner | rho always-claim | "
             "rho patient |", "|---|---|---|---|---|---|---|"]
    for nm, src, key in pols:
        for c in sorted({float(r["c"]) for r in src if r["policy"] == key}):
            sel = [r for r in src
                   if r["policy"] == key and abs(float(r["c"]) - c) < 1e-9]
            g = lambda k: np.mean([float(r[k]) for r in sel])
            lines.append(f"| {nm} | {c} | {g('d0_eff'):.3f} | {g('d0_fp'):.3f}"
                         f" | {g('br_rho'):.2f} | {g('ac_rho'):.2f} | "
                         f"{g('patient_rho'):.2f} |")
    with open(f"{EXP_DIR}/results_table.md", "w") as f:
        f.write("# M2b learned commitment --- audits\n\n"
                + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {EXP_DIR}/fig_m2b_gap.png and results_table.md")


# ------------------------- driver ------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["distill6", "distill8", "league8"],
                    default="league8")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--coop_iters", type=int, default=1200)
    ap.add_argument("--br_iters", type=int, default=1000)
    ap.add_argument("--audit_iters", type=int, default=1500)
    ap.add_argument("--distill_iters", type=int, default=2000)
    a = ap.parse_args()
    if a.figs:
        make_figs(); return
    if a.probe:
        probe(); return
    cs, seeds = CS_ARM[a.arm], SEEDS
    if a.smoke:
        cs, seeds = [1.0], [0]
        a.generations, a.coop_iters, a.br_iters = 2, 60, 50
        a.audit_iters, a.distill_iters = 80, 200
    os.makedirs("results", exist_ok=True)
    os.makedirs(PDIR, exist_ok=True)
    header = ["policy", "c", "seed", "d0_eff", "d0_fp", "br_rho", "br_d1eff",
              "ac_rho", "ac_d1eff", "patient_rho", "patient_d1eff"]
    done = set()
    if not a.smoke and os.path.exists(OUT):
        done = {(r["policy"], float(r["c"]), int(r["seed"]))
                for r in csv.DictReader(open(OUT))}
    B = 512
    for c in cs:
        for s in seeds:
            if (a.arm, c, s) in done:
                print(f"[{a.arm}] c={c} s={s}: done, skip", flush=True)
                continue
            if a.arm == "distill6":
                p = distill(6, c, iters=a.distill_iters, seed=s)
                rf = make_can_rollout(p, c, B)
            elif a.arm == "distill8":
                p = distill(8, c, iters=a.distill_iters, seed=s)
                rf = make_rollout8(p, c, B)
            else:
                p = league8_train(c, a.generations, a.coop_iters, a.br_iters,
                                  seed=s)
                rf = make_rollout8(p, c, B)
            if not a.smoke:
                with open(f"{PDIR}/m2b_{a.arm}_c{c}_s{s}.msgpack", "wb") as fh:
                    fh.write(flax.serialization.to_bytes(p))
            met = audit(rf, c, br_iters=a.audit_iters, seed=7 + s,
                        feat=6 if a.arm == "distill6" else 8)
            row = dict(policy=a.arm, c=c, seed=s,
                       **{k: float(v) for k, v in met.items()})
            if not a.smoke:
                new = not os.path.exists(OUT)
                with open(OUT, "a", newline="") as f:
                    w = csv.DictWriter(f, header)
                    if new:
                        w.writeheader()
                    w.writerow(row); f.flush()
            print(f"[{a.arm}] c={c} s={s}: d0_eff={met['d0_eff']:.3f} "
                  f"fp={met['d0_fp']:.3f} br={met['br_rho']:.2f} "
                  f"ac={met['ac_rho']:.2f} patient={met['patient_rho']:.2f}",
                  flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
