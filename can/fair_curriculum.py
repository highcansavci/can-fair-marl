"""Mixed-N curriculum (reviewer: convert the transfer fragility into a result).

Train CAN cycling the team size over {4,6,8,12} per iteration (the policy is
permutation-equivariant, so one set of weights handles all N) vs.\ the N=6-only
baseline, with the SAME total gradient steps, then evaluate zero-shot transfer to
N=6/12/24. Tests whether size diversity at train time closes the high-N, high-c
transfer gap.

    python -m can.fair_curriculum --smoke
    python -m can.fair_curriculum
"""
import os
import csv
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .fair_xattn import (XAttn, make_rollout, features, evaluate, disc_rtg,
                         coop_welfare, eval_at)

T = 100
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]
NS_TRAIN = [4, 6, 8, 12]
NS_EVAL = [6, 12, 24]
DMAX = 2


def train_curriculum(c, Ns, iters=3000, B=512, lr=3e-3, seed=0, dmax=DMAX):
    """Vanilla CAN training cycling team size over Ns (Ns=[6] -> N=6-only)."""
    pol = XAttn()
    key = jax.random.PRNGKey(seed); key, ki = jax.random.split(key)
    params = pol.init(ki, jnp.zeros((1, 6, 6)))               # N-agnostic weights
    tx = optax.adam(lr); opt = tx.init(params)
    rollouts = {n: make_rollout(n, T, B, c)[1] for n in Ns}

    def make_upd(n):
        rollout = rollouts[n]

        @jax.jit
        def upd(params, opt, key, ec):
            key, kd, ks, kr = jax.random.split(key, 4)
            Dn = jax.random.randint(kd, (B,), 0, dmax + 1)
            rank = jnp.argsort(jnp.argsort(jax.random.uniform(ks, (B, n)), -1), -1)
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
                pg = -(adv_t[:, :, None] * lp * coop[None]).sum(-1).mean()
                return pg - ec * (ent * coop[None]).sum(-1).mean()
            g = jax.grad(loss)(params); u, opt = tx.update(g, opt)
            return optax.apply_updates(params, u), opt, key
        return upd

    upds = {n: make_upd(n) for n in Ns}
    ent_hi, ent_lo = 0.05, 0.003
    for it in range(iters):
        ec = ent_hi * (1 - it / iters) + ent_lo * (it / iters)
        n = Ns[it % len(Ns)]
        params, opt, key = upds[n](params, opt, key, ec)
    return params


def transfer_rho(params, c):
    """Best-response-free always-claim transfer audit rho at N=6/12/24."""
    return {ne: eval_at(params, ne, T, c)[1][0] for ne in NS_EVAL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--iters", type=int, default=3000)
    a = ap.parse_args()
    cs, seeds = CS, SEEDS
    if a.smoke:
        cs, seeds, a.iters = [0.9], [0], 400
    os.makedirs("results", exist_ok=True)
    out = "results/curriculum.csv"
    if not a.smoke and os.path.exists(out):
        os.remove(out)
    header = ["regime", "c", "seed"] + [f"t{n}_rho" for n in NS_EVAL]
    for s in seeds:
        for c in cs:
            for regime, Ns in [("N6only", [6]), ("mixedN", NS_TRAIN)]:
                p = train_curriculum(c, Ns, iters=a.iters, seed=s)
                tr = transfer_rho(p, c)
                row = dict(regime=regime, c=c, seed=s,
                           **{f"t{n}_rho": float(tr[n]) for n in NS_EVAL})
                if not a.smoke:
                    new = not os.path.exists(out)
                    with open(out, "a", newline="") as f:
                        w = csv.DictWriter(f, header)
                        if new:
                            w.writeheader()
                        w.writerow(row); f.flush()
                print(f"[{regime}] c={c} s={s}: transfer rho 6/12/24="
                      f"{tr[6]:.2f}/{tr[12]:.2f}/{tr[24]:.2f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
