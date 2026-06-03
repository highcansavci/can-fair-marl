"""Re-run all paper-2 experiments with the new BOUNDED free-ride metric
(defector share of delivered total / fair share, in [0,N]). Saves incrementally:
  results/rerun_vanilla.csv     (vanilla CAN: transfer + efficiency + baseline)
  results/rerun_league.csv      (league/PSRO CAN: robustness vs c)
  results/rerun_reliability.csv (single + population co-training, c=0.3/0.5)
"""
import os
import csv

from .fair_xattn import (train, league_train, cotrain, cotrain_pop,
                         best_response_audit, eval_at)

N, T = 6, 100
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]
NS = [6, 12, 24]


def wrow(path, row, header):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, header)
        if new:
            w.writeheader()
        w.writerow(row); f.flush()


def main():
    os.makedirs("results", exist_ok=True)
    for p in ["rerun_vanilla.csv", "rerun_league.csv", "rerun_reliability.csv"]:
        fp = f"results/{p}"
        if os.path.exists(fp):
            os.remove(fp)

    # (A) vanilla CAN: transfer (bounded) + D=0 efficiency + best-response baseline
    hv = ["c", "seed", "d0_eff", "d1_rho", "br_rho"] + \
         [f"t{n}_rho" for n in NS] + [f"t{n}_eff" for n in NS]
    for c in CS:
        for s in SEEDS:
            _, _, p = train(N=N, T=T, c=c, iters=2500, seed=s, dmax=2)
            br = best_response_audit(p, N, T, c, iters=1500, seed=700 + s)
            tr = {n: eval_at(p, n, T, c) for n in NS}
            row = dict(c=c, seed=s, d0_eff=tr[6][0][2], d1_rho=tr[6][1][0], br_rho=br,
                       **{f"t{n}_rho": tr[n][1][0] for n in NS},
                       **{f"t{n}_eff": tr[n][0][2] for n in NS})
            wrow("results/rerun_vanilla.csv", row, hv)
            print(f"[vanilla] c={c} s={s}: d0eff={tr[6][0][2]:.2f} br={br:.2f} "
                  f"transfer rho 6/12/24={tr[6][1][0]:.2f}/{tr[12][1][0]:.2f}/"
                  f"{tr[24][1][0]:.2f}", flush=True)

    # (B) league / PSRO CAN: robustness vs c
    hl = ["c", "seed", "d0_eff", "br_rho"]
    for c in CS:
        for s in SEEDS:
            p = league_train(N, T, c, generations=6, coop_iters=1200,
                             br_iters=1000, seed=s)
            br = best_response_audit(p, N, T, c, iters=1500, seed=800 + s)
            e0 = eval_at(p, N, T, c)[0][2]
            wrow("results/rerun_league.csv",
                 dict(c=c, seed=s, d0_eff=e0, br_rho=br), hl)
            print(f"[league] c={c} s={s}: d0eff={e0:.2f} br={br:.2f}", flush=True)

    # (C) reliability: single vs population co-training (c=0.3/0.5)
    hr = ["c", "seed", "method", "br_rho"]
    for c in [0.3, 0.5]:
        for s in SEEDS:
            ps = cotrain(N, T, c, iters=3000, seed=s)
            wrow("results/rerun_reliability.csv",
                 dict(c=c, seed=s, method="single",
                      br_rho=best_response_audit(ps, N, T, c, iters=1500, seed=900 + s)), hr)
            pp = cotrain_pop(N, T, c, K=4, iters=3000, seed=s)
            wrow("results/rerun_reliability.csv",
                 dict(c=c, seed=s, method="population",
                      br_rho=best_response_audit(pp, N, T, c, iters=1500, seed=950 + s)), hr)
            print(f"[reliability] c={c} s={s} done", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
