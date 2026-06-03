"""Report D>=1 efficiency (delivered/T WITH a best-response defector present) for
CAN league + vanilla on the base game --- the axis where the oracle gap lives, to
test whether high-c robustness is 'free' or 'bought with waste'. Matches the mains'
protocol (fresh best-response defector, gen6 league budget, 5 seeds).

    python -m can.fair_d1eff
"""
import os
import csv

from .fair_xattn import train, league_train, best_response_audit, eval_at

N, T = 6, 100
CS = [0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2, 3, 4]


def main():
    os.makedirs("results", exist_ok=True)
    out = "results/d1eff.csv"
    if os.path.exists(out):
        os.remove(out)
    header = ["regime", "c", "seed", "d0_eff", "d1_eff", "br_rho"]
    for s in SEEDS:
        for c in CS:
            # league
            p = league_train(N, T, c, generations=6, coop_iters=1200,
                             br_iters=1000, seed=s)
            d0 = eval_at(p, N, T, c)[0][2]
            rho, d1 = best_response_audit(p, N, T, c, iters=1500, seed=800 + s,
                                          return_eff=True)
            for regime, row in [("league", dict(regime="league", c=c, seed=s,
                                d0_eff=float(d0), d1_eff=float(d1), br_rho=float(rho)))]:
                new = not os.path.exists(out)
                with open(out, "a", newline="") as f:
                    w = csv.DictWriter(f, header)
                    if new:
                        w.writeheader()
                    w.writerow(row); f.flush()
            print(f"[league] c={c} s={s}: d0_eff={d0:.3f} d1_eff={d1:.3f} "
                  f"br_rho={rho:.2f}", flush=True)
            # vanilla
            _, _, pv = train(N=N, T=T, c=c, iters=2500, seed=s, dmax=2)
            d0v = eval_at(pv, N, T, c)[0][2]
            rhov, d1v = best_response_audit(pv, N, T, c, iters=1500, seed=700 + s,
                                            return_eff=True)
            with open(out, "a", newline="") as f:
                csv.DictWriter(f, header).writerow(
                    dict(regime="vanilla", c=c, seed=s, d0_eff=float(d0v),
                         d1_eff=float(d1v), br_rho=float(rhov)))
                f.flush()
            print(f"[vanilla] c={c} s={s}: d0_eff={d0v:.3f} d1_eff={d1v:.3f} "
                  f"br_rho={rhov:.2f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
