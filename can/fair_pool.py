"""Pool the independent gen6 league/vanilla runs into single estimates, so the
headline robustness numbers average over run-to-run (GPU-nondeterminism) variance,
not just seed variance. Writes results/{league,vanilla}_pooled.csv.

  league:  rerun_league + arch_ablation_gen6(XAttn) + d1eff(league)   = 15 runs/c
  vanilla: rerun_vanilla + d1eff(vanilla)                              = 10 runs/c

    python -m can.fair_pool
"""
import csv


def rows(p):
    return list(csv.DictReader(open(p)))


def write(path, sources):
    """sources: list of (rowlist, filterfn). Re-indexes seeds to 0..5k-1."""
    out, base = [], 0
    for rl, filt in sources:
        for x in rl:
            if filt(x):
                out.append(dict(c=x["c"], seed=base + int(x["seed"]),
                                d0_eff=x["d0_eff"], br_rho=x["br_rho"]))
        base += 5
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, ["c", "seed", "d0_eff", "br_rho"])
        w.writeheader(); w.writerows(out)
    print(f"wrote {path} ({len(out)} rows)")


def main():
    write("results/league_pooled.csv", [
        (rows("results/rerun_league.csv"), lambda x: True),
        (rows("results/arch_ablation_gen6.csv"), lambda x: x["arch"] == "XAttn"),
        (rows("results/d1eff.csv"), lambda x: x["regime"] == "league"),
    ])
    write("results/vanilla_pooled.csv", [
        (rows("results/rerun_vanilla.csv"), lambda x: True),
        (rows("results/d1eff.csv"), lambda x: x["regime"] == "vanilla"),
    ])


if __name__ == "__main__":
    main()
