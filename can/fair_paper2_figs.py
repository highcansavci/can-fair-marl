"""Paper-2 figures (method name: CAN = Cross Attention Networks; the centralized
upper bound is the 'centralized oracle'). Writes to paper/."""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("paper2", exist_ok=True)
N = 6


def _load(path):
    """Read a sweep CSV into {c: {col: [per-seed values]}}."""
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        c = float(r["c"])
        d = out.setdefault(c, {})
        for k, v in r.items():
            if k in ("c", "seed", "method"):
                continue
            d.setdefault(k, []).append(float(v))
    return out, rows


def _agg(d, c, col):
    v = np.array(d[c][col])
    return v.mean(), v.std()


def _boot(vals, n=4000, seed=0):
    """Mean and 95%% bootstrap CI as (mean, lower_err, upper_err) distances."""
    v = np.asarray(vals, float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    m = float(v.mean())
    return m, m - lo, hi - m


def _aggci(d, c, col):
    return _boot(d[c][col])


def _ebars(triples):
    """Unpack [(mean,lo,hi),...] into (means, [lo...], [hi...]) for errorbar yerr."""
    means = [m for m, _, _ in triples]
    return means, [[l for _, l, _ in triples], [h for _, _, h in triples]]


# ---- a calm, cohesive palette for the schematic ----
INK = "#2b2f38"          # primary text
MUTE = "#7b828d"         # captions / secondary
NEUTRAL = ("#f5f6f9", "#b7bec9", "#3a4049")   # fill, edge, text
ACCENT = ("#fff3df", "#df8a2c", "#7a4a12")    # warm: the CAN hero block
GREEN = ("#eaf5ed", "#46a06e", "#1f5e3c")     # cooperate
BLUE = ("#eaf1fb", "#5277bd", "#2b3f66")      # allocation / mechanism
RED = ("#fcecec", "#d05c5c", "#7a2b2b")       # defectors


def box(ax, x, y, w, h, t, pal=NEUTRAL, fs=8, lw=1.2, shadow=True):
    fc, ec, tc = pal
    bs = "round,pad=0.018,rounding_size=0.045"
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.006, y - 0.014), w, h, boxstyle=bs,
                                    fc="#11131a", ec="none", alpha=0.07, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=bs, fc=fc, ec=ec, lw=lw,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3)


def arrow(ax, a, b, c, d, color="#5b626d", rad=0.0, lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (a, b), (c, d), arrowstyle="-|>", mutation_scale=12, lw=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}",
        capstyle="round", joinstyle="round", zorder=4))


# ---------- Fig 1: CAN architecture + league training ----------
def fig_arch():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.patch.set_facecolor("white")

    # ---- (a) CAN policy + graded-contention game (the decentralized loop) ----
    ax1.set_title("(a)  CAN policy + graded-contention game", fontsize=10,
                  color=INK, pad=8, loc="left", x=0.04)
    MID = 0.55  # the flow band: every block is vertically centred here
    # agent tokens (input)
    for yy, lbl in [(0.73, "Agent 1:  $[u_1,\\,r_1,\\,$rank$_1]$"),
                    (0.53, "Agent $j$:  $[u_j,\\,r_j,\\,$rank$_j]$")]:
        box(ax1, 0.02, yy - 0.055, 0.30, 0.11, lbl, pal=NEUTRAL, fs=7.5)
    ax1.text(0.17, 0.355, "$\\vdots$", ha="center", fontsize=13, color=MUTE)
    ax1.text(0.17, 0.90, "$N$ agent tokens  (observed behaviour)", ha="center",
             fontsize=7.5, color=MUTE)
    # mechanism band: attention -> claim/yield -> graded allocation
    box(ax1, 0.375, MID - 0.115, 0.205, 0.23,
        "Cross-\nAttention\n$(QK^{\\top}\\!\\to V)$", pal=ACCENT, fs=8.5, lw=1.4)
    box(ax1, 0.61, MID - 0.075, 0.15, 0.15, "claim /\nyield", pal=GREEN, fs=8)
    box(ax1, 0.79, MID - 0.145, 0.165, 0.29,
        "graded\nallocation\n(split among\nclaimers,\nwaste $c$)", pal=BLUE, fs=7.3)
    # flow arrows
    arrow(ax1, 0.32, 0.71, 0.372, MID + 0.02, color="#aab0ba")
    arrow(ax1, 0.32, 0.53, 0.372, MID, color="#aab0ba")
    arrow(ax1, 0.582, MID, 0.608, MID)
    arrow(ax1, 0.762, MID, 0.788, MID)
    # utilities feed back to tokens (closing the loop)
    arrow(ax1, 0.872, MID - 0.145, 0.17, 0.30, color=BLUE[1], rad=-0.34,
          ls=(0, (5, 3)), lw=1.3)
    ax1.text(0.55, 0.205, "utilities $\\mathbf{u}$  (close the loop)", ha="center",
             fontsize=7.5, color=BLUE[1], style="italic")
    ax1.text(0.5, 0.04, "decentralized — agents act on observed state, "
             "no central allocator", ha="center", fontsize=7.5, style="italic",
             color=MUTE)

    # ---- (b) league (PSRO) training as a cycle ----
    ax2.set_title("(b)  League (PSRO) training", fontsize=10, color=INK, pad=8,
                  loc="left", x=0.06)
    box(ax2, 0.17, 0.64, 0.46, 0.20, "Cooperators\n(CAN, shared policy)",
        pal=GREEN, fs=9, lw=1.4)
    # growing pool: stacked frozen defectors (back sheets give depth)
    for i in range(3):
        last = i == 2
        box(ax2, 0.19 + 0.028 * i, 0.14 + 0.028 * i, 0.42, 0.18,
            "Pool of frozen\nbest-response defectors" if last else "",
            pal=RED, fs=8, lw=1.3 if last else 1.0, shadow=last)
    # cycle arrows
    arrow(ax2, 0.60, 0.64, 0.64, 0.33, color=RED[1], rad=-0.33, lw=1.5)
    ax2.text(0.84, 0.49, "(1) train a\nfresh exploiter", ha="center", fontsize=7.5,
             color=RED[1])
    arrow(ax2, 0.42, 0.33, 0.38, 0.64, color=GREEN[1], rad=-0.33, lw=1.5)
    ax2.text(0.10, 0.49, "(2) retrain CAN\nvs whole pool", ha="center", fontsize=7.5,
             color=GREEN[1])
    ax2.text(0.405, 0.485, "$\\circlearrowright$", ha="center", fontsize=18,
             color="#aab0ba")
    ax2.text(0.5, 0.03, "alternate $\\Rightarrow$ robust to the whole adversary "
             "history", ha="center", fontsize=7.5, style="italic", color=MUTE)

    for ax in (ax1, ax2):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.tight_layout(); fig.savefig("paper/fig_arch.png", dpi=200,
                                    facecolor="white"); plt.close(fig)


# ---------- Fig 2: reliability (best-response rho by training method) ----------
def fig_reliability():
    # single + population co-training (bounded-metric rerun)
    rel_rows = list(csv.DictReader(open("results/rerun_reliability.csv")))
    league, _ = _load("results/rerun_league.csv")
    data = {}
    for c in sorted(set(float(r["c"]) for r in rel_rows)):
        single = [float(r["br_rho"]) for r in rel_rows
                  if abs(float(r["c"]) - c) < 1e-9 and r["method"] == "single"]
        pop = [float(r["br_rho"]) for r in rel_rows
               if abs(float(r["c"]) - c) < 1e-9 and r["method"] == "population"]
        data[c] = {"naive co-train": single, "population": pop,
                   "CAN (league)": league[c]["br_rho"]}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, c in zip(axes, [0.3, 0.5]):
        meths = list(data[c]); xs = np.arange(len(meths))
        cols = ["tab:gray", "tab:orange", "tab:green"]
        for x, m, col in zip(xs, meths, cols):
            v = np.array(data[c][m])
            ax.scatter([x] * len(v), v, color=col, s=45, zorder=3)
            ax.plot([x - 0.18, x + 0.18], [v.mean()] * 2, color=col, lw=2.5)
        ax.axhline(1.0, ls="--", c="tab:blue", lw=1, label="fair / centralized oracle")
        ax.axhline(N, ls=":", c="tab:red", lw=1, label=f"yield (=N={N})")
        ax.set_xticks(xs); ax.set_xticklabels(meths, rotation=15, fontsize=8)
        ax.set_title(f"c = {c}", fontsize=9)
        if c == 0.3:
            ax.set_ylabel(r"best-response free-ride $\rho$")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Best-response free-ride $\\rho$ by training scheme "
                 "(per-seed points)", fontsize=9)
    fig.tight_layout(); fig.savefig("paper/fig_reliability.png", dpi=150); plt.close(fig)


# ---------- Figs 3-4: from the c-sweep CSV ----------
def fig_transfer_and_eff():
    van, _ = _load("results/rerun_vanilla.csv")
    CS = sorted(van)
    # transfer
    NSER = [6, 12, 24]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for c in CS:
        means, yerr = _ebars([_aggci(van, c, f"t{n}_rho") for n in NSER])
        ax.errorbar(NSER, means, yerr=yerr, marker="o", label=f"c={c}")
    ax.plot(NSER, NSER, ls=":", c="grey", label="yield (=N)")
    ax.axhline(1.0, ls="--", c="tab:blue", label="centralized oracle")
    ax.set_xlabel("eval team size $N$ (CAN trained at $N=6$)")
    ax.set_ylabel(r"free-ride $\rho$"); ax.set_yscale("log")
    ax.set_title(r"Best-response free-ride $\rho$ vs. evaluation team size $N$")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("paper/fig_transfer.png", dpi=150); plt.close(fig)
    # efficiency vs c
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    means, yerr = _ebars([_aggci(van, c, "d0_eff") for c in CS])
    ax.errorbar(CS, means, yerr=yerr, marker="o", color="tab:green",
                label="CAN (ours)")
    ax.plot(CS, [1 - c for c in CS], marker="x", ls="--", c="tab:red",
            label="fixed all-contest ($1-c$)")
    ax.axhline(1.0, ls="--", c="tab:blue", label="centralized oracle")
    ax.set_xlabel("contention waste $c$")
    ax.set_ylabel("efficiency when no free-rider (D=0)")
    ax.set_title(r"Efficiency with no free-rider ($D=0$) vs. contention $c$")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("paper/fig_efficiency.png", dpi=150); plt.close(fig)


# ---------- Fig 5: robustness vs contention (vanilla vs league) ----------
def fig_robustness():
    van, _ = _load("results/vanilla_pooled.csv")
    lea, _ = _load("results/league_pooled.csv")
    CS = sorted(set(van) & set(lea))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for d, col, lbl, mk in [(van, "tab:gray", "vanilla co-train", "o"),
                            (lea, "tab:green", "CAN (league)", "s")]:
        means, yerr = _ebars([_aggci(d, c, "br_rho") for c in CS])
        ax.errorbar(CS, means, yerr=yerr, marker=mk, color=col, label=lbl, capsize=3)
    ax.axhline(1.0, ls="--", c="tab:blue", label="fair / centralized oracle")
    ax.set_xlabel("contention waste $c$")
    ax.set_ylabel(r"best-response free-ride $\rho$ (lower = robust)")
    ax.set_title(r"Best-response free-ride $\rho$ vs. contention $c$")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("paper/fig_robustness.png", dpi=150); plt.close(fig)


# ---------- Fig 6: efficiency-vs-exploitability tradeoff (head-to-head) ----------
def fig_tradeoff():
    brows = list(csv.DictReader(open("results/baselines_graded.csv")))
    lrows = list(csv.DictReader(open("results/league_pooled.csv")))
    series = {
        "GGF": ([float(r["d0_eff"]) for r in brows if r["method"] == "GGF"],
                [float(r["br_rho"]) for r in brows if r["method"] == "GGF"],
                "tab:gray", "o"),
        "FEN": ([float(r["d0_eff"]) for r in brows if r["method"] == "FEN"],
                [float(r["br_rho"]) for r in brows if r["method"] == "FEN"],
                "#df8a2c", "^"),
        "SOTO": ([float(r["d0_eff"]) for r in brows if r["method"] == "SOTO"],
                 [float(r["br_rho"]) for r in brows if r["method"] == "SOTO"],
                 "#5277bd", "s"),
        "CAN (league, ours)": ([float(r["d0_eff"]) for r in lrows],
                               [float(r["br_rho"]) for r in lrows],
                               "#2f9e44", "*"),
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    # good region: efficient (eff->1) AND robust (rho->1) = bottom-right
    ax.axhspan(0.8, 1.6, xmin=(0.9 - 0.0) / 1.1, xmax=1.0, color="#eaf5ed", zorder=0)
    ax.axhline(1.0, ls="--", c="#888", lw=1)
    ax.axvline(1.0, ls="--", c="#888", lw=1)
    for lbl, (xs, ys, col, mk) in series.items():
        ax.scatter(xs, ys, c=col, marker=mk, s=120 if mk == "*" else 55,
                   edgecolors="white", linewidths=0.6, label=lbl, zorder=4,
                   alpha=0.95)
    ax.scatter([1.0], [1.0], marker="*", s=260, c="#f0b429",
               edgecolors="#7a5800", linewidths=0.8, zorder=5,
               label="centralized oracle")
    ax.set_xlim(0.0, 1.1); ax.set_ylim(0.7, 6.4)
    ax.set_xlabel("efficiency when no free-rider present  (D=0; $\\to$1 = no waste)")
    ax.set_ylabel(r"best-response free-ride $\rho$  ($\to$1 = robust)")
    ax.set_title(r"Efficiency ($D=0$) vs. best-response free-ride $\rho$")
    ax.text(0.30, 6.05, "FEN/GGF: efficient but\nexploitable (yield)",
            fontsize=7.5, color="#555", ha="center")
    ax.text(0.27, 1.45, "SOTO: robust but\nwasteful (all-contest)", fontsize=7.5,
            color="#555", ha="center")
    ax.text(0.985, 2.15, "robust +\nefficient", fontsize=7.5, color="#2f7d4a",
            ha="right", style="italic")
    ax.legend(fontsize=7.5, loc="center right", framealpha=0.95)
    fig.tight_layout(); fig.savefig("paper/fig_tradeoff.png", dpi=160)
    plt.close(fig)


# ---------- Fig 6b: per-environment tradeoff (3-env generality) ----------
def fig_tradeoff_envs():
    ENVS = [("base", "results/baselines_graded.csv", "results/league_pooled.csv",
             "(a) base: single resource"),
            ("congestion", "results/baselines_congestion.csv",
             "results/congestion_can.csv", "(b) congestion: $M$ servers"),
            ("stakes", "results/baselines_stakes.csv", "results/stakes_can.csv",
             "(c) stakes: random value")]
    spec = [("GGF", "tab:gray", "o"), ("FEN", "#df8a2c", "^"),
            ("SOTO", "#5277bd", "s")]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharey=True)
    for ax, (name, bpath, cpath, title) in zip(axes, ENVS):
        brows = list(csv.DictReader(open(bpath)))
        crows = list(csv.DictReader(open(cpath)))
        ax.axhspan(0.8, 1.6, xmin=0.9 / 1.1, xmax=1.0, color="#eaf5ed", zorder=0)
        ax.axhline(1.0, ls="--", c="#888", lw=0.8)
        for meth, col, mk in spec:
            xs = [float(r["d0_eff"]) for r in brows if r["method"] == meth]
            ys = [float(r["br_rho"]) for r in brows if r["method"] == meth]
            ax.scatter(xs, ys, c=col, marker=mk, s=42, edgecolors="white",
                       linewidths=0.5, label=meth, zorder=4, alpha=0.9)
        xs = [float(r["d0_eff"]) for r in crows]
        ys = [float(r["br_rho"]) for r in crows]
        ax.scatter(xs, ys, c="#2f9e44", marker="*", s=150, edgecolors="white",
                   linewidths=0.6, label="CAN (ours)", zorder=5)
        ax.scatter([1.0], [1.0], marker="*", s=230, c="#f0b429",
                   edgecolors="#7a5800", linewidths=0.8, zorder=6,
                   label="oracle")
        ax.set_xlim(0.0, 1.1); ax.set_ylim(0.7, 6.4)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("$D{=}0$ efficiency ($\\to$1 = no waste)", fontsize=8)
    axes[0].set_ylabel(r"best-response $\rho$ ($\to$1 = robust)")
    axes[0].legend(fontsize=7.5, loc="center left", framealpha=0.95)
    fig.suptitle(r"$D{=}0$ efficiency vs. best-response $\rho$ across three "
                 "environments (per seed$\\times c$)", fontsize=9.5)
    fig.tight_layout(); fig.savefig("paper/fig_tradeoff_envs.png", dpi=160)
    plt.close(fig)


# ---------- Fig 7: architecture ablation (rho vs c by aggregator) ----------
def fig_ablation():
    g6 = list(csv.DictReader(open("results/arch_ablation_gen6.csv")))
    g5 = list(csv.DictReader(open("results/arch_ablation_gen5.csv")))
    CS = [0.3, 0.5, 0.7, 0.9]

    def series(rows, arch):
        out = []
        for c in CS:
            v = [float(r["br_rho"]) for r in rows
                 if r["arch"] == arch and abs(float(r["c"]) - c) < 1e-9]
            out.append(_boot(v))
        return out

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    spec = [(g6, "XAttn", "#2f9e44", "*", "cross-attention (CAN)", 2.0, 1.0),
            (g6, "GRU", "#5277bd", "s", "bi-GRU", 1.8, 0.95),
            (g5, "DeepSets", "#df8a2c", "^", r"deep-sets$^\dagger$", 1.0, 0.55),
            (g5, "MeanPool", "#9aa1ab", "o", r"mean-pool$^\dagger$", 1.0, 0.55)]
    for rows, arch, col, mk, lbl, lw, al in spec:
        means, yerr = _ebars(series(rows, arch))
        ax.errorbar(CS, means, yerr=yerr, marker=mk, color=col, label=lbl,
                    lw=lw, alpha=al, capsize=2,
                    ms=11 if mk == "*" else 6)
    ax.axhline(1.0, ls="--", c="#888", lw=1, label="fair / oracle")
    ax.set_xlabel("contention waste $c$")
    ax.set_ylabel(r"best-response free-ride $\rho$ (lower = robust)")
    ax.set_title(r"Best-response $\rho$ vs. $c$ by policy architecture")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig("paper/fig_ablation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_arch(); fig_reliability(); fig_transfer_and_eff(); fig_robustness()
    fig_tradeoff(); fig_tradeoff_envs(); fig_ablation()
    print("wrote paper/fig_{arch,reliability,transfer,efficiency,robustness,"
          "tradeoff,tradeoff_envs,ablation}.png")
