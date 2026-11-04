from ast import literal_eval
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

L, H, D = 10, 2, 2000
lambda_, nu1, nu2 = 0.2, 0.1, 2.0
directory = Path(__file__).resolve().parent

PS, PSB, SPEC = "#D9D9D9", "#FC8D62", "#64A8C3"
ERRORBAR = dict(fmt="o", color="black", markersize=6, elinewidth=1.75, capsize=0, zorder=5)
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 18,
    "axes.titlesize": 18, "axes.labelsize": 18, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "lines.linewidth": 1.8, "legend.fontsize": 14,
    "legend.facecolor": "white", "legend.edgecolor": "0.8", "legend.framealpha": 0.7,
})


def read_csvs(files, empirical=False):
    rows = []
    for file in files:
        row = pd.read_csv(file).iloc[[-1]]
        if empirical:
            row["seed"] = int(file.stem.rsplit("_seed_", 1)[1])
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def matrices(frame, column, shape):
    return np.array([np.array(literal_eval(value)).reshape(shape) if isinstance(value, str)
                     else np.full(shape, np.nan) for value in frame[column]])


def shade(ax):
    changes = np.flatnonzero(np.diff(phase)) + 1
    for start, stop in zip(np.r_[0, changes], np.r_[changes, len(phase)]):
        if phase[start] >= 0:
            ax.axvspan(edges[start], edges[stop], color=[PS, PSB, SPEC][phase[start]],
                       alpha=0.5, linewidth=0, antialiased=False, zorder=0)
    for i in changes:
        if phase[i - 1] >= 0 and phase[i] >= 0:
            ax.axvline(edges[i], color="black", linestyle="-.", linewidth=1.8, zorder=1)


name = f"flipping_spike_theory_lambda_{lambda_}_nu1_{nu1}_nu2_{nu2}_L{L}_H{H}"
files = [file for branch in ["PS", "MEM", "SPEC"]
         for file in sorted((directory / "results_theory" / f"{name}_{branch}").glob("*_alpha_*.csv"))]
t = read_csvs(files)
alpha = np.sort(t.alpha.unique())
m, q = matrices(t, "m", (H, 2)), matrices(t, "q", (H, H))
keep = np.isfinite(t[["free_entropy", "train_error", "test_error"]]).all(axis=1).to_numpy()
keep = keep & np.isfinite(m).all(axis=(1, 2)) & np.isfinite(q).all(axis=(1, 2))
covariance = q[keep] - m[keep] @ m[keep].transpose(0, 2, 1)
keep[np.flatnonzero(keep)] &= np.linalg.eigvalsh(covariance).min(axis=1) >= -1e-8
t = t.loc[keep]
t = t.loc[t.groupby("alpha").free_entropy.idxmax()].set_index("alpha").reindex(alpha)
m, q = matrices(t, "m", (H, 2)), matrices(t, "q", (H, H))
m2 = abs(m[:, 0, 1] - m[:, 1, 1]) / 2
q2 = (q[:, 0, 0] + q[:, 1, 1] - q[:, 0, 1] - q[:, 1, 0]) / 4
phase = np.where(m2 > 1e-3, 2, np.where(q2 > 1e-3, 1, 0))
phase[~np.isfinite(m).all(axis=(1, 2))] = -1
edges = np.r_[0, (alpha[:-1] + alpha[1:]) / 2, alpha[-1]]

name = name.replace("_theory_", "_empirics_") + f"_D{D}_initI0"
e = read_csvs(sorted((directory / "results_empirics" / name).glob("*_alpha_*_seed_*.csv")), empirical=True)
for frame in [t, e]:
    m = matrices(frame, "m", (H, 2))
    frame[["norm1", "norm2"]] = np.linalg.norm(m, axis=1)
    if frame is t:
        ordered = np.ones(len(frame), dtype=bool)
    else:
        cells = np.clip(np.searchsorted(edges, frame.alpha, side="right") - 1, 0, len(alpha) - 1)
        ordered = phase[cells] == 2
    for i in np.flatnonzero(ordered):
        m[i] = m[i, np.argsort(m[i, :, 1])]
    for h in range(H):
        for f in range(2):
            frame[f"m{h}{f}"] = m[:, h, f]
avg = e.groupby("alpha").mean(numeric_only=True)
spread = e.groupby("alpha").std(ddof=0, numeric_only=True)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained", gridspec_kw={"wspace": 0})
critical = edges[np.flatnonzero(phase == 2)[0]]
sx, sy = 0.55 * np.sqrt([nu2, nu1])
for sign in [-1, 1]:
    ax[0].annotate("", xy=(sign * sx, sy), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="-|>", color="0.6", linewidth=1.8, mutation_scale=13, alpha=0.8), zorder=1)
selected = np.unique(np.r_[np.arange(0, len(avg), 2), len(avg) - 1])
colors = np.where(avg.index[selected] < critical, PS, SPEC)
for h in range(H):
    x, y = f"m{h}1", f"m{h}0"
    ax[0].plot(t[x], t[y], "k-", zorder=2)
    ax[0].errorbar(avg[x].iloc[selected], avg[y].iloc[selected],
                   xerr=spread[x].iloc[selected], yerr=spread[y].iloc[selected],
                   fmt="none", ecolor="0.4", elinewidth=1.75, capsize=0, zorder=3)
    ax[0].scatter(avg[x].iloc[selected], avg[y].iloc[selected], c=colors,
                  s=85, edgecolors="0.25", linewidths=0.7, zorder=5)
ax[0].scatter([0], [0], s=10, color="0.6", zorder=2)
xcols, ycols = ["m01", "m11"], ["m00", "m10"]
xmax = max(sx, np.nanmax(abs(t[xcols])), np.nanmax(abs(avg[xcols]) + spread[xcols]))
ymin = min(0, np.nanmin(t[ycols]), np.nanmin(avg[ycols] - spread[ycols]))
ymax = max(sy, np.nanmax(t[ycols]), np.nanmax(avg[ycols] + spread[ycols]))
ypad = 0.04 * (ymax - ymin)
ax[0].set(xlabel=r"$m_{:2}$", ylabel=r"$m_{:1}$",
          xlim=(-1.04 * xmax, 1.04 * xmax), ylim=(ymin - ypad, ymax + ypad))
for a in ax[1:]:
    shade(a)
    a.set(xlabel=r"$\alpha$", xlim=(0, max(t.index.max(), avg.index.max())))
ax[1].plot(t.index, t.test_error, "k-", label=r"$\mathcal{E}$")
ax[1].plot(t.index, t.train_error, "k--", label=r"$\mathcal{E}_{\mathrm{train}}$")
ax[1].errorbar(avg.index, avg.test_error, yerr=spread.test_error, **ERRORBAR)
ax[1].errorbar(avg.index, avg.train_error, yerr=spread.train_error, **dict(ERRORBAR, fmt="x"))
ax[1].legend()
for f, linestyle, marker in [(1, "-", "o"), (2, "--", "x")]:
    column = f"norm{f}"
    ax[2].plot(t.index, t[column], color="black", linestyle=linestyle, label=rf"$\Vert m_{{:{f}}}\Vert_2$")
    ax[2].errorbar(avg.index, avg[column], yerr=spread[column], **dict(ERRORBAR, fmt=marker))
ax[2].legend()
fig.savefig(directory / "Figure2.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
