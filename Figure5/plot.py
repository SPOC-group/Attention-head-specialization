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

PS, PSB, SPEC, BO_COLOR = "#D9D9D9", "#FC8D62", "#64A8C3", "#FF0612"
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
    m, q = matrices(frame, "m", (H, 2)), matrices(frame, "q", (H, H))
    # Take the best head for each feature before averaging over seeds.
    q_diagonal = np.diagonal(q, axis1=1, axis2=2)
    for f in range(2):
        frame[f"rho{f + 1}"] = (m[:, :, f]**2 / q_diagonal).max(axis=1)
avg = e.groupby("alpha").mean(numeric_only=True)
spread = e.groupby("alpha").std(ddof=0, numeric_only=True)

name = f"flipping_spike_BO_nu1_{nu1}_nu2_{nu2}_L{L}"
bo = read_csvs(sorted((directory / "results_theory" / name).glob("*_alpha_*.csv")))
bo = bo.groupby("alpha")[["test_error", "rho1", "rho2"]].mean()

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained", gridspec_kw={"wspace": 0})
for a in ax:
    shade(a)
    a.set(xlabel=r"$\alpha$", xlim=(0, max(t.index.max(), avg.index.max(), bo.index.max())))
ax[0].plot(t.index, t.test_error, "k-", label=r"$\mathcal{E}$")
ax[0].plot(t.index, t.train_error, "k--", label=r"$\mathcal{E}_{\mathrm{train}}$")
ax[0].errorbar(avg.index, avg.test_error, yerr=spread.test_error, **ERRORBAR)
ax[0].errorbar(avg.index, avg.train_error, yerr=spread.train_error, **dict(ERRORBAR, fmt="x"))
ax[0].plot(bo.index, bo.test_error, color=BO_COLOR, linestyle="--", label=r"$\mathcal{E}_{\mathrm{BO}}$")
ax[0].legend()
for f in [1, 2]:
    column = f"rho{f}"
    ax[f].plot(t.index, t[column], "k-", label=rf"$\rho^{{\mathrm{{ERM}}}}_{f}$")
    ax[f].plot(bo.index, bo[column], color=BO_COLOR, linestyle="--", label=rf"$\rho_{f}^{{\mathrm{{BO}}}}$")
    ax[f].errorbar(avg.index, avg[column], yerr=spread[column], **ERRORBAR)
    ax[f].legend()
fig.savefig(directory / "Figure5.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
