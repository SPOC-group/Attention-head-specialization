from ast import literal_eval
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

folder = Path(__file__).resolve().parent
lam, nu1, nu2 = 0.2, 4.0, 4.0
F, L, H, D = 2, 5, 4, 2000
colors = ['#D9D9D9', '#64A8C3', '#397993']
errorbar = dict(fmt='o', color='black', markersize=6, elinewidth=1.75, capsize=0, zorder=5)
plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 18,
    'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'lines.linewidth': 1.8, 'legend.fontsize': 14,
    'legend.facecolor': 'white', 'legend.edgecolor': '0.8', 'legend.framealpha': 0.7,
})


def read_csvs(directory):
    return pd.concat([pd.read_csv(file).iloc[[-1]] for file in sorted(directory.glob('*_alpha_*.csv'))],
                     ignore_index=True)


def amplitudes(m):
    # Count independent centered directions to identify the phase.
    basis, values = [], np.zeros(F)
    for f in range(F):
        v = m[:, f] - m[:, f].mean()
        for _ in range(2):
            for u in basis:
                v -= u * (u @ v)
        values[f] = np.linalg.norm(v)
        if values[f] > 1e-2:
            basis.append(v / values[f])
    return values


def align(m, reference):
    # Match feature signs and head order before averaging over seeds.
    best_cost = np.inf
    for signs in product((-1, 1), repeat=F):
        candidate = m * signs
        cost = ((candidate[:, None] - reference[None, :])**2).sum(axis=2)
        rows, columns = linear_sum_assignment(cost)
        if cost[rows, columns].sum() < best_cost:
            best_cost = cost[rows, columns].sum()
            best = candidate[rows[np.argsort(columns)]]
    return best


name = f'non_isotropic_gaussian_theory_lambda_{lam}_nu1_{nu1}_nu2_{nu2}_F{F}_L{L}_H{H}'
theory = read_csvs(folder / 'results_theory' / f'{name}_FREE').set_index('alpha').sort_index()
emp_name = name.replace('_theory_', '_empirics_') + f'_D{D}_initI0'
empirics = read_csvs(folder / 'results_empirics' / emp_name)
m = np.array([np.array(literal_eval(value)).reshape(H, F) for value in theory.m])
phase = (np.array([amplitudes(mi) for mi in m]) > 1e-2).sum(axis=1)
phase[~np.isfinite(m).all(axis=(1, 2))] = -1
reference = m[np.flatnonzero(phase == phase.max())[-1]]

for frame in [theory, empirics]:
    m = np.array([np.array(literal_eval(value)).reshape(H, F) for value in frame.m])
    if frame is empirics:
        m = np.array([align(mi, reference) for mi in m])
    for h in range(H):
        for f in range(F):
            frame[f'm{h}{f}'] = m[:, h, f]

groups = empirics.groupby('alpha')
mean, std = groups.mean(numeric_only=True), groups.std(ddof=0, numeric_only=True)
alpha = theory.index.to_numpy()
edges = np.r_[0, (alpha[:-1] + alpha[1:]) / 2, alpha[-1]]
xmax = max(theory.index.max(), mean.index.max())


def shade(ax):
    changes = np.flatnonzero(np.diff(phase)) + 1
    for start, stop in zip(np.r_[0, changes], np.r_[changes, len(phase)]):
        if phase[start] >= 0:
            ax.axvspan(edges[start], edges[stop], color=colors[phase[start]],
                       alpha=0.5, linewidth=0, antialiased=False, zorder=0)
    for i in changes:
        if phase[i - 1] >= 0 and phase[i] >= 0:
            ax.axvline(edges[i], color='black', linestyle='-.', linewidth=1.8, zorder=1)


fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained', gridspec_kw={'wspace': 0})
for a in ax[1:]:
    shade(a)
    a.set(xlabel=r'$\alpha$', xlim=(0, 6.0))
ax[0].add_patch(Ellipse((0, 0), width=0.5 * np.sqrt(nu2), height=0.5 * np.sqrt(nu1),
                        facecolor='none', edgecolor='0.7', linewidth=3, zorder=1))
selected = np.unique(np.r_[np.arange(0, len(mean), 2), len(mean) - 1])
cells = np.clip(np.searchsorted(edges, mean.index[selected], side='right') - 1, 0, len(phase) - 1)
point_colors = [colors[r] if r >= 0 else '0.7' for r in phase[cells]]
for h in range(H):
    x, y = f'm{h}1', f'm{h}0'
    ax[0].plot(theory[x], theory[y], 'k-', zorder=2)
    ax[0].errorbar(mean[x].iloc[selected], mean[y].iloc[selected],
                   xerr=std[x].iloc[selected], yerr=std[y].iloc[selected],
                   fmt='none', ecolor='0.4', elinewidth=1.75, capsize=0, zorder=3)
    ax[0].scatter(mean[x].iloc[selected], mean[y].iloc[selected], c=point_colors,
                  s=85, edgecolors='0.25', linewidths=0.7, zorder=5)
ax[0].scatter([0], [0], s=10, color='0.6', zorder=2)
ax[0].set(xlabel=r'$m_{:2}$', ylabel=r'$m_{:1}$')
ax[0].set_aspect('equal', adjustable='datalim')
ax[1].plot(theory.index, theory.test_error, 'k-', label=r'$\mathcal{E}$')
ax[1].plot(theory.index, theory.train_error, 'k--', label=r'$\mathcal{E}_{\mathrm{train}}$')
ax[1].errorbar(mean.index, mean.test_error, yerr=std.test_error, **errorbar)
ax[1].errorbar(mean.index, mean.train_error, yerr=std.train_error, **dict(errorbar, fmt='x'))
ax[1].legend()
for f in range(F):
    for h in range(H):
        column = f'm{h}{f}'
        ax[2].plot(theory.index, theory[column], 'k-', zorder=2)
        ax[2].errorbar(mean.index, mean[column], yerr=std[column], **errorbar)
ax[2].set(ylabel=r'$m_{hf}$')
fig.savefig(folder / 'Figure9.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)
