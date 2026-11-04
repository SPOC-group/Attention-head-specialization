from ast import literal_eval
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parent
lam, nu1, nu2 = 0.2, 5.0, 4.0
F, L, H, D = 2, 5, 4, 2000
colors = ['#D9D9D9', '#64A8C3', '#397993']
bo_color = '#FF0612'
errorbar = dict(fmt='o', color='black', markersize=6, elinewidth=1.75, capsize=0, zorder=5)
plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 18,
    'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'lines.linewidth': 1.8, 'legend.fontsize': 14,
    'legend.facecolor': 'white', 'legend.edgecolor': '0.8', 'legend.framealpha': 0.75,
})


def read_csvs(directory):
    return pd.concat([pd.read_csv(file).iloc[[-1]] for file in sorted(directory.glob('*_alpha_*.csv'))],
                     ignore_index=True)


name = f'non_isotropic_gaussian_theory_lambda_{lam}_nu1_{nu1}_nu2_{nu2}_F{F}_L{L}_H{H}'
theory = read_csvs(folder / 'results_theory' / f'{name}_FREE').set_index('alpha').sort_index()
emp_name = name.replace('_theory_', '_empirics_') + f'_D{D}_initI0'
empirics = read_csvs(folder / 'results_empirics' / emp_name)
bo_name = f'non_isotropic_gaussian_BO_nu1_{nu1}_nu2_{nu2}_F{F}_L{L}'
bo = read_csvs(folder / 'results_theory' / bo_name).set_index('alpha').sort_index()

for frame in [theory, empirics]:
    m = np.array([np.array(literal_eval(value)).reshape(H, F) for value in frame.m])
    q = np.array([np.array(literal_eval(value)).reshape(H, H) for value in frame.q])
    # Take the best head for each feature before averaging over seeds.
    q_diagonal = np.diagonal(q, axis1=1, axis2=2)
    for f in range(F):
        frame[f'rho{f + 1}'] = (m[:, :, f]**2 / q_diagonal).max(axis=1)
    if frame is theory:
        phase = (np.linalg.norm(m, axis=1) > 1e-2).sum(axis=1)
        phase[~np.isfinite(m).all(axis=(1, 2))] = -1

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
for a in ax:
    shade(a)
    a.set(xlabel=r'$\alpha$', xlim=(0, xmax))
ax[0].plot(theory.index, theory.test_error, 'k-', label=r'$\mathcal{E}$')
ax[0].plot(theory.index, theory.train_error, 'k--', label=r'$\mathcal{E}_{\mathrm{train}}$')
ax[0].errorbar(mean.index, mean.test_error, yerr=std.test_error, **errorbar)
ax[0].errorbar(mean.index, mean.train_error, yerr=std.train_error, **dict(errorbar, fmt='x'))
ax[0].plot(bo.index, bo.test_error, color=bo_color, linestyle='--', label=r'$\mathcal{E}_{\mathrm{BO}}$')
ax[0].legend(loc='lower right')
nus = np.array([nu1, nu2])
bo_alpha = np.unique(np.r_[np.linspace(0, xmax, 2001), bo.index, 1 / nus**2])
bo_rho = np.maximum(0, (bo_alpha[:, None] * nus**2 - 1) / (nus * (bo_alpha[:, None] * nus + 1)))
for f, column in enumerate(['rho1', 'rho2']):
    ax[f + 1].plot(theory.index, theory[column], 'k-',
                   label=rf'$\rho_{f+1}^{{\mathrm{{ERM}}}}$')
    ax[f + 1].plot(bo_alpha, bo_rho[:, f], color=bo_color, linestyle='--',
                   label=rf'$\rho_{f + 1}^{{\mathrm{{BO}}}}$')
    ax[f + 1].errorbar(mean.index, mean[column], yerr=std[column], **errorbar)
ax[1].legend(loc='lower right')
ax[2].legend(loc='upper right')

fig.savefig(folder / 'Figure6.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)
