"""Run with python plot.py to save Figure7.pdf beside this file."""

from ast import literal_eval
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd

directory = Path(__file__).resolve().parent
lam, nu1, nu2, L, H, D = 0.2, 2.0, 2.0, 10, 2, 2000
diagram_max = 6.0
delta = nu2 / (nu1 + nu2)
PS, SPEC, CUT = '#D9D9D9', '#64A8C3', '#FD030F'
ERRORBAR = dict(color='black', markersize=6, elinewidth=1.75, capsize=0, zorder=5)
plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 18,
    'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'lines.linewidth': 1.8, 'legend.fontsize': 14,
    'legend.facecolor': 'white', 'legend.edgecolor': '0.8', 'legend.framealpha': 0.7,
})


def read_csvs(files):
    return pd.concat([pd.read_csv(file).iloc[[-1]] for file in files], ignore_index=True)


def matrices(frame, column, shape):
    return np.array([np.array(literal_eval(value)).reshape(shape) if isinstance(value, str)
                     else np.full(shape, np.nan) for value in frame[column]])


def interpolate(values):
    values = np.atleast_1d(values)
    roots = np.interp(values, delta_grid, critical, left=np.nan, right=np.nan)
    limits = roots.copy()
    for i, value in enumerate(values):
        j = np.searchsorted(delta_grid, value)
        if j < len(delta_grid) and np.isclose(delta_grid[j], value, rtol=0, atol=1e-10):
            roots[i], limits[i] = critical[j], known[j]
        elif 0 < j < len(delta_grid) and np.isnan(roots[i]):
            limits[i] = np.minimum(known[j - 1], known[j])
    return roots, limits


def phase_at(alpha, root, limit):
    phase = np.where(alpha < limit, 0, -1)
    if np.isfinite(root):
        phase[alpha >= root] = 1
    return phase


def shade_cut(ax, root, limit, xmax):
    points = np.array([root, limit])
    edges = np.unique(np.r_[0, points[(points > 0) & (points < xmax)], xmax])
    phase = phase_at((edges[:-1] + edges[1:]) / 2, root, limit)
    changes = np.flatnonzero(np.diff(phase)) + 1
    for start, stop in zip(np.r_[0, changes], np.r_[changes, len(phase)]):
        if phase[start] >= 0:
            ax.axvspan(edges[start], edges[stop], color=[PS, SPEC][phase[start]],
                       alpha=0.5, linewidth=0, antialiased=False, zorder=0)
    for i in changes:
        if phase[i - 1] >= 0 and phase[i] >= 0:
            ax.axvline(edges[i], color='black', linestyle='-.', linewidth=1.8, zorder=1)


# At each alpha, keep the physical solution with the largest free entropy.
name = f'flipping_spike_theory_lambda_{lam}_nu1_{nu1}_nu2_{nu2}_L{L}_H{H}'
files = [file for branch in ['PS', 'MEM', 'SPEC']
         for file in sorted((directory / 'results_theory' / f'{name}_{branch}').glob('*_alpha_*.csv'))]
t = read_csvs(files)
alpha = np.sort(t.alpha.unique())
m, q = matrices(t, 'm', (H, 2)), matrices(t, 'q', (H, H))
keep = np.isfinite(t[['free_entropy', 'train_error', 'test_error']]).all(axis=1).to_numpy()
keep = keep & np.isfinite(m).all(axis=(1, 2)) & np.isfinite(q).all(axis=(1, 2))
covariance = q[keep] - m[keep] @ m[keep].transpose(0, 2, 1)
keep[np.flatnonzero(keep)] &= np.linalg.eigvalsh(covariance).min(axis=1) >= -1e-8
t = t.loc[keep]
t = t.loc[t.groupby('alpha').free_entropy.idxmax()].set_index('alpha').reindex(alpha)

emp_name = name.replace('_theory_', '_empirics_') + f'_D{D}_initI0'
e = read_csvs(sorted((directory / 'results_empirics' / emp_name).glob('*_alpha_*_seed_*.csv')))
for frame in [t, e]:
    frame[['norm1', 'norm2']] = np.linalg.norm(matrices(frame, 'm', (H, 2)), axis=1)
groups = e.groupby('alpha')
avg, spread = groups.mean(numeric_only=True), groups.std(ddof=0, numeric_only=True)
xmax = max(t.index.max(), avg.index.max())

# Censored boundaries only establish the phase below their last stable alpha.
name = f'flipping_spike_phase_boundary_ratio_lambda_{lam}_nutot_{nu1 + nu2}_L{L}_H2'
records = []
for file in (directory / 'results_flipping_spike_phase_boundary_ratio' / name).rglob('*.json'):
    report = json.loads(file.read_text())
    row = report['boundaries'][0]
    root, limit = np.nan, np.nan
    if row['status'] == 'crossing_found':
        root = limit = row['alpha_c']
    elif row['status'] == 'zero_signal':
        root = limit = np.inf
    elif row['status'] in ['stable_up_to_alpha_max', 'preempted_by_PS_MEM']:
        limit = row['bracket_low']
    records.append((report['arguments']['delta'], root, limit))
delta_grid, critical, known = np.array(sorted(records)).T

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained', gridspec_kw={'wspace': 0})
xedges = np.linspace(0, diagram_max, 601)
yedges = np.linspace(delta_grid[0], delta_grid[-1], 601)
x, y = (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
roots, limits = interpolate(y)
z = np.array([phase_at(x, root, limit) for root, limit in zip(roots, limits)])
ax[0].pcolormesh(xedges, yedges, z, cmap=ListedColormap(['white', PS, SPEC]),
                 norm=BoundaryNorm(np.arange(-1.5, 2), 3), alpha=0.5,
                 edgecolors='none', antialiased=False, rasterized=True)
line_grid = np.unique(np.r_[y, delta_grid, (delta_grid[:-1] + delta_grid[1:]) / 2])
curve, _ = interpolate(line_grid)
ax[0].plot(np.where(np.isfinite(curve), curve, np.nan), line_grid, 'k-', zorder=3)
ax[0].axhline(delta, color=CUT, linewidth=1.8, zorder=4)
ax[0].set(xlabel=r'$\alpha$', ylabel=r'$\delta$', xlim=(0, diagram_max),
          ylim=(delta_grid[0], delta_grid[-1]))

ax[1].plot(t.index, t.test_error, 'k-', label=r'$\mathcal{E}$')
ax[1].plot(t.index, t.train_error, 'k--', label=r'$\mathcal{E}_{\mathrm{train}}$')
ax[1].errorbar(avg.index, avg.test_error, yerr=spread.test_error, fmt='o', **ERRORBAR)
ax[1].errorbar(avg.index, avg.train_error, yerr=spread.train_error, fmt='x', **ERRORBAR)
ax[1].legend()
for f, style, marker in [(1, '-', 'o'), (2, '--', 'x')]:
    column = f'norm{f}'
    ax[2].plot(t.index, t[column], color='black', linestyle=style, label=rf'$\Vert m_{{:{f}}}\Vert_2$')
    ax[2].errorbar(avg.index, avg[column], yerr=spread[column], fmt=marker, **ERRORBAR)
ax[2].set_ylim(bottom=-0.1)
ax[2].legend()
roots, limits = interpolate([delta])
for a in ax[1:]:
    shade_cut(a, roots[0], limits[0], xmax)
    a.set(xlabel=r'$\alpha$', xlim=(0, xmax))
    for spine in a.spines.values():
        spine.set(color=CUT, linewidth=1.8)

fig.savefig(directory / 'Figure7.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)
