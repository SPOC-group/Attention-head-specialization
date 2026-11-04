"""Generate Figure 4 from the results beside this script: python plot.py."""

from ast import literal_eval
from itertools import product
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


directory = Path(__file__).resolve().parent
LAMBDA = 0.2
NU1, NU2 = 5.0, 4.0
F, L, H, D = 2, 5, 4, 4000
NU_TOTAL = NU1 + NU2
DELTA_CUT = (NU1 - NU2) / NU_TOTAL

PS, PSB, SPEC, SPEC2 = '#D9D9D9', '#FC8D62', '#64A8C3', '#397993'
CUT, BO_COLOR = '#FD030F', '#FF0612'
ERRORBAR = dict(fmt='o', color='black', markersize=6, elinewidth=1.75, capsize=0, zorder=5)
plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 18,
    'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'lines.linewidth': 1.8, 'legend.fontsize': 14,
    'legend.facecolor': 'white', 'legend.edgecolor': '0.8', 'legend.framealpha': 0.7,
})

def read_csvs(files, empirical=False):
    frames = []
    for file in files:
        row = pd.read_csv(file).iloc[[-1]]
        if empirical:
            row['seed'] = int(file.stem.rsplit('_seed_', 1)[1])
        frames.append(row)
    return pd.concat(frames, ignore_index=True)

def matrices(frame, column, shape):
    return np.array([np.array(literal_eval(value)).reshape(shape) if isinstance(value, str)
                     else np.full(shape, np.nan) for value in frame[column]])

def shade(ax, edges, phase, colors):
    changes = np.flatnonzero(np.diff(phase)) + 1
    for start, stop in zip(np.r_[0, changes], np.r_[changes, len(phase)]):
        if phase[start] >= 0:
            ax.axvspan(edges[start], edges[stop], color=colors[phase[start]],
                       alpha=0.5, linewidth=0, antialiased=False, zorder=0)
    for i in changes:
        if phase[i - 1] >= 0 and phase[i] >= 0:
            ax.axvline(edges[i], color='black', linestyle='-.', linewidth=1.8, zorder=1)

def errors(ax, t, avg, spread):
    ax.plot(t.index, t.test_error, 'k-', label=r'$\mathcal{E}$')
    ax.plot(t.index, t.train_error, 'k--', label=r'$\mathcal{E}_{\mathrm{train}}$')
    ax.errorbar(avg.index, avg.test_error, yerr=spread.test_error, **ERRORBAR)
    ax.errorbar(avg.index, avg.train_error, yerr=spread.train_error, **dict(ERRORBAR, fmt='x'))
    ax.legend()

def align(m, reference):
    best_cost = np.inf
    for signs in product((-1, 1), repeat=2):
        candidate = m * signs
        cost = ((candidate[:, None] - reference[None, :])**2).sum(axis=2)
        rows, columns = linear_sum_assignment(cost)
        if cost[rows, columns].sum() < best_cost:
            best_cost = cost[rows, columns].sum()
            best = candidate[rows[np.argsort(columns)]]
    return best

def boundary(row):
    if row['status'] == 'crossing_found':
        return row['alpha_c'], row['alpha_c']
    if row['status'] == 'zero_signal':
        return np.inf, np.inf
    if row['status'] in ['stable_up_to_alpha_max', 'preempted_by_PS_MEM']:
        return np.nan, row['bracket_low']
    return np.nan, np.nan

def interpolate(x, grid, roots, limits):
    x = np.atleast_1d(x)
    result = np.array([np.interp(x, grid, column, left=np.nan, right=np.nan)
                       for column in roots.T]).T
    known = result.copy()
    for k, value in enumerate(x):
        j = np.searchsorted(grid, value)
        if j < len(grid) and np.isclose(grid[j], value, rtol=0, atol=1e-10):
            result[k], known[k] = roots[j], limits[j]
        elif 0 < j < len(grid):
            unknown = np.isnan(result[k])
            known[k, unknown] = np.minimum(limits[j - 1, unknown], limits[j, unknown])
    return result, known

def phase_at(alpha, roots, limits):
    phase = np.full(np.shape(alpha), -1, dtype=int)
    eligible = np.ones(np.shape(alpha), dtype=bool)
    for r, (critical, limit) in enumerate(zip(roots, limits)):
        phase[eligible & (alpha < limit)] = r
        eligible &= np.isfinite(critical) & (alpha >= critical)
    phase[eligible] = len(roots)
    return phase

def shade_cut(ax, roots, limits, xmax, colors):
    points = np.r_[roots, limits]
    edges = np.unique(np.r_[0, points[(points > 0) & (points < xmax)], xmax])
    shade(ax, edges, phase_at((edges[:-1] + edges[1:]) / 2, roots, limits), colors)

boundary_folder = directory / 'results_phase_boundary' / f'non_isotropic_gaussian_phase_boundary_lambda_{LAMBDA}_nutot_{NU_TOTAL}_F{F}_L{L}_H{H}'
records = []
for file in boundary_folder.glob('*.json'):
    report = json.loads(file.read_text())
    rows = {row['r_from']: row for row in report['boundaries']}
    pairs = np.array([boundary(rows[r]) for r in range(F)])
    records.append((report['arguments']['delta'], pairs[:, 0], pairs[:, 1]))
records.sort()
deltas = np.array([row[0] for row in records])
critical = np.array([row[1] for row in records])
stable_until = np.array([row[2] for row in records])

name = f'non_isotropic_gaussian_theory_lambda_{LAMBDA}_nu1_{NU1}_nu2_{NU2}_F{F}_L{L}_H{H}'
t = read_csvs(sorted((directory / 'results_theory' / f'{name}_FREE').glob('*_alpha_*.csv'))).set_index('alpha').sort_index()
m = matrices(t, 'm', (H, F))
reference = m[np.isfinite(m).all(axis=(1, 2))][-1]
emp_name = name.replace('_theory_', '_empirics_') + f'_D{D}_initI0'
e = read_csvs(sorted((directory / 'results_empirics' / emp_name).glob('*_alpha_*_seed_*.csv')), empirical=True)
emp_m = np.array([align(mi, reference) for mi in matrices(e, 'm', (H, F))])
for h in range(H):
    for f in range(F):
        t[f'm{h}{f}'] = m[:, h, f]
        e[f'm{h}{f}'] = emp_m[:, h, f]
groups = e.groupby('alpha')
avg, spread = groups.mean(numeric_only=True), groups.std(ddof=0, numeric_only=True)

xmax = max(t.index.max(), avg.index.max())
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained', gridspec_kw={'wspace': 0})
xedges, yedges = np.linspace(0, xmax, 601), np.linspace(deltas[0], deltas[-1], 601)
x, y = (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
roots, limits = interpolate(y, deltas, critical, stable_until)
z = np.array([phase_at(x, r, k) for r, k in zip(roots, limits)])
ax[0].pcolormesh(xedges, yedges, z, cmap=ListedColormap(['white', PS, SPEC, SPEC2]),
                 norm=BoundaryNorm(np.arange(-1.5, 3), 4), alpha=0.5,
                 edgecolors='none', antialiased=False, rasterized=True)
line_grid = np.unique(np.r_[y, deltas, (deltas[:-1] + deltas[1:]) / 2])
curves, _ = interpolate(line_grid, deltas, critical, stable_until)
for curve in curves.T:
    ax[0].plot(np.where(np.isfinite(curve), curve, np.nan), line_grid, color='black', linewidth=1.8, zorder=3)
ax[0].axhline(DELTA_CUT, color=CUT, linewidth=1.8, zorder=4)
ax[0].set(xlabel=r'$\alpha$', ylabel=r'$\delta$', xlim=(0, 6.0), ylim=(deltas[0], deltas[-1]))
roots, limits = interpolate([DELTA_CUT], deltas, critical, stable_until)
errors(ax[1], t, avg, spread)
for f in range(F):
    for h in range(H):
        column = f'm{h}{f}'
        ax[2].plot(t.index, t[column], 'k-', zorder=2)
        ax[2].errorbar(avg.index, avg[column], yerr=spread[column], **ERRORBAR)
ax[2].set(ylabel=r'$m_{hf}$')
for a in ax[1:]:
    shade_cut(a, roots[0], limits[0], xmax, [PS, SPEC, SPEC2])
    a.set(xlabel=r'$\alpha$', xlim=(0, 6.0))
    for spine in a.spines.values():
        spine.set(color=CUT, linewidth=1.8)
fig.savefig(directory / 'Figure4.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)
print(directory / 'Figure4.pdf')
