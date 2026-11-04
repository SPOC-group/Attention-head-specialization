"""Generate Figure 3 from the results beside this script: python plot.py."""

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
LAMBDA = 0.065
NU1, NU2 = 0.5, 2.0
L, H, D = 10, 2, 6000
DIAGRAM_LAMBDA_MIN = 0.055

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

def overlap(a, b):
    return (a['status'] == b['status'] == 'crossing_found' and
            max(a['bracket_low'], b['bracket_low']) <= min(a['bracket_high'], b['bracket_high']))

# Figure3: PS -> PSB -> SPEC, or a direct PS -> SPEC transition.
lambda_grid, roots3, limits3 = [], [], []
for file in (directory / 'results_flipping_spike_phase_boundary' / f'flipping_spike_phase_boundary_nu1_{NU1}_nu2_{NU2}_L{L}_H{H}').rglob('*.json'):
    report = json.loads(file.read_text())
    rows = {row['transition']: row for row in report['boundaries']}
    qm, pm, mm = [rows[key] for key in ['PS_MEM', 'PS_SPEC', 'MEM_SPEC']]
    m, m_limit = boundary(qm)
    s, s_limit = boundary(pm)
    g, g_limit = boundary(mm)
    roots = [np.nan, np.nan]
    limits = [np.minimum(report['arguments']['alpha_max'], np.minimum(m_limit, s_limit)), np.nan]
    if np.isfinite(s) and (not np.isfinite(m) or pm['bracket_high'] < qm['bracket_low'] or overlap(pm, qm)):
        roots, limits = [s, s], [s, s]
    elif np.isfinite(m) and (pm['status'] in ['preempted_by_PS_MEM', 'zero_signal'] or
                             np.isfinite(s) and qm['bracket_high'] < pm['bracket_low']):
        roots, limits = [m, np.nan], [m, np.nan]
        if overlap(mm, qm):
            roots, limits = [g, g], [g, g]
        elif np.isfinite(g) and mm['bracket_low'] >= qm['bracket_high']:
            roots, limits = [m, g], [m, g]
        elif mm['status'] in ['stable_up_to_alpha_max', 'zero_signal']:
            limits[1] = min(report['arguments']['alpha_max'], g_limit)
    lambda_grid.append(report['arguments']['lam'])
    roots3.append(roots)
    limits3.append(limits)
order = np.argsort(lambda_grid)
lambda_grid, roots3, limits3 = np.array(lambda_grid)[order], np.array(roots3)[order], np.array(limits3)[order]
cut3, known3 = interpolate([LAMBDA], lambda_grid, roots3, limits3)
known3[:, 1] = np.interp([LAMBDA], lambda_grid, limits3[:, 1])

name = f'flipping_spike_theory_lambda_{LAMBDA}_nu1_{NU1}_nu2_{NU2}_L{L}_H{H}'
t = read_csvs(sorted((directory / 'results_theory' / f'{name}_SPEC').glob('*_alpha_*.csv')))
alpha = np.sort(t.alpha.unique())
m, q = matrices(t, 'm', (H, 2)), matrices(t, 'q', (H, H))
finite = np.isfinite(t[['free_entropy', 'train_error', 'test_error']]).all(axis=1)
finite = finite & np.isfinite(m).all(axis=(1, 2)) & np.isfinite(q).all(axis=(1, 2))
t = t.loc[finite].set_index('alpha').reindex(alpha)
emp_name = name.replace('_theory_', '_empirics_') + f'_D{D}_initI0'
e = read_csvs(sorted((directory / 'results_empirics' / emp_name).glob('*_alpha_*_seed_*.csv')), empirical=True)

# Order heads only in SPEC, using the same permutation on both axes of q.
for frame, alphas in [(t, t.index), (e, e.alpha.to_numpy())]:
    m, q = matrices(frame, 'm', (H, 2)), matrices(frame, 'q', (H, H))
    for i in np.flatnonzero(phase_at(alphas, cut3[0], known3[0]) == 2):
        order = np.argsort(m[i, :, 1])
        m[i], q[i] = m[i, order], q[i][np.ix_(order, order)]
    for h in range(H):
        frame[f'm{h}1'] = m[:, h, 1]
        for k in range(H):
            frame[f'q{h}{k}'] = q[:, h, k]
groups = e.groupby('alpha')
avg, spread = groups.mean(numeric_only=True), groups.std(ddof=0, numeric_only=True)

xmax = max(t.index.max(), avg.index.max())
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained', gridspec_kw={'wspace': 0})
points = np.r_[roots3.ravel(), limits3.ravel()]
xedges = np.unique(np.r_[0, points[(points > 0) & (points < xmax)], xmax])
yedges = np.r_[lambda_grid[0] - (lambda_grid[1] - lambda_grid[0]) / 2,
                (lambda_grid[:-1] + lambda_grid[1:]) / 2,
                lambda_grid[-1] + (lambda_grid[-1] - lambda_grid[-2]) / 2]
z = np.array([phase_at((xedges[:-1] + xedges[1:]) / 2, r, k) for r, k in zip(roots3, limits3)])
ax[0].pcolormesh(xedges, yedges, z, cmap=ListedColormap(['#eeeeee', PS, PSB, SPEC]),
                 norm=BoundaryNorm(np.arange(-1.5, 3), 4), alpha=0.5,
                 edgecolors='none', antialiased=False, rasterized=True)
for critical_alpha in roots3.T:
    ax[0].plot(critical_alpha, lambda_grid, color='black', linewidth=1.8)
tricritical = lambda_grid[limits3[:, 1] > roots3[:, 0]].max()
ax[0].axhline(tricritical, color='black', linestyle='-.', linewidth=1.8, label=r'$\lambda_{\mathrm{tric}}$')
ax[0].axhline(LAMBDA, color=CUT, linewidth=1.8)
ax[0].set(xlabel=r'$\alpha$', ylabel=r'$\lambda$', xlim=(0, xmax), ylim=(DIAGRAM_LAMBDA_MIN, lambda_grid[-1]))
ax[0].legend(frameon=False)
for h in range(2):
    for k in range(2):
        column = f'q{h}{k}'
        ax[1].plot(t.index, t[column], 'k-')
        ax[1].errorbar(avg.index, avg[column], yerr=spread[column], **ERRORBAR)
    column = f'm{h}1'
    ax[2].plot(t.index, t[column], 'k-')
    ax[2].errorbar(avg.index, avg[column], yerr=spread[column], **ERRORBAR)
ax[1].set(ylabel=r"$q_{hh'}$")
ax[2].set(ylabel=r'$m_{:2}$')
for a in ax[1:]:
    shade_cut(a, cut3[0], known3[0], xmax, [PS, PSB, SPEC])
    a.set(xlabel=r'$\alpha$', xlim=(0, xmax))
    for spine in a.spines.values():
        spine.set(color=CUT, linewidth=1.8)
fig.savefig(directory / 'Figure3.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)
print(directory / 'Figure3.pdf')
