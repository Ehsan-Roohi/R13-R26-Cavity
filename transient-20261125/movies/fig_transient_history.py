"""Figure for the time-dependent appendix: (a) rate of change of the solutions, max |dU/dt| over the interior for u/U_w, v/U_w and
T/T_w, versus time; (b) cold-to-hot area fraction (q . grad T > 0) over 0.1 <= x/L, y/L <= 0.9 on 4x4-cell blocks (the definition of
table tab:q_parts), versus time, sharp lid, time-dependent R13; horizontal lines: steady pooled DSMC values of that table (the 8-run start-up DSMC
is too noisy for this sign-based measure: its late-time values 0.66 and 0.35 differ from the pooled 0.85 and 0.09).
Writes figures/revision/fig_transient_history.pdf and TRANSIENT_HISTORY.json."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
H = Path(__file__).resolve().parent; P = H.parent
OUT = P / 'Manuscript' / 'FINAL_ALL_REFEREES_20261112' / 'figures' / 'revision'
TU = 1e-6 / (208.0 * 300.0) ** .5 * 1e9; DSTEP = 1.0323529147672158e-12 * 1e9; LID = 100 / (208.0 * 300.0) ** .5
c160 = (np.arange(160) + .5) / 160; c40 = (np.arange(40) + .5) / 40
Y4, X4 = np.meshgrid(c40, c40, indexing='ij'); M4 = (X4 >= .1) & (X4 <= .9) & (Y4 >= .1) & (Y4 <= .9)
res = {}

def rate(frames, times, idx=(1, 2, 3), scale=(LID, LID, 1.0)):
    F = np.asarray(frames, float); r = []
    for k in range(1, len(F)):
        d = max(float(np.abs(F[k][..., i] - F[k - 1][..., i]).max()) / s for i, s in zip(idx, scale))
        r.append(d / ((times[k] - times[k - 1]) * TU))
    return np.array(times[1:]) * TU, np.array(r)

def area(T, qx, qy):   # fields on 160x160 cell centres -> 4x4 blocks -> fraction with q.grad T > 0 inside the mask
    b = lambda a: a.reshape(40, 4, 40, 4).mean(axis=(1, 3))
    T, qx, qy = b(T), b(qx), b(qy)
    return float(np.mean(((qx * np.gradient(T, c40, axis=1) + qy * np.gradient(T, c40, axis=0)) > 0)[M4]))

def to160(a, x, y):
    Y, X = np.meshgrid(c160, c160, indexing='ij'); return RGI((y, x), a, bounds_error=False, fill_value=None)(np.column_stack([Y.ravel(), X.ravel()])).reshape(160, 160)

fig, ax = plt.subplots(1, 2, figsize=(5.3, 2.35), constrained_layout=True)
col = {'R13s20': '#1f77b4', 'R13s05': '#6baed6', 'R26g05': '#d62728', 'R13g05': '#2ca02c', 'R13g20': '#9467bd'}
lab = {'R13s20': r'R13, sharp, $Kn_{\rm Gu}=0.2$', 'R13s05': r'R13, sharp, $Kn_{\rm Gu}=0.05$', 'R26g05': r'R26, smooth, $Kn_{\rm Gu}=0.05$',
       'R13g05': r'R13, smooth, $Kn_{\rm Gu}=0.05$', 'R13g20': r'R13, smooth, $Kn_{\rm Gu}=0.2$'}
T13 = P / 'R13_TRANSIENT_20261124'
src = {'R13s20': T13 / 'R13_TRANSIENT_K20.npz', 'R13s05': T13 / 'R13_TRANSIENT_K05.npz', 'R13g05': T13 / 'R13_TRANSIENT_K05_GAUSS.npz', 'R13g20': T13 / 'R13_TRANSIENT_K20_GAUSS.npz'}
for k, p in src.items():
    z = np.load(p); t, r = rate(z['frames'], z['times']); ax[0].semilogy(t, r, color=col[k], lw=1.1, label=lab[k]); res[k] = dict(final_rate=float(r[-1]), min_rate=float(r.min()), t_min=float(t[r.argmin()]))
z = np.load(P / 'R26_TRANSIENT_20261125' / 'fetched' / 'R26_GAUSS_K05_MOVIE_FIELDS.npz')
t, r = rate(z['frames'][:, 1:-1, 1:-1], z['times']); ax[0].semilogy(t, r, color=col['R26g05'], lw=1.1, label=lab['R26g05']); res['R26g05'] = dict(final_rate=float(r[-1]))
ax[0].set_xlabel('$t$ (ns)'); ax[0].set_ylabel(r'largest rate of change (ns$^{-1}$)'); ax[0].set_xlim(0, 82); ax[0].set_ylim(1e-11, 3)
ax[0].legend(fontsize=6.5, frameon=False, loc='lower left')
# (b) cold-to-hot area, sharp lid
D = np.load(P / 'DSMC_TRANSIENT_MOVIE_20261124' / 'fetched' / 'MOVIE_SMALL.npz')
for kn, tag, c13 in ((0.2, 'KN020', 'K20'), (0.05, 'KN005', 'K05')):
    dF = D[tag].astype(float); dF[:, 3] += 1; st = D[tag + '_steps']; K = int(np.searchsorted(st, 160 * 500)) + 1
    td, ad = [], []
    for k in range(4, K):
        h = min(k // 4, 6); w = dF[max(k - h, 1):k + h + 1].mean(0)
        up = lambda a: np.repeat(np.repeat(a, 2, 0), 2, 1)       # 80x80 blocks of 2x2 cells -> 160 grid (block means unchanged)
        td.append(st[k] * DSTEP - 250 * DSTEP); ad.append(area(up(w[3]), up(w[4]), up(w[5])))
    z = dict(np.load(T13 / ('R13_TRANSIENT_%s.npz' % c13))); n = z['frames'].shape[1]; xn = np.arange(1, n + 1) / (n + 1)
    tr_, ar = [], []
    for k in range(1, len(z['times']), 2):
        U = z['frames'][k]; tr_.append(z['times'][k] * TU)
        ar.append(area(*(to160(U[..., i], xn, xn) for i in (3, 4, 5))))
    ls = '-' if kn == 0.2 else '--'
    ax[1].axhline({0.2: 0.85, 0.05: 0.09}[kn], color='0.25', lw=1.0, ls=ls, label=r'DSMC steady pool, $Kn_{\rm Gu}=%g$' % kn)   # table tab:q_parts
    ax[1].plot(tr_, ar, color=col['R13s20'], lw=1.1, ls=ls, label=r'R13, $Kn_{\rm Gu}=%g$' % kn)
    res['area_' + tag] = dict(DSMC_last=float(np.mean(ad[-20:])), R13_last=float(ar[-1]), DSMC_t5ns=float(np.interp(5, td, ad)), R13_t5ns=float(np.interp(5, tr_, ar)))
ax[1].set_xlabel('$t$ (ns)'); ax[1].set_ylabel('cold-to-hot area fraction'); ax[1].set_xlim(0, 82); ax[1].set_ylim(0, 1)
ax[1].legend(fontsize=6.5, frameon=False, loc='center right')
for a in ax: a.tick_params(labelsize=7); a.xaxis.label.set_size(8); a.yaxis.label.set_size(8)
fig.savefig(OUT / 'fig_transient_history.pdf'); fig.savefig(H / 'fig_transient_history_preview.png', dpi=200)
(H / 'TRANSIENT_HISTORY.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
