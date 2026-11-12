"""Render the 2026-11-11 grid comparison figures from latest_centerlines.csv and R26_KN020_qx_difference_maps.npz.
Usage: python render_latest.py [directory]   (default: the directory of this script)
Requires numpy and matplotlib. Writes PDF (vector) and PNG versions of each figure next to the data.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
SYMBOL = {'qx': r'$q_x/(\rho_0 c_0^3)$', 'qy': r'$q_y/(\rho_0 c_0^3)$', 'T': r'$T/T_w$', 'ux': r'$u_x/U_w$', 'uy': r'$u_y/U_w$'}
AXIS = {'horizontal': r'$x/L$  ($y/L=0.5$)', 'vertical': r'$y/L$  ($x/L=0.5$)'}
STYLE = {
    ('0.20', 'SPARTA matched historical'): ('DSMC (SPARTA, matched ensemble)', dict(color='#1A1A1A', ls='-', lw=1.0)),
    ('0.20', 'N90'): ('R26 N90 (paper)', dict(color='#7A5195', ls=':', lw=1.5)),
    ('0.20', 'N120 step 36'): (r'R26 N120, step 36 (RMS $1.07\times10^{-3}$)', dict(color='#C12273', ls='--', lw=1.4)),
    ('0.20', 'N180 step 8'): (r'R26 N180, step 8 (RMS $4.33\times10^{-3}$)', dict(color='#007F75', ls='-', lw=1.3)),
    ('0.05', 'SPARTA seven-lineage pool'): ('DSMC (SPARTA, 7 lineages pooled)', dict(color='#1A1A1A', ls='-', lw=1.0)),
    ('0.05', 'N100'): ('R13 N100 (practical tolerance)', dict(color='#7A5195', ls=':', lw=1.5)),
    ('0.05', 'N120'): ('R13 N120 (practical tolerance)', dict(color='#C12273', ls='--', lw=1.4)),
    ('0.05', 'N180'): (r'R13 N180 accepted4 (RMS $2.0\times10^{-5}$)', dict(color='#D06A00', ls='-', lw=1.3)),
}
ORDER = {'0.20': ['SPARTA matched historical', 'N90', 'N120 step 36', 'N180 step 8'],
         '0.05': ['SPARTA seven-lineage pool', 'N100', 'N120', 'N180']}

data = defaultdict(list)
with (HERE / 'latest_centerlines.csv').open(newline='', encoding='utf-8') as handle:
    for row in csv.DictReader(handle):
        data[row['Kn'], row['series'], row['component'], row['cut']].append((float(row['position_over_L']), float(row['value'])))
data = {key: np.array(values).T for key, values in data.items()}

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def sci(ax):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)


def panel(ax, kn, component, cut, limits=(0, 1)):
    for series in ORDER[kn]:
        s, v = data[kn, series, component, cut]
        keep = (s >= limits[0]) & (s <= limits[1])
        label, style = STYLE[kn, series]
        ax.plot(s[keep], v[keep], label=label, **style)
    ax.set_xlim(*limits)
    ax.set_xlabel(AXIS[cut])
    ax.set_ylabel(SYMBOL[component])
    ax.grid(alpha=.25, lw=.5)
    sci(ax)


def save(fig, stem):
    fig.savefig(HERE / (stem + '.pdf'))
    fig.savefig(HERE / (stem + '.png'), dpi=220)
    plt.close(fig)


def four_panels(kn, panels, stem):
    fig, axes = plt.subplots(2, 2, figsize=(5.3, 4.6))
    for ax, (component, cut) in zip(axes.flat, panels):
        panel(ax, kn, component, cut)
    fig.tight_layout(rect=(0, .12, 1, 1), h_pad=1.2, w_pad=1.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, handlelength=2.6, columnspacing=1.4)
    save(fig, stem)


HEAT = [('qx', 'horizontal'), ('qx', 'vertical'), ('qy', 'horizontal'), ('qy', 'vertical')]
four_panels('0.20', HEAT, 'R26_KN020_heat_flux_latest')
four_panels('0.20', [('ux', 'vertical'), ('uy', 'horizontal'), ('T', 'vertical'), ('T', 'horizontal')], 'R26_KN020_flow_latest')
four_panels('0.05', HEAT, 'R13_KN005_heat_flux_latest')

# R26 near the lid and the signed differences along x/L = 0.5, on the DSMC positions.
fig, axes = plt.subplots(1, 2, figsize=(5.3, 3.1))
panel(axes[0], '0.20', 'qx', 'vertical', limits=(.8, 1))
s, dsmc = data['0.20', 'SPARTA matched historical', 'qx', 'vertical']
at = {n: np.interp(s, *data['0.20', n, 'qx', 'vertical']) for n in ('N90', 'N120 step 36', 'N180 step 8')}
axes[1].axhline(0, color='#9A9A9A', lw=.6)
axes[1].plot(s, at['N120 step 36'] - dsmc, color='#1A1A1A', ls='-.', lw=1.0, label='N120 $-$ DSMC')
axes[1].plot(s, at['N120 step 36'] - at['N90'], color='#7A5195', ls=':', lw=1.5, label='N120 $-$ N90')
axes[1].plot(s, at['N180 step 8'] - at['N120 step 36'], color='#007F75', lw=1.3, label='N180 $-$ N120')
axes[1].set_xlim(0, 1)
axes[1].set_xlabel(AXIS['vertical'])
axes[1].set_ylabel(r'$\Delta q_x/(\rho_0 c_0^3)$')
axes[1].grid(alpha=.25, lw=.5)
sci(axes[1])
fig.tight_layout(rect=(0, .27, 1, 1), w_pad=1.6)
for ax, x0 in ((axes[0], .27), (axes[1], .77)):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(x0, 0), ncol=1, frameon=False, handlelength=2.4, fontsize=6.6)
save(fig, 'R26_KN020_near_lid_qx_latest')

# R13 close to the lid and to the right wall.
fig, axes = plt.subplots(1, 2, figsize=(5.3, 2.9))
panel(axes[0], '0.05', 'qx', 'vertical', limits=(.8, 1))
panel(axes[1], '0.05', 'qy', 'horizontal', limits=(.8, 1))
fig.tight_layout(rect=(0, .19, 1, 1), w_pad=1.6)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, handlelength=2.6, columnspacing=1.4, fontsize=7)
save(fig, 'R13_KN005_near_wall_q_latest')

# Signed qx maps: N180 step 8 minus N120 step 36 (left) and N120 step 36 minus DSMC (right), one colour scale.
with np.load(HERE / 'R26_KN020_qx_difference_maps.npz', allow_pickle=False) as maps:
    x, y = maps['x'], maps['y']
    fields = (maps['N180_step8_minus_N120_step36'], maps['N120_step36_minus_DSMC'])
limit = float(np.percentile(np.abs(fields[1]), 99))
fig, axes = plt.subplots(1, 2, figsize=(5.3, 2.75), constrained_layout=True)
for ax, values in zip(axes, fields):
    image = ax.pcolormesh(x, y, values, cmap='RdBu_r', vmin=-limit, vmax=limit, shading='nearest', rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x/L$')
    ax.set_ylabel(r'$y/L$')
    ax.set_xticks([0, .5, 1])
    ax.set_yticks([0, .5, 1])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(True)
bar = fig.colorbar(image, ax=axes, shrink=.8, pad=.02, extend='both')
bar.set_label(r'$\Delta q_x/(\rho_0 c_0^3)$')
bar.formatter = ScalarFormatter(useMathText=True)
bar.formatter.set_powerlimits((-2, 2))
bar.update_ticks()
save(fig, 'R26_KN020_qx_change_maps_latest')
print('rendered 6 figures in', HERE)
