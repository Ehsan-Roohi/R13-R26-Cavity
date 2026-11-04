"""Render physical centerline comparisons from the release's centerline_data.csv."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import numpy as np


GROUPS = {
    'flow': ('Density, velocity and temperature', ('rho', 'ux', 'uy', 'T')),
    'heat_flux': ('Heat flux', ('qx', 'qy')),
    'stress': ('Deviatoric stress', ('sigma_xx', 'sigma_xy', 'sigma_yy', 'sigma_zz')),
    'third_moments': ('Third moments', ('mxxx', 'mxxy', 'mxyy', 'myyy')),
    'fourth_moments': ('Deviatoric fourth moments', ('Rxx', 'Rxy', 'Ryy', 'Rzz')),
    'delta_and_A': ('Scalar and reduced fourth moments', ('Delta', 'Axx', 'Axy', 'Ayy', 'Azz')),
}
SYMBOL = {'rho': r'\rho/\rho_0', 'ux': r'u_x/U_w', 'uy': r'u_y/U_w', 'T': r'T/T_w',
          'qx': r'q_x/(\rho_0 c_0^3)', 'qy': r'q_y/(\rho_0 c_0^3)',
          'Delta': r'\Delta/(\rho_0 c_0^4)'}
for suffix in ('xx', 'xy', 'yy', 'zz'):
    SYMBOL['sigma_' + suffix] = r'\sigma_{' + suffix + r'}/(\rho_0 c_0^2)'
    for family in ('R', 'A'):
        SYMBOL[family + suffix] = family + '_{' + suffix + r'}/(\rho_0 c_0^4)'
for suffix in ('xxx', 'xxy', 'xyy', 'yyy'):
    SYMBOL['m' + suffix] = 'm_{' + suffix + r'}/(\rho_0 c_0^3)'

GRID_COLORS = {20: '#8463A1', 28: '#2568A4', 32: '#5D75A6', 40: '#007F75', 45: '#AD557E',
               50: '#A26C00', 60: '#C85427', 80: '#4B64AF', 90: '#87518E', 100: '#7F7736', 120: '#CF2148'}
MARKERS = ('o', 's', '^', 'v', 'D', 'P', 'X', 'h')
CUTS = ('horizontal', 'vertical')


def mesh(name):
    return int(name.split()[0][1:])


def read_profiles(path):
    cases = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    status = {}
    fixed_coordinates = {}
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            case = row['Kn'], row['model']
            label = row['series']
            cases[case][label][row['component'], row['cut']].append(
                (float(row['position_over_L']), float(row['value'])))
            status[case, label] = row['status']
            fixed_coordinates[row['Kn']] = float(row['fixed_coordinate_over_L'])
    for profiles in cases.values():
        for by_panel in profiles.values():
            for panel, values in by_panel.items():
                xy = np.array(values).T
                assert np.isfinite(xy).all() and np.all(np.diff(xy[0]) > 0)
                by_panel[panel] = xy
    return cases, status, fixed_coordinates


def panel(ax, cases, status, kn, model, name, cut, limits=(0, 1)):
    models = cases[kn, model]
    for i, label in enumerate(sorted(models, key=mesh)):
        xy = models[label].get((name, cut))
        if xy is None:
            continue
        state = status[(kn, model), label]
        linestyle = '--' if state == 'provisional' else (':' if state.startswith('archived') else '-')
        color = GRID_COLORS[mesh(label)]
        ax.plot(*xy, label=f'{label} · {state}', color=color, ls=linestyle, lw=1.5,
                marker=MARKERS[i % len(MARKERS)], ms=3, mfc='white', mew=.75,
                markevery=max(1, len(xy[0]) // 11), zorder=2 if state == 'provisional' else 1)
    refs = [(label, data[name, cut]) for label, data in cases[kn, 'DSMC'].items() if (name, cut) in data]
    assert len(refs) == 1, (kn, name, cut)
    label, xy = refs[0]
    ax.plot(*xy, label=label, color='#151515', lw=1.65, marker='.', ms=3.5, markevery=8, zorder=3)
    ax.set_xlim(*limits)
    if limits != (0, 1):
        visible = []
        for line in ax.lines:
            x, y = line.get_data()
            inside = (limits[0] <= x) & (x <= limits[1])
            visible.extend(y[inside])
            edges = [edge for edge in limits if x[0] <= edge <= x[-1]]
            visible.extend(np.interp(edges, x, y))
        lo, hi = min(visible), max(visible)
        pad = .08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel('$' + SYMBOL[name] + '$')
    ax.set_xlabel('$x/L$' if cut == 'horizontal' else '$y/L$')
    ax.grid(alpha=.2, linewidth=.5)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', scilimits=(-2, 2))


def figure(cases, status, kn, model, fixed, title, fields, zoom=False):
    columns = 1 if zoom else 2
    width = 9 if zoom else 11.6
    bottom = 1.4 if len(cases[kn, model]) > 5 else .95
    fig, axes = plt.subplots(len(fields), columns, figsize=(width, 2.24 * len(fields) + .7 + bottom), squeeze=False)
    fig.subplots_adjust(left=.11, right=.97, top=1 - .95 / fig.get_figheight(),
                        bottom=bottom / fig.get_figheight(), wspace=.34, hspace=.60)
    fig.suptitle(f'{model} · Kn = {kn} · {title}', fontsize=15, fontweight='semibold')
    for i, field in enumerate(fields):
        for j, cut in enumerate(('vertical',) if zoom else CUTS):
            ax = axes[i, j]
            panel(ax, cases, status, kn, model, field, cut, (.85, .97) if zoom else (0, 1))
            if i == 0:
                ax.set_title(f'{cut.capitalize()} cut · ' + ('y' if cut == 'horizontal' else 'x') + f'/L = {fixed:g}')
    legend = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    order = sorted(legend, key=lambda label: (not label.startswith('SPARTA'), mesh(label) if label.startswith('N') else 0))
    anchor = .18 / fig.get_figheight() if bottom > 1 else .025
    fig.legend([legend[label] for label in order], order, loc='lower center', ncol=4,
               bbox_to_anchor=(.5, anchor), frameon=False, fontsize=8.6)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--model', choices=('R13', 'R26'))
    parser.add_argument('--kn', choices=('0.05', '0.20'))
    args = parser.parse_args()
    output = args.directory
    cases, status, fixed = read_profiles(output / 'centerline_data.csv')
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'pdf.fonttype': 42})
    for kn in (args.kn,) if args.kn else ('0.05', '0.20'):
        code = kn.replace('.', '')
        for model in (args.model,) if args.model else ('R13', 'R26'):
            stem = f'{model}_KN{code}'
            groups = list(GROUPS.items())[:3] if model == 'R13' else list(GROUPS.items())
            with PdfPages(output / f'{stem}_all_profiles.pdf') as pdf:
                for key, (title, fields) in groups:
                    fig = figure(cases, status, kn, model, fixed[kn], title, fields)
                    fig.savefig(output / f'{stem}_{key}.png', dpi=180)
                    pdf.savefig(fig)
                    plt.close(fig)
                if kn == '0.20' and model == 'R26':
                    fig = figure(cases, status, kn, model, fixed[kn], 'Near-lid horizontal heat flux', ('qx',), zoom=True)
                    fig.savefig(output / f'{stem}_near_lid_qx.png', dpi=200)
                    pdf.savefig(fig)
                    plt.close(fig)


if __name__ == '__main__':
    main()
