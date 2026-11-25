"""Movie of the steady heat-flux field in the sharp-lid cavity: DSMC (pooled, 7 seeds) | R13 | R26, one Knudsen number per movie.
Background: temperature T/T_w - 1.  Moving tracers follow the heat-flux direction (speed proportional to |q|), so the motion shows
where heat goes, not a time history.  Grey lines: velocity streamlines.  Dashed black line: boundary of the region where heat flows
from cold to hot (q . grad T > 0).  For display only, every field is smoothed with a Gaussian of 1.5 cells (DSMC noise).
Usage: python render_steady_movie.py 0.2|0.05"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib import animation
from scipy.ndimage import gaussian_filter, map_coordinates

FFMPEG = r'<ffmpeg>'
matplotlib.rcParams['animation.ffmpeg_path'] = FFMPEG
H = Path(__file__).resolve().parent
kn = sys.argv[1]; tag = {'0.2': 'KN020', '0.05': 'KN005'}[kn]
Z = np.load(H / 'STEADY_FIELDS.npz')
N = 160; c = (np.arange(N) + .5) / N
SM = 1.5
def get(src, f): return gaussian_filter(Z['%s_%s_%s' % (tag, src, f)], SM, mode='nearest')
srcs = ['DSMC', 'R13', 'R26']
F = {s: {f: get(s, f) for f in ('theta', 'qx', 'qy', 'vx', 'vy')} for s in srcs}
# orientation check: rows = y, lid (vx > 0) at the last row
for s in srcs: assert F[s]['vx'][-3].mean() > 5 * abs(F[s]['vx'][2].mean()), s

rng = np.random.default_rng(7); NP = 2200; TAIL = 10; LIFE = 90
qmax = max(np.percentile(np.hypot(F[s]['qx'], F[s]['qy']), 99) for s in srcs)
def vel(s, P):   # P: (n,2) in [0,1]^2 -> heat-flux direction scaled by |q|/qmax
    ij = np.vstack([P[:, 1] * N - .5, P[:, 0] * N - .5])
    qx = map_coordinates(F[s]['qx'], ij, order=1, mode='nearest'); qy = map_coordinates(F[s]['qy'], ij, order=1, mode='nearest')
    m = np.hypot(qx, qy) + 1e-30; s_ = np.sqrt(m / qmax)
    return np.column_stack([qx, qy]) / m[:, None] * s_[:, None]
P0 = rng.uniform(.01, .99, (NP, 2)); age0 = rng.integers(0, LIFE, NP)
respawn = rng.uniform(.01, .99, (4000, 2 * NP))     # same respawn sequence for the three solutions
state = {s: dict(P=P0.copy(), age=age0.copy(), hist=[P0.copy()] * TAIL) for s in srcs}
DT = .006; ptr = [0]

PB = np.load(H / 'PARTS_B4.npz'); c40 = (np.arange(40) + .5) / 40
PARTS = [('F', r'Fourier part  $-\frac{15}{4}\mu\nabla T$'), ('S', r'stress part  $-\frac{3\mu}{2p}T\,\nabla\!\cdot\sigma$'),
         ('R', r'fourth-moment part  $-\frac{3\mu}{4p}\nabla\!\cdot R$'), ('Delta', r'scalar fourth-moment part  $-\frac{\mu}{4p}\nabla\Delta$'),
         ('N', r'nonlinear part (products of $\sigma$, $q$, $m$ with gradients)')]
PROJ = {}
for s in srcs:
    q = PB['%s_%s_q' % (tag, s)]; Y4, X4 = np.meshgrid(c40, c40, indexing='ij'); M4 = (X4 >= .1) & (X4 <= .9) & (Y4 >= .1) & (Y4 <= .9)
    qq = np.mean(np.sum(q[M4] ** 2, -1))
    for k, _ in PARTS: PROJ[s, k] = np.mean(np.sum(PB['%s_%s_%s' % (tag, s, k)][M4] * q[M4], -1)) / qq
qref = max(np.percentile(np.sum(PB['%s_%s_q' % (tag, s)] ** 2, -1), 99) for s in srcs)
def proj_field(s, k):   # local share q^a . q / |q|max^2 : > 0 carries heat along q, < 0 opposes it
    return np.sum(PB['%s_%s_%s' % (tag, s, k)] * PB['%s_%s_q' % (tag, s)], -1) / qref
FR = 450; SEG = FR // len(PARTS)
fig, axall = plt.subplots(2, 3, figsize=(16, 11.4), dpi=120, constrained_layout=True, gridspec_kw=dict(height_ratios=[1, 1], hspace=.12))
axes = axall[0]; axb = axall[1]
fig.set_facecolor('white')
th = F['DSMC']['theta'].ravel() - 1; lim = np.percentile(np.abs(th), 99.5)
norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
lcs = {}
for ax, s in zip(axes, srcs):
    pm = ax.pcolormesh(c, c, F[s]['theta'] - 1, cmap='RdBu_r', alpha=.95, norm=norm, shading='auto', rasterized=True)
    ax.streamplot(c, c, F[s]['vx'], F[s]['vy'], color=(.35, .35, .35, .55), linewidth=.6, density=1.1, arrowsize=.6)
    g4 = lambda a: gaussian_filter(a, 3.0, mode='nearest'); T4 = g4(F[s]['theta'])
    gx = np.gradient(T4, c, axis=1); gy = np.gradient(T4, c, axis=0)
    af = (g4(F[s]['qx']) * gx + g4(F[s]['qy']) * gy) > 0
    ax.contour(c, c, af.astype(float), levels=[.5], colors='#1b7837', linewidths=1.8, linestyles='--')
    lc = LineCollection([], linewidths=1.3, colors=[(0, 0, 0, 1)]); ax.add_collection(lc); lcs[s] = lc
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xlabel('$x/L$', fontsize=13); ax.tick_params(labelsize=11)
    ax.text(.5, 1.025, s, transform=ax.transAxes, ha='center', va='bottom', fontsize=16)
axes[0].set_ylabel('$y/L$', fontsize=13)
for ax in axes[1:]: ax.set_yticklabels([])
cb = fig.colorbar(pm, extend='both', ax=axes, shrink=.85, pad=.01); cb.set_label('$T/T_w-1$', fontsize=13); cb.ax.tick_params(labelsize=11)
fig.get_layout_engine().set(rect=(0, .05, 1, .95))
fig.text(.01, .028, 'Kn = %s.  Upper row: temperature; dark tracers move along the heat flux q; green dashed: heat flows cold to hot; grey: streamlines.' % kn, fontsize=11, color='0.25')

pn = TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2); ims = {}; lcb = {}; ptxt = {}
for ax, s in zip(axb, srcs):
    ims[s] = ax.pcolormesh(c40, c40, proj_field(s, 'F'), cmap='PuOr_r', norm=pn, shading='nearest', rasterized=True)
    lc = LineCollection([], linewidths=1.2); ax.add_collection(lc); lcb[s] = lc
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.set_xlabel('$x/L$', fontsize=13); ax.tick_params(labelsize=11)
    ptxt[s] = ax.text(.5, 1.02, '', transform=ax.transAxes, ha='center', va='bottom', fontsize=13)
axb[0].set_ylabel('$y/L$', fontsize=13)
for ax in axb[1:]: ax.set_yticklabels([])
cb2 = fig.colorbar(ims['DSMC'], ax=axb, shrink=.85, pad=.01, extend='both'); cb2.set_label(r'$q^{\alpha}\cdot q\,/\,\max|q|^2$', fontsize=13); cb2.ax.tick_params(labelsize=11)
ptitle = axb[1].text(.5, 1.11, '', transform=axb[1].transAxes, ha='center', va='bottom', fontsize=15, weight='bold')
fig.text(.01, .008, 'Lower row: one part of the heat flux at a time (4x4 cell blocks, as in table 7). Orange: the part carries heat along q;  purple: it opposes q.  Tracers move along the part.  Share = projection on q over 0.1-0.9.', fontsize=10.5, color='0.25')
PF = {}
def pvel(s, k, P):
    a = PB['%s_%s_%s' % (tag, s, k)]; ij = np.vstack([P[:, 1] * 40 - .5, P[:, 0] * 40 - .5])
    vx = map_coordinates(a[..., 0], ij, order=1, mode='nearest'); vy = map_coordinates(a[..., 1], ij, order=1, mode='nearest')
    m = np.hypot(vx, vy) + 1e-30; sc = np.sqrt(np.minimum(m * m / qref, 4.0))
    return np.column_stack([vx, vy]) / m[:, None] * sc[:, None]
NB = 900; Pb = {s: rng.uniform(.03, .97, (NB, 2)) for s in srcs}; Hb = {s: [Pb[s].copy()] * TAIL for s in srcs}; ageb = {s: rng.integers(0, LIFE, NB) for s in srcs}
frame = [0]
def step_parts():
    f = frame[0]; frame[0] += 1; k, lab = PARTS[min(f // SEG, len(PARTS) - 1)]
    if f % SEG == 0:
        ptitle.set_text(lab)
        for s in srcs:
            ims[s].set_array(proj_field(s, k).ravel()); ptxt[s].set_text('%s   share %+.2f' % (s, PROJ[s, k]))
    for s in srcs:
        P = Pb[s]; P = np.clip(P + DT * pvel(s, k, P), .002, .998); ageb[s] += 1; dead = (ageb[s] >= LIFE) | (f % SEG == 0)
        P[dead] = rng.uniform(.03, .97, (dead.sum(), 2)); ageb[s][dead] = 0
        h = Hb[s][1:] + [P.copy()]
        for x in h: x[dead] = P[dead]
        Hb[s] = h; Pb[s] = P; A = np.stack(h, 1)
        seg = np.stack([A[:, :-1], A[:, 1:]], 2).reshape(-1, 2, 2)
        col = np.zeros((len(seg), 4)); col[:, :3] = (.05, .05, .05)
        col[:, 3] = np.tile(np.linspace(.05, .9, TAIL - 1), NB) * np.repeat(np.minimum(1, np.minimum(ageb[s], LIFE - ageb[s]) / 12), TAIL - 1)
        lcb[s].set_segments(seg); lcb[s].set_color(col)

def step(_):
    k = ptr[0]; ptr[0] += 1
    for s in srcs:
        st = state[s]; P = st['P']
        k1 = vel(s, P); k2 = vel(s, np.clip(P + .5 * DT * k1, 0, 1))
        P = np.clip(P + DT * k2, .002, .998); st['age'] += 1
        dead = st['age'] >= LIFE
        r = respawn[k % 4000].reshape(NP, 2)
        P[dead] = r[dead]; st['age'][dead] = 0
        hist = st['hist'][1:] + [P.copy()]
        for h in hist: h[dead] = P[dead]
        st['hist'] = hist; st['P'] = P
        H_ = np.stack(hist, 1)   # (NP, TAIL, 2)
        seg = np.stack([H_[:, :-1], H_[:, 1:]], 2).reshape(-1, 2, 2)
        a = np.tile(np.linspace(.05, .95, TAIL - 1), NP)
        life = np.repeat(np.minimum(1, np.minimum(st['age'], LIFE - st['age']) / 12), TAIL - 1)
        col = np.zeros((len(seg), 4)); col[:, :3] = (.08, .08, .1); col[:, 3] = a * life
        lcs[s].set_segments(seg); lcs[s].set_color(col)
    if _ != 'spin': step_parts()
    return list(lcs.values())

for _ in range(LIFE): step('spin')      # spin-up so tracers fill the cavity at frame 1
out = H / ('cavity_heatflux_parts_%s.mp4' % tag)
ani = animation.FuncAnimation(fig, step, frames=FR, blit=False)
ani.save(out, writer=animation.FFMpegWriter(fps=30, bitrate=9000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p']))
print(out, out.stat().st_size)
