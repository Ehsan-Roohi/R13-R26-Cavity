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

fig, axes = plt.subplots(1, 3, figsize=(16, 6.0), dpi=120, constrained_layout=True)
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
fig.text(.01, .015, 'Kn = %s   |   dark tracers: heat-flux direction   |   green dashed: heat flows cold to hot   |   grey: streamlines' % kn, fontsize=11, color='0.25')

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
    return list(lcs.values())

for _ in range(LIFE): step(0)      # spin-up so tracers fill the cavity at frame 1
out = H / ('cavity_heatflux_steady_%s.mp4' % tag)
ani = animation.FuncAnimation(fig, step, frames=450, blit=False)
ani.save(out, writer=animation.FFMpegWriter(fps=30, bitrate=9000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p']))
print(out, out.stat().st_size)
