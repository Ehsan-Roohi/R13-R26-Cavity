"""Start-up movie: DSMC (mean of 8 seeds, 2x2-cell blocks) | transient R13, same physical time, sharp lid, gas at rest at t = 0.
Background: temperature T/T_w - 1; arrows: heat-flux direction (length ~ sqrt|q|); grey: velocity streamlines; green dashed: heat flows
cold to hot (q . grad T > 0).  DSMC is smoothed with a Gaussian of 1 block for display.  Time in ns (time unit L/sqrt(R T0) = 4.003 ns).
Usage: python render_startup_movie.py k20|k05"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import animation
from scipy.ndimage import gaussian_filter
matplotlib.rcParams['animation.ffmpeg_path'] = r'<ffmpeg>'
H = Path(__file__).resolve().parent; P = H.parent
c = sys.argv[1]; tag = {'k20': 'KN020', 'k05': 'KN005'}[c]; kn = {'k20': '0.2', 'k05': '0.05'}[c]
REV = Path(r'<revision workspace>/revision/KN005_R13_R26_GRID_REPRODUCTION_20261019')
sys.path.insert(0, str(REV / 'R13_MATCHED_MAXWELL_PREPARATION' / 'MATCHED_SOURCE'))
import rana_original_reference_solver as rs
from r13_maxwell_production import production_maxwell
rs.production = production_maxwell
sys.path.insert(0, str(P / 'R13_TRANSIENT_20261124')); import r13_transient as rt
TU = 1e-6 / (208.0 * 300.0) ** .5 * 1e9      # ns per time unit
DSTEP = 1.0323529147672158e-12 * 1e9          # ns per DSMC step

# ---------------- DSMC
D = np.load(P / 'DSMC_TRANSIENT_MOVIE_20261124' / 'fetched' / 'MOVIE_SMALL.npz')
dF = D[tag].astype(np.float64); dsteps = D[tag + '_steps']; dt_ns = dsteps * DSTEP    # frame k averages (t_k - 500 steps, t_k]
dF[:, 0] += 1; dF[:, 3] += 1
dmid = np.maximum(dt_ns - 250 * DSTEP, 0)                                          # mid-time of each block
c80 = (np.arange(80) + .5) / 80
# ---------------- R13
Z = np.load(P / 'R13_TRANSIENT_20261124' / ('R13_TRANSIENT_%s.npz' % c.upper()))
cfg, _ = rt.load_case(c); n = cfg.nx; xn = np.r_[0.0, np.arange(1, n + 1) / (n + 1), 1.0]
def r13_full(u):
    full = np.zeros((n + 2, n + 2, 17)); full[1:-1, 1:-1] = u
    full[1:-1, 0] = rs.eliminated_wall_state(u, cfg, side='left'); full[1:-1, -1] = rs.eliminated_wall_state(u, cfg, side='right')
    full[0, 1:-1] = rs.eliminated_wall_state(u, cfg, side='bottom'); full[-1, 1:-1] = rs.eliminated_wall_state(u, cfg, side='top')
    for (a, b), (c1, c2) in {(0, 0): ((0, 1), (1, 0)), (0, -1): ((0, -2), (1, -1)), (-1, 0): ((-1, 1), (-2, 0)), (-1, -1): ((-1, -2), (-2, -1))}.items():
        full[a, b] = .5 * (full[c1] + full[c2])
    return full
rtimes = Z['times'] * TU
# ---------------- frames: DSMC frame times; R13 frame nearest in time (R13 frames every 0.129 units = 500 DSMC steps)
lid = cfg.lid_velocity
fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6), dpi=130, constrained_layout=True)
fig.get_layout_engine().set(rect=(0, .11, 1, .89))
lim = .03; norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
for ax, s in zip(axes, ('DSMC', 'R13 (time-dependent)')):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.set_xlabel('$x/L$', fontsize=13); ax.tick_params(labelsize=11)
    ax.text(.5, 1.02, s, transform=ax.transAxes, ha='center', va='bottom', fontsize=16)
axes[0].set_ylabel('$y/L$', fontsize=13); axes[1].set_yticklabels([])
sm = plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'); cb = fig.colorbar(sm, ax=axes, shrink=.85, pad=.01, extend='both'); cb.set_label('$T/T_w-1$', fontsize=13)
ttl = fig.text(.45, .075, '', ha='center', fontsize=15, weight='bold')
fig.text(.01, .008, 'Kn = %s; gas at rest, lid starts at t = 0.  Arrows: heat flux;  grey: streamlines;  green dashed: cold-to-hot heat flow (q.grad T smoothed over 3 blocks).\nDSMC: mean of 8 runs, up to 13 blocks of 500 steps.' % kn, fontsize=10.5, color='0.25')
art = []
Q = 20; qi = np.linspace(.03, .97, Q)
def draw(F, x, ax, qsc):
    T, u, v, qx, qy = F
    art.append(ax.pcolormesh(x, x, T - 1, cmap='RdBu_r', norm=norm, shading='gouraud', rasterized=True))
    sp = np.hypot(u, v)
    if sp.max() > 1e-3 * lid:
        ax.streamplot(x, x, u, v, color=(.3, .3, .3, .5), linewidth=.6, density=1.0, arrowsize=.6)
    gx = np.gradient(T, x, axis=1); gy = np.gradient(T, x, axis=0); sg = len(x) / 80 * 3.0; af = gaussian_filter(qx * gx + qy * gy, sg, mode='nearest')
    if af.max() > 0 > af.min(): art.append(ax.contour(x, x, af, levels=[0.0], colors='#1b7837', linewidths=1.6, linestyles='--'))
    from scipy.interpolate import RegularGridInterpolator as RGI
    Y, X = np.meshgrid(qi, qi, indexing='ij'); pts = np.column_stack([Y.ravel(), X.ravel()])
    ax_ = RGI((x, x), qx)(pts); ay_ = RGI((x, x), qy)(pts); m = np.hypot(ax_, ay_) + 1e-30
    L = np.sqrt(m / qsc).clip(0, 1.4) * .035
    art.append(ax.quiver(X.ravel(), Y.ravel(), ax_ / m * L, ay_ / m * L, angles='xy', scale_units='xy', scale=1, width=.0028, color='k', alpha=.75))
qsc_ref = [None]
def frame(k):
    art.clear()
    for ax, s in zip(axes, ('DSMC', 'R13 (time-dependent)')):
        ax.cla(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.set_xlabel('$x/L$', fontsize=13); ax.tick_params(labelsize=11)
        ax.text(.5, 1.02, s, transform=ax.transAxes, ha='center', va='bottom', fontsize=16)
    axes[0].set_ylabel('$y/L$', fontsize=13); axes[1].set_yticklabels([])
    t = dmid[k]
    h = min(k // 4, 6); w = dF[max(k - h, 1) if k else 0:k + h + 1].mean(0)   # centred time window of 2h+1 blocks (h grows from 0 to 6)
    Fd = [gaussian_filter(w[j], 1.0 if j < 4 else 2.0, mode='nearest') for j in (3, 1, 2, 4, 5)]
    ir = int(np.argmin(np.abs(rtimes - t))); full = r13_full(Z['frames'][ir])
    ts = rs  # noqa
    Fr = [full[..., 3], full[..., 1], full[..., 2], full[..., 4], full[..., 5]]
    if qsc_ref[0] is None: qsc_ref[0] = np.percentile(np.hypot(Z['steady'][..., 4], Z['steady'][..., 5]), 95)
    draw(Fd, c80, axes[0], qsc_ref[0]); draw(Fr, xn, axes[1], qsc_ref[0])
    ttl.set_text('t = %5.1f ns' % t)
    return art
K = int(np.searchsorted(dsteps, 160 * 500)) + 1; idx = [k for k in range(0, K) for _ in range(4 if k < 24 else 1)]   # first 12 ns shown 4x slower
out = H / ('cavity_startup_DSMC_R13_%s.mp4' % tag)
ani = animation.FuncAnimation(fig, frame, frames=idx, blit=False)
ani.save(out, writer=animation.FFMpegWriter(fps=12, bitrate=9000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p']))
print(out, out.stat().st_size)
