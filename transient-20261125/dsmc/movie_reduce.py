"""Seed average of the DSMC start-up frames (read-only on the run stage; writes only into the current directory).
For each Kn: mean over the 8 seeds of the 500-step block averages (nrho, u, v, T, heatx, heaty) on the 160x160 grid, frame k = steps
(500(k-1), 500k], frame 0 = initial state at step 0.  Nondimensional as in the paper: rho/rho0, u/C0, T/T0, q/(rho0 C0^3),
C0 = sqrt(208*300), rho0 = N0 * 6.6335e-26.  Writes MOVIE_FULL_<KN>.npz (float32, all frames, 160x160) and
MOVIE_SMALL.npz (float16, 2x2-cell blocks (80x80), frames 0-160 and every 4th after) for transfer.
Usage: python movie_reduce.py <run stage>"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
S = Path(sys.argv[1]); MASS = 6.6335e-26; C0 = (208.0 * 300.0) ** .5
N0 = {'KN005': 3.4538410244383913e+25, 'KN020': 8.6346025610959782e+24}
small = {}; info = {}
for kn in ('KN005', 'KN020'):
    seeds = sorted(d for d in S.iterdir() if d.is_dir() and d.name.startswith(kn + '_S'))
    assert len(seeds) == 8 and all((d / 'DONE').exists() for d in seeds), seeds
    steps = sorted(int(p.name.split('.')[-1]) for p in seeds[0].glob('mov.block.*')); assert steps == list(range(0, 200001, 500)), (steps[:3], len(steps))
    acc = np.zeros((len(steps), 6, 160, 160)); rho0 = N0[kn] * MASS
    scale = np.array([N0[kn], C0, C0, 300.0, rho0 * C0 ** 3, rho0 * C0 ** 3])
    for d in seeds:
        for k, st in enumerate(steps):
            a = np.loadtxt(d / ('mov.block.%d' % st), skiprows=9)
            assert a.shape == (25600, 9), (d.name, st, a.shape)
            i = np.floor(a[:, 1] / 1e-6 * 160).astype(int); j = np.floor(a[:, 2] / 1e-6 * 160).astype(int)
            f = np.empty((6, 160, 160)); f[:, j, i] = (a[:, 3:9] / scale).T
            acc[k] += f
    acc /= 8.0
    acc[0] = acc[1] * 0 + np.array([1, 0, 0, 1, 0, 0])[:, None, None]   # step-0 dump is taken before any sample: use the initial state
    np.savez_compressed('MOVIE_FULL_%s.npz' % kn, frames=acc.astype(np.float32), steps=np.array(steps), dt=1.0323529147672158e-12, names=np.array(['rho', 'u', 'v', 'T', 'qx', 'qy']))
    keep = list(range(0, 161)) + list(range(164, len(steps), 4))
    b = acc[keep].reshape(len(keep), 6, 80, 2, 80, 2).mean(axis=(3, 5))
    b[:, 0] -= 1; b[:, 3] -= 1                                        # store rho-1 and T-1 for float16 precision
    small[kn] = b.astype(np.float16); small[kn + '_steps'] = np.array(steps)[keep]
    info[kn] = dict(seeds=[d.name for d in seeds], frames=len(steps), kept=len(keep), final_T_range=[float(acc[-1, 3].min()), float(acc[-1, 3].max())])
np.savez_compressed('MOVIE_SMALL.npz', **small)
info['MOVIE_SMALL_sha256'] = hashlib.sha256(Path('MOVIE_SMALL.npz').read_bytes()).hexdigest(); info['bytes'] = Path('MOVIE_SMALL.npz').stat().st_size
Path('MOVIE_REDUCE_DONE.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info))
