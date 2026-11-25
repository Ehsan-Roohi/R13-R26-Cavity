"""Heat-flux parts q^F, q^sigma, q^R, q^Delta, q^N (and q) of DSMC (pool), R13, R26 on 4x4 cell blocks (40x40), exactly as the b4
lines of review_20261123/block_sensitivity.py that give table 7.  Writes PARTS_B4.npz with keys TAG_SRC_PART (40,40,2)."""
import sys
from pathlib import Path
import numpy as np
HB = Path(r'<project>/Manuscript/heatflux_balance_20261123'); sys.path.insert(0, str(HB))
import heatflux_balance as hb
def block(F, b=4):
    n = 160 // b; return {k: a.reshape((n, b, n, b) + a.shape[2:]).mean(axis=(1, 3)) for k, a in F.items()}
out = {}; c40 = (np.arange(40) + .5) / 40
for kn, K in hb.CASES.items():
    hb.centers = (np.arange(160) + .5) / 160
    src = {'DSMC': hb.dsmc_raw(K, 'POOLED'), 'R26': hb.r26_fields(K)[0], 'R13': hb.r13_fields(K)[0]}
    hb.centers = c40
    Y, X = np.meshgrid(c40, c40, indexing='ij'); M = (X >= .1) & (X <= .9) & (Y >= .1) & (Y <= .9)
    for n, F in src.items():
        P, q, gT = hb.budget(block(F), K['case'])
        for k in ('F', 'S', 'R', 'Delta', 'N'): out['%s_%s_%s' % (K['tag'], n, k)] = P[k]
        out['%s_%s_q' % (K['tag'], n)] = q
        qq = np.mean(np.sum(q[M] ** 2, -1))
        print(K['tag'], n, ' '.join('%s %+.2f' % (k, np.mean(np.sum(P[k][M] * q[M], -1)) / qq) for k in ('F', 'S', 'R', 'Delta', 'N')))
np.savez_compressed(Path(__file__).resolve().parent / 'PARTS_B4.npz', **out)
