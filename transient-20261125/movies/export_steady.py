"""Steady fields (pooled DSMC, R26, R13) of the paper's sharp-lid cavity at Kn 0.05 and 0.2 on the 160x160 DSMC cell centres,
read with the loaders of heatflux_balance.py (same states as table 7).  Writes STEADY_FIELDS.npz (nondimensional: rho, vx, vy, theta, qx, qy)."""
import sys, json
from pathlib import Path
import numpy as np
HB = Path(r'<project>/Manuscript/heatflux_balance_20261123'); sys.path.insert(0, str(HB))
import heatflux_balance as hb
out = {}; prov = {}
for kn, K in hb.CASES.items():
    src = {'DSMC': hb.dsmc_raw(K, 'POOLED')}
    src['R26'], prov['R26_%s' % K['tag']] = hb.r26_fields(K)
    src['R13'], prov['R13_%s' % K['tag']], _ = hb.r13_fields(K)
    for n, F in src.items():
        for f, a in dict(rho=F['rho'], vx=F['v'][..., 0], vy=F['v'][..., 1], theta=F['theta'], qx=F['q'][..., 0], qy=F['q'][..., 1]).items():
            out['%s_%s_%s' % (K['tag'], n, f)] = np.asarray(a, np.float64)
H = Path(__file__).resolve().parent
np.savez_compressed(H / 'STEADY_FIELDS.npz', **out)
(H / 'STEADY_FIELDS_PROVENANCE.json').write_text(json.dumps(prov, indent=1, default=str))
print(sorted(out)[:6], len(out))
