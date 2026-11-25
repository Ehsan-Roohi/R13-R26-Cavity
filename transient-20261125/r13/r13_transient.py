"""Time-dependent R13 for the sharp-lid cavity (owner 2026-11-24: "برو جلو").  Same spatial operator, wall closure, Maxwell
production and mass border as the paper's steady R13 (rana_original_reference_solver.assemble_system, matched Maxwell production):
    M(U) dU/dt + A(U) U - b(U) = 0,
M = diag over the 17 primitive variables: 1 for rho, rho for vx, vy, (3/2) rho for theta, 1 for q and sigma (their equations carry
only their own convective derivative), 0 for the algebraic closure rows (R, m, Delta are given by relations, not by evolution).
Linearly implicit BDF2 (backward Euler first step); coefficients frozen at the extrapolated state 2U^n - U^(n-1).
Initial state: gas at rest, rho = theta = 1; lid moves impulsively at t = 0 (as in the DSMC start-up runs).
Time unit L/sqrt(R T0) (= 4.0 ns for L = 1 um, argon at 300 K).
Usage: python r13_transient.py k20|k05 dt t_end frame_every out.npz"""
import io, json, sys, tarfile, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from nd_ordering import NDFactor
REV = Path(r'<revision workspace>/revision/KN005_R13_R26_GRID_REPRODUCTION_20261019')
import os
LID = os.environ.get('R13_LID_PROFILE')   # 'gauss' -> lid-profile solver of R13_LIDPROFILE_20261118 (U0[1-exp(-a^2/w^2)][1-exp(-b^2/w^2)], w = R13_LID_WIDTH)
sys.path.insert(0, str(Path(r'<project>/R13_LIDPROFILE_20261118/src_lid') if LID else REV / 'R13_MATCHED_MAXWELL_PREPARATION' / 'MATCHED_SOURCE'))
import rana_original_reference_solver as rs
from r13_maxwell_production import production_maxwell
rs.production = production_maxwell
AUD = REV / 'R13_INDEPENDENT_FINAL_SAVED_AUDIT_20261020'; TAR = REV / 'ROOT_TWO_R13_CLOSED_RECOVERY_20261020' / 'CLOSED_TWO_R13.tar'

def load_case(c):
    cfg = rs.RanaOriginalConfig(**json.loads((AUD / ('ACTUAL_%s_CANONICAL_AUDIT_ONCE.json' % c)).read_bytes())['configuration']); cfg.validate()
    man = json.loads((AUD / 'AUDIT_MANIFEST_NOT_GO.json').read_bytes())['cases'][c]
    with tarfile.open(TAR) as tf: steady = np.load(io.BytesIO(tf.extractfile(man['state_member']).read()))
    return cfg, steady

def mass_diag(U):
    m = np.zeros(U.shape); m[..., 0] = 1; m[..., 1] = U[..., 0]; m[..., 2] = U[..., 0]; m[..., 3] = 1.5 * U[..., 0]; m[..., 4:9] = 1
    return np.r_[m.ravel(), 0.0]

def run(c, dt, t_end, every, out):
    cfg, steady = load_case(c); n = cfg.nx * cfg.ny * rs.NVAR
    U0 = rs.equilibrium(cfg); z_prev = None; z = np.r_[U0.ravel(), 0.0]
    frames = [U0.copy()]; times = [0.0]; nsteps = int(round(t_end / dt)); t0 = time.time(); log = []
    for k in range(1, nsteps + 1):
        zx = z if z_prev is None else 2 * z - z_prev
        Ux = zx[:-1].reshape(cfg.ny, cfg.nx, rs.NVAR)
        A, b = rs.assemble_system(Ux, cfg); Md = mass_diag(Ux)
        if z_prev is None:   # backward Euler
            L = A + sparse.diags(Md / dt); r = b + Md / dt * z
        else:                # BDF2
            L = A + sparse.diags(1.5 * Md / dt); r = b + Md / dt * (2 * z - .5 * z_prev)
        znew = NDFactor(L, cfg.ny, cfg.nx).solve(r)
        z_prev, z = z, znew
        U = z[:-1].reshape(cfg.ny, cfg.nx, rs.NVAR)
        assert np.isfinite(U).all() and U[..., 0].min() > 0 and U[..., 3].min() > 0, ('nonphysical', k)
        if k % every == 0:
            frames.append(U.copy()); times.append(k * dt)
            d = float(np.max(np.abs(U - steady)) / np.max(np.abs(steady))) if not LID else float(np.max(np.abs(A @ z - b)))   # Gaussian lid: steady residual of the frozen system
            log.append(dict(step=k, t=k * dt, rel_maxdiff_to_steady=d, lagrange=float(z[-1]), wall=time.time() - t0))
            print('t %.3f  maxdiff/steady %.3e  lambda %.2e  %.0fs' % (k * dt, d, z[-1], time.time() - t0), flush=True)
    np.savez_compressed(out, lid=str(LID), frames=np.array(frames), times=np.array(times), steady=steady, nx=cfg.nx, kn_rana=cfg.kn, lid_speed=cfg.lid_velocity, dt=dt)
    Path(str(out) + '.json').write_text(json.dumps(dict(case=c, dt=dt, t_end=t_end, every=every, log=log, config=cfg.__dict__), indent=1, default=str))

if __name__ == '__main__':
    run(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
