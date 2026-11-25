"""Time-dependent R26 for the sharp-lid cavity (owner 2026-11-25: "Yes, build R26").  Same discrete operator as the steady R26 of
the paper (R26NodeBVP with compatible_fv_bulk_residual, pinned source/, encoded variables log rho, log theta).  The interior bulk
rows of the steady residual are the Gu & Emerson (2009) balance laws without the time derivative, per unit volume, in the form
d_t Q + (steady row) = 0 with Q = (rho, rho vx, rho vy, rho theta, q, sigma, R, m, Delta) (momentum and temperature rows are
written for rho v and rho theta; all other rows carry only the time derivative of their own variable).  Wall, extrapolation and
corner rows and the mass-border row are algebraic.  Hence, in the scaled residual r(w) of the steady solver,
    Phi(w) = r(w) + D * (1.5 Q(w) - 2 Q^n + 0.5 Q^(n-1)) / dt         (BDF2; backward Euler first step)
with D = 1/bulk_scale on interior rows (0 elsewhere and on the mass-border row).  Each step: modified Newton with a lagged
Jacobian L = J_r + 1.5/dt D dQ/dw (J_r = producer finite-difference Jacobian, colour groups in parallel), LU reused while the
Newton iteration contracts; refreshed when it does not.  Initial state: gas at rest, rho = theta = 1; lid moves at t = 0.
Time unit L/sqrt(R T0).  Checkpoint after every saved frame; resumes from the last checkpoint after a requeue."""
import hashlib, json, os, signal, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent; O = H / 'output'; START = time.monotonic()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    C = json.loads((H / 'CONFIG.json').read_bytes())
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    if C.get('lid_ramp_width'): os.environ['R26_LID_RAMP_WIDTH'] = repr(float(C['lid_ramp_width']))   # Gaussian lid U0[1-exp(-a^2/w^2)][1-exp(-b^2/w^2)]
    else: os.environ.pop('R26_LID_RAMP_WIDTH', None)   # sharp lid
    if not C.get('local_test'):
        assert os.environ['SLURM_JOB_ID'].isdigit() and int(os.environ['SLURM_CPUS_PER_TASK']) == C['processes']
    for name, pin in C['pins'].items():
        assert sha(H / name) == pin, ('SOURCE_CHANGED', name)
    O.mkdir(exist_ok=True)

    def publish(name, v):
        t = O / (name + '.tmp')
        with t.open('w', encoding='utf8') as f:
            json.dump(v, f, indent=1, default=lambda o: o.tolist()); f.flush(); os.fsync(f.fileno())
        os.replace(t, O / name)

    def stop(*_): raise TimeoutError('SCHEDULER_SIGNAL')
    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, 'SIGUSR1'): signal.signal(signal.SIGUSR1, stop)
    paths = [str(H / 'source'), str(H)]; sys.path[:0] = paths
    import numpy as np
    if not hasattr(np, 'trapz'): np.trapz = np.trapezoid
    from scipy import sparse
    from scipy.sparse.linalg import splu
    from r26_cases import jfm_maxwell_cavity_case
    from r26_discretization import R26NodeBVP
    from r26_fv_backend import compatible_fv_bulk_residual, wall_bounded_control_volume_weights, fv_absolute_difference_step
    import r26_solver as solver
    from r26_mass_row_jacobian import hybrid_mass_jacobian
    from parallel_fd import ParallelFD

    N, KN, BETA, dt = C['N'], C['Kn'], C['beta'], C['dt']
    case = jfm_maxwell_cavity_case(N, kn=KN, lid_speed_m_per_s=100, wall_temperature_K=300, grid_stretch_beta=BETA)
    pb = R26NodeBVP(case, bulk_operator=compatible_fv_bulk_residual, mass_weights=wall_bounded_control_volume_weights(case.x, case.y))
    tr = solver.LogStateTransform(pb.shape); obj = solver.EncodedR26Objective(pb, tr, C['invalid_penalty'])
    S = dict(speed=100.0, obj=obj, pb=pb, pfd=None)
    TR = float(C.get('lid_time_ramp', 0.0))   # lid speed 100 sin^2(pi t / 2 TR) m/s for t < TR (start-up regularisation)

    def set_speed(s):
        if s == S['speed']: return
        if S['pfd'] is not None and hasattr(S['pfd'], 'close'): S['pfd'].close()
        cs = jfm_maxwell_cavity_case(N, kn=KN, lid_speed_m_per_s=s, wall_temperature_K=300, grid_stretch_beta=BETA)
        S['pb'] = R26NodeBVP(cs, bulk_operator=compatible_fv_bulk_residual, mass_weights=wall_bounded_control_volume_weights(cs.x, cs.y))
        S['obj'] = solver.EncodedR26Objective(S['pb'], tr, C['invalid_penalty']); S['speed'] = s; S['pfd'] = make_pfd(s)

    def speed_at(t):
        return 100.0 if TR <= 0 or t >= TR else float(100.0 * np.sin(.5 * np.pi * t / TR) ** 2)
    shape = pb.shape; n = int(np.prod(shape))
    Dm = np.zeros(shape); Dm[1:-1, 1:-1, :] = 1.0 / np.asarray(case.scaling.bulk, float); Dm[pb.mass_j, pb.mass_i, 0] = 0.0
    Dv = Dm.ravel()

    def Q(w):
        U = tr.decode(w); q = U.copy(); q[..., 1] = U[..., 0] * U[..., 1]; q[..., 2] = U[..., 0] * U[..., 2]; q[..., 3] = U[..., 0] * U[..., 3]
        return q.ravel()

    def dQdw(w):
        U = tr.decode(w); r, vx, vy, th = U[..., 0], U[..., 1], U[..., 2], U[..., 3]
        base = (np.arange(n).reshape(shape)[..., 0]).ravel()          # node offsets
        rows, cols, vals = [], [], []
        def add(a, b, v): rows.append(base + a); cols.append(base + b); vals.append(np.ravel(v))
        add(0, 0, r); add(1, 0, r * vx); add(1, 1, r); add(2, 0, r * vy); add(2, 2, r); add(3, 0, r * th); add(3, 3, r * th)
        for k in range(4, 17): add(k, k, np.ones_like(r))
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))

    make_pfd = lambda s: None if C['processes'] <= 1 else ParallelFD(dict(sys_path=paths, N=N, Kn=KN, beta=BETA, lid_speed_m_per_s=s, wall_temperature_K=300,
                                                            invalid_penalty=C['invalid_penalty']), C['processes'], context='spawn')
    S['pfd'] = make_pfd(100.0)
    lower, upper = tr.least_squares_bounds(); pattern = solver.jacobian_sparsity(pb)

    def jac(w):
        o = S['obj']; hs = fv_absolute_difference_step(w)
        J = hybrid_mass_jacobian(lambda v: np.asarray(o(v), float), w, method='2-point', abs_step=hs,
                                 bounds=(lower, upper), sparsity=pattern, problem=S['pb'], transform=tr, objective=o, fd_implementation=S['pfd'])
        if C.get('central_jacobian'):   # average of the +h and -h assemblies = central difference (as the steady worker)
            Jm = hybrid_mass_jacobian(lambda v: np.asarray(o(v), float), w, method='2-point', abs_step=-hs,
                                      bounds=(lower, upper), sparsity=pattern, problem=S['pb'], transform=tr, objective=o, fd_implementation=S['pfd'])
            J = 0.5 * (sparse.csr_matrix(J) + sparse.csr_matrix(Jm))
        return sparse.csr_matrix(J)

    # ---------------- state / resume
    ck = O / 'CHECKPOINT.npz'
    if ck.exists():
        z = np.load(ck); k0 = int(z['k']); w, wprev = z['w'], (z['wprev'] if z['has_prev'] else None)
        frames = list(z['frames']); times = list(z['times']); log = json.loads((O / 'LOG.json').read_text())
    else:
        U0 = np.zeros(shape); U0[..., 0] = 1.0; U0[..., 3] = 1.0
        w = tr.encode(U0); wprev = None; k0 = 0; frames = [U0]; times = [0.0]; log = []
    nsteps = int(round(C['t_end'] / dt)); Jr = None; lu = None; lu_coef = None; nJ = 0
    try:
        def solve_step(k, w, hist, coef, h, x):
            nonlocal Jr, lu, lu_coef, nJ
            obj = S['obj']; Phi = lambda x: np.asarray(obj(x), float) + Dv * (coef * Q(x) + hist) / h
            f = Phi(x); obj.invalid_evaluations = 0; its = 0; fresh = False; maxits = C['newton_max_total']
            while True:
                if float(np.abs(f).max()) <= C['newton_linf']: break
                if its >= maxits: raise RuntimeError('NEWTON_ITERATION_CAP step %d Linf %.3e' % (k, np.abs(f).max()))
                if lu is None or lu_coef != (coef, h) or (its >= C['newton_max_lagged'] and not fresh):
                    if fresh and lu is None: raise RuntimeError('NEWTON_FAILED step %d Linf %.3e' % (k, np.abs(f).max()))
                    Jr = jac(x); nJ += 1; L = (Jr + sparse.diags(Dv) @ dQdw(x) * (coef / h)).tocsc()
                    lu = splu(L); lu_coef = (coef, h); fresh = True; its = 0; S['obj'].invalid_evaluations = 0
                dx = -lu.solve(f); its += 1; f0 = float(np.linalg.norm(f)); alpha = 1.0
                while True:   # backtracking on the 2-norm (needed for the impulsive start)
                    xn = x + alpha * dx
                    try: fn = Phi(xn)
                    except (FloatingPointError, ValueError, OverflowError): fn = np.full_like(f, np.nan)
                    if np.isfinite(fn).all() and obj.invalid_evaluations == 0 and float(np.linalg.norm(fn)) < (1 - 1e-4 * alpha) * f0: break
                    obj.invalid_evaluations = 0; alpha *= .5
                    if alpha < 1e-3: break
                if alpha < 1e-3:
                    if fresh and its > 1: raise RuntimeError('NEWTON_DIVERGED step %d' % k)
                    if fresh: raise RuntimeError('NEWTON_NO_DESCENT step %d Linf %.3e' % (k, np.abs(f).max()))
                    lu = None; continue
                x, f = xn, fn
                if C.get('verbose'): print('  k %d h %.2e it %d fresh %s alpha %.3g Linf %.3e argmax %s' % (k, h, its, fresh, alpha, np.abs(f).max(), np.unravel_index(int(np.abs(f).argmax()), shape)), flush=True)
            assert obj.invalid_evaluations == 0, 'INVALID_STATE'
            return x, f, its

        for k in range(k0 + 1, nsteps + 1):
            if wprev is None:   # start-up: backward Euler substeps dt/64, dt/64, dt/32, ..., dt/2 (sum dt) from the state at rest
                ws = w
                ts = 0.0
                for h in [dt / 64] + [dt / 2 ** j for j in range(6, 0, -1)]:
                    ts += h; set_speed(speed_at(ts)); lu = None
                    ws, f, its = solve_step(k, ws, -Q(ws), 1.0, h, ws.copy())
                wprev, w = w, ws
            else:
                s = speed_at(k * dt)
                if s != S['speed']: set_speed(s); lu = None
                try:
                    x, f, its = solve_step(k, w, -2.0 * Q(w) + 0.5 * Q(wprev), 1.5, dt, 2 * w - wprev)
                except (RuntimeError, AssertionError) as e:   # adaptive: m backward-Euler substeps of dt/m (m = 2, 4, 8, 16) from w
                    print('substep step %d after %s' % (k, e), flush=True); x = None
                    for m in (2, 4, 8, 16):
                        try:
                            xs = w.copy()
                            for i in range(1, m + 1):
                                s2 = speed_at((k - 1) * dt + i * dt / m)
                                if s2 != S['speed']: set_speed(s2)
                                lu = None; S['obj'].invalid_evaluations = 0
                                xs, f, its = solve_step(k, xs, -Q(xs), 1.0, dt / m, xs.copy())
                            x = xs; print('  step %d done with %d substeps' % (k, m), flush=True); break
                        except (RuntimeError, AssertionError) as e2:
                            print('  %d substeps failed: %s' % (m, e2), flush=True)
                    if x is None: raise RuntimeError('STEP_FAILED %d' % k)
                    lu = None
                wprev, w = w, x
            if k % C['frame_every'] == 0:
                U = tr.decode(w); frames.append(U); times.append(k * dt)
                rec = dict(step=k, t=k * dt, newton_its=its, jacobians=nJ, Linf_phi=float(np.abs(f).max()),
                           lid_speed=S['speed'], steady_Linf=float(np.abs(np.asarray(S['obj'](w), float)).max()), elapsed=time.monotonic() - START)
                log.append(rec); print(json.dumps(rec), flush=True)
                with open(O / 'CHECKPOINT.tmp.npz', 'wb') as fh:
                    np.savez(fh, k=k, w=w, wprev=wprev, has_prev=True, frames=np.array(frames), times=np.array(times)); fh.flush(); os.fsync(fh.fileno())
                os.replace(O / 'CHECKPOINT.tmp.npz', ck); publish('LOG.json', log)
        np.savez_compressed(O / 'R26_TRANSIENT.npz', frames=np.array(frames), times=np.array(times), x=np.asarray(case.x), y=np.asarray(case.y), dt=dt, Kn=KN, beta=BETA, N=N)
        publish('DONE.json', dict(steps=nsteps, jacobians=nJ, final_steady_Linf=log[-1]['steady_Linf'] if log else None, elapsed=time.monotonic() - START))
    finally:
        if S['pfd'] is not None and hasattr(S['pfd'], 'close'): S['pfd'].close()


if __name__ == '__main__':
    main()
