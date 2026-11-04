"""N120 bounded continuation body, appended after the pinned pilot preflight."""
from r26_column_scaled_lsmr_candidate import solve_scaled

try:
    lo, hi = tr.least_squares_bounds()
    objective = solver.EncodedR26Objective(problem, tr, 1e12)
    if C.get('fresh_j_from_seed'):
        center = tr.encode(state)
        center_delta = 0.
        pattern = solver.jacobian_sparsity(problem)
        abs_step = solver.fv_absolute_difference_step(center)
        def counted(vector):
            deadline()
            return objective(vector)
        publish('PHASE.json', {'phase': 'fresh_jacobian_parity'})
        solver.checkpoint_group_parity(counted, center, bounds=(lo, hi), abs_step=abs_step,
                                       sparsity=pattern, problem=problem, objective=objective,
                                       observe=lambda value: publish('J_PARITY.json', value))
        publish('PHASE.json', {'phase': 'fresh_jacobian_assembly'})
        matrix = solver.hybrid_mass_jacobian(counted, center, method='2-point', abs_step=abs_step,
                                              bounds=(lo, hi), sparsity=pattern, problem=problem,
                                              transform=tr, objective=objective)
        matrix_path = O/'J_01_NATIVE_ONLY.npz'
        center_path = O/'J_01_CENTER.npy'
        pending = O/'J_01_NATIVE_ONLY.tmp.npz'
        save_npz(pending, matrix, compressed=False)
        os.replace(pending, matrix_path)
        np.save(center_path, center)
        syncdir()
        meta = {'J_file': matrix_path.name, 'J_sha256': sha(matrix_path), 'center_sha256': sha(center_path),
                'shape': list(matrix.shape), 'nnz': matrix.nnz, 'configuration_sha256': sha(H/'CONFIG.json'),
                'native_only': True, 'export_allowed': False}
        publish('J_01_COMPLETE.json', meta)
    else:
        recovery = C['native_J_recovery']
        native = Path(recovery['directory'])
        receipt = native / 'J_01_COMPLETE.json'
        assert sha(receipt) == recovery['receipt_sha256']
        meta = json.loads(receipt.read_text())
        assert meta['configuration_sha256'] == recovery['configuration_sha256']
        assert meta['native_only'] and not meta['export_allowed']
        matrix_path = native / meta['J_file']
        center_path = native / 'J_01_CENTER.npy'
        assert matrix_path.parent == native
        assert sha(matrix_path) == meta['J_sha256'] and sha(center_path) == meta['center_sha256']
        center = np.load(center_path, allow_pickle=False).ravel()
        center_delta = float(np.max(np.abs(center-tr.encode(state))))
        assert center_delta < 1e-14
        matrix = load_npz(matrix_path)
    state = tr.decode(center)
    base = problem.evaluate(state)
    residual = base.residual.ravel()
    invalid_before = objective.invalid_evaluations
    assert np.array_equal(objective(center), residual) and objective.invalid_evaluations == invalid_before
    assert list(matrix.shape) == meta['shape'] == [center.size, center.size]
    assert matrix.nnz == meta['nnz'] and np.isfinite(matrix.data).all()
    publish('NATIVE_J_ADMISSION.json', {
        'hashverified': True, 'center_maxabs': center_delta, 'native_only': True,
        'matrix_sha256': meta['J_sha256'], 'center_sha256': meta['center_sha256'],
    })
    initial_phi = .5 * float(residual @ residual)
    initial_raw = raw_full_sse(base)
    held_out = float(base.diagnostics.held_out_continuity)
    held_gradient = None
    if C.get('improve_held_out'):
        from r26_fv_backend import interior_control_volume_widths
        cv = interior_control_volume_widths(case.y)[:, None] * interior_control_volume_widths(case.x)[None, :]
        mass_j, mass_i, _ = base.mass_row
        indices = np.arange(center.size).reshape(problem.shape)
        functional = np.zeros_like(residual)
        functional[indices[1:-1, 1:-1, 0].ravel()] = (
            -cv * case.scaling.bulk[0] / cv[mass_j-1, mass_i-1]
        ).ravel()
        functional[indices[base.mass_row]] = 0.
        reconstructed = float(functional @ residual)
        assert abs(reconstructed - held_out) <= 1e-10, 'HELD_OUT_TELESCOPING_REPLAY'
        held_gradient = matrix.T @ functional
        held_gradient_squared = float(held_gradient @ held_gradient)
        assert np.isfinite(held_gradient_squared) and held_gradient_squared > 0
        publish('HELD_OUT_DIRECTION_MODEL.json', {'actual': held_out, 'reconstructed': reconstructed,
                                                  'gradient_norm': float(np.sqrt(held_gradient_squared))})
    mass_limit = max(abs(base.diagnostics.mass_error), 10*np.finfo(float).eps)
    base_balance = global_balance_diagnostics(state, case)
    assert base_balance['wall_effective_pressure_min'] > 0 and base_balance['thermodynamic_pressure_min'] > 0
    row_scales = np.empty(problem.shape)
    row_scales[1:-1, 1:-1] = case.scaling.bulk
    row_scales[0, 0] = row_scales[0, -1] = row_scales[-1, 0] = row_scales[-1, -1] = case.scaling.corner
    for node in problem.boundary_nodes:
        row_scales[node.j, node.i, :11] = case.scaling.wall
        row_scales[node.j, node.i, 11:] = case.scaling.extrapolation
    row_scales[base.mass_row] = case.scaling.mass
    assert np.all(np.isfinite(row_scales) & (row_scales > 0))
    # Alternate direction only. The accepted model and nonlinear tests use the original equations.
    mixed_weights = np.hypot(1., row_scales.ravel()/np.max(row_scales))
    linear_records = []
    trials = []
    selected = None
    for name, weights in (('column_scaled', None), ('physical_blend', mixed_weights)):
        deadline()
        publish('PHASE.json', {'phase': 'bounded_lsmr', 'method': name})
        linear_start = time.monotonic()
        try:
            direction, linear = solve_scaled(matrix, -residual, atol=1e-7, btol=1e-7,
                                             maxiter=300, seconds=180, row_weights=weights)
        except TimeoutError:
            linear_records.append({'method': name, 'status': 'TIME_LIMIT',
                                   'elapsed_seconds': time.monotonic()-linear_start})
            publish('LINEAR_RESULT.json', {'records': linear_records})
            continue
        linear.update(method=name, status='DIRECTION_RETURNED', iteration_cap=300,
                      time_cap_seconds=180, atol=1e-7, btol=1e-7)
        linear_records.append(linear)
        publish('LINEAR_RESULT.json', {'records': linear_records})
        if not np.isfinite(direction).all():
            continue
        if held_gradient is not None:
            # Restore the omitted continuity equation in the step model.
            correction = (float(held_gradient @ direction) + held_out) / held_gradient_squared
            direction = direction - correction * held_gradient
        publish('PHASE.json', {'phase': 'original_equation_line_search', 'method': name})
        for exponent in range(13):
            deadline()
            alpha = 2.**-exponent
            trial = np.clip(center+alpha*direction, lo, hi)
            trial_state = tr.decode(trial)
            # Restore the existing mean-density constraint in log coordinates before evaluating any gate.
            trial.reshape(problem.shape)[..., 0] += np.log(case.mean_density/problem.mean_density(trial_state))
            if not (np.isfinite(trial).all() and np.all(trial >= lo) and np.all(trial <= hi)):
                continue
            represented = trial-center
            model = residual+matrix@represented
            model_phi = .5*float(model@model)
            row = {'method': name, 'alpha': alpha, 'model_phi': model_phi}
            trials.append(row)
            if not np.isfinite(model_phi) or model_phi >= initial_phi*(1-1e-4*alpha):
                row['rejected'] = 'original_model_prediction'
                continue
            try:
                trial_state = tr.decode(trial)
                evaluation = problem.evaluate(trial_state)
                balance = global_balance_diagnostics(trial_state, case)
            except (ValueError, FloatingPointError, OverflowError):
                row['rejected'] = 'invalid_physical_evaluation'
                continue
            phi = .5*float(evaluation.residual.ravel() @ evaluation.residual.ravel())
            raw = raw_full_sse(evaluation)
            diag = evaluation.diagnostics
            checks = {
                'scaled_armijo': bool(np.isfinite(phi) and phi < initial_phi*(1-1e-4*alpha)),
                'restored_raw_merit_decrease': bool(np.isfinite(raw) and raw < initial_raw),
                'mass_constraint': bool(abs(diag.mass_error) <= mass_limit),
                'positive_density_temperature': bool(diag.min_density > 0 and diag.min_temperature > 0),
                'positive_wall_thermodynamic_pressure': bool(balance['wall_effective_pressure_min'] > 0 and
                                                            balance['thermodynamic_pressure_min'] > 0),
            }
            if held_gradient is not None:
                checks['held_out_continuity_decrease'] = abs(diag.held_out_continuity) < abs(held_out)
            row.update(phi=phi, raw_sse=raw, diagnostics=asdict(diag), checks=checks)
            if all(checks.values()):
                selected = (trial_state, evaluation, balance, row)
                break
        if selected is not None:
            break
    publish('TRIALS.json', {'rows': trials, 'initial_phi': initial_phi, 'initial_raw_sse': initial_raw})
    if selected is None:
        publish('RESULT.json', {'status': 'NO_GUARDED_STEP', 'accepted_provisional': 0,
                                'elapsed': time.monotonic()-start, 'scientific_acceptance': False})
    else:
        trial_state, evaluation, balance, row = selected
        diag = evaluation.diagnostics
        criteria = {'scaled_residual_1e-9': diag.total_linf <= 1e-9,
                    'held_out_continuity_1e-7': abs(diag.held_out_continuity) <= 1e-7}
        name = 'checkpoint_0001.npz'
        path = O/name
        temp = O/(name+'.tmp')
        with temp.open('xb') as handle:
            np.savez_compressed(handle, state=trial_state, encoded=tr.encode(trial_state),
                                residual=evaluation.residual.ravel(), x=case.x, y=case.y,
                                scientific_accepted=np.array(False))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        syncdir()
        checkpoint = dict(path=name, sha256=sha(path), configuration_sha256=sha(H/'CONFIG.json'),
                          iteration=1, RMS=diag.total_l2_rms, phi_before=initial_phi, phi_after=row['phi'],
                          restored_continuity_plus_mass_raw_SSE_before=initial_raw,
                          restored_continuity_plus_mass_raw_SSE_after=row['raw_sse'],
                          diagnostics=asdict(diag), balances=balance, step_checks=row['checks'],
                          original_solver_thresholds=criteria, scientific_accepted=False)
        publish('checkpoint_0001.json', checkpoint)
        publish('LATEST_COMPLETE.json', checkpoint)
        publish('RESULT.json', {'status': 'GUARDED_CHECKPOINT_PUBLISHED', 'accepted_provisional': 1,
                                'original_solver_thresholds': criteria,
                                'elapsed': time.monotonic()-start, 'scientific_acceptance': False})
except BaseException as exc:
    publish('RESULT.json', {'status': 'BOUNDED_CONTINUATION_STOPPED', 'exception': type(exc).__name__,
                            'reason': str(exc), 'scientific_acceptance': False})
    raise
