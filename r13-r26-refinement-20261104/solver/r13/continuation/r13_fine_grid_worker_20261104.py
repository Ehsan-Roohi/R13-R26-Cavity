"""Warm-start one finer R13 grid from the converged N80 physical state."""
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

resource.setrlimit(resource.RLIMIT_AS, (128 * 2**30, 128 * 2**30))
root = Path(__file__).resolve().parent
plan = json.loads((root / 'CONFIG.json').read_text())
for name, pin in plan['pins'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == pin, name
assert os.environ['SLURM_CPUS_PER_TASK'] == '32'
assert os.environ['SLURM_MEM_PER_NODE'] == '131072'
sys.path.insert(0, str(root))
import numpy as np
import r13_maxwell_adapter as adapter


def prolongate(parent, n):
    """Bilinear at interior nodes, with nearest interior values beyond parent coverage."""
    old_n = parent.shape[0]
    assert parent.shape == (old_n, old_n, 17)
    old = np.arange(1, old_n + 1) / (old_n + 1)
    new = np.arange(1, n + 1) / (n + 1)
    along_x = np.empty((old_n, n, 17))
    result = np.empty((n, n, 17))
    for k in range(17):
        for j in range(old_n):
            along_x[j, :, k] = np.interp(new, old, parent[j, :, k])
        for i in range(n):
            result[:, i, k] = np.interp(new, old, along_x[:, i, k])
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_state(state, n):
    assert state.shape == (n, n, 17) and np.isfinite(state).all()
    assert (state[..., 0] > 0).all() and (state[..., 3] > 0).all()


model = adapter.load_archived_solver()
adapter.install_candidate(model)
config = model.RanaOriginalConfig(**plan['configuration'])
config.validate()
assert config.nx == config.ny and config.nx in (100, 120)
assert config.continuity_mode == 'conservative-fv' and config.nonlinear_solver == 'jfnk'
parent = np.load(root / 'parent_state.npy', allow_pickle=False)
check_state(parent, 80)
seed = prolongate(parent, config.nx)
weights = model.conservative_volume_weights(config)
seed[..., 0] /= float(weights @ seed[..., 0].ravel())
check_state(seed, config.nx)
mass_error = float(weights @ seed[..., 0].ravel() - 1.)
assert abs(mass_error) <= 1e-14

output = root / 'output'
output.mkdir(exist_ok=False)


def publish(name, record):
    tmp = output / (name + '.tmp')
    with tmp.open('x') as handle:
        json.dump(model.json_safe(record), handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, output / name)


def save_state(name, state):
    tmp = output / (name + '.tmp')
    with tmp.open('xb') as handle:
        np.save(handle, state, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    dest = output / name
    os.replace(tmp, dest)
    return sha(dest)


publish('START.json', {'job': os.environ['SLURM_JOB_ID'], 'configuration': plan['configuration'],
                       'parent': plan['grid_parent'], 'preconditioner_ordering': plan['ordering']})
publish('TRANSFER.json', {'seed_sha256': save_state('seed.npy', seed), 'seed_mass_error': mass_error,
                          'parent_sha256': plan['pins']['parent_state.npy'],
                          'method': 'separable linear interpolation at interior nodes with nearest extension'})
original_lu = model.splu


def ordered_lu(matrix):
    start = time.monotonic()
    publish('PHASE.json', {'phase': 'preconditioner', 'event': 'begin', 'matrix_nnz': matrix.nnz,
                           'ordering': plan['ordering']})
    factor = original_lu(matrix, permc_spec=plan['ordering'])
    publish('PHASE.json', {'phase': 'preconditioner', 'event': 'end', 'factor_nnz': factor.nnz,
                           'seconds': time.monotonic() - start, 'ordering': plan['ordering']})
    return factor


model.splu = ordered_lu


def observe(record):
    if record.get('line_search_status') == 'accepted':
        frame = sys._getframe(1)
        assert frame.f_code is model.solve_rana_jfnk.__code__
        state = frame.f_locals['updated'][:-1].copy().reshape(config.ny, config.nx, 17)
        check_state(state, config.nx)
        name = f"accepted_{int(record['outer_iteration']):04d}.npy"
        pin = save_state(name, state)
        publish('LATEST.json', {'checkpoint': name, 'sha256': pin, 'record': record,
                                'scientific_acceptance': False})
    with (output / 'history.jsonl').open('a') as handle:
        handle.write(json.dumps(model.json_safe(record), allow_nan=False) + '\n')


state, solver = model.solve_rana_original(config, initial=seed, callback=observe)
check_state(state, config.nx)
result = model.build_result(config, state, solver, state_file='state.npy')
result['state_sha256'] = save_state('state.npy', state)
result['independent_audit_pending'] = True
publish('report.json', result)
sys.exit(0 if solver['converged'] and result['passed_physical_consistency_gates'] else 2)
