"""Owner 2026-11-25 "Gaussian lid, all three": Gaussian lid (lid_ramp_width 0.0316, the grids with true steady roots: Kn 0.2 56x56, Kn 0.05 70x70 beta 1.25), central Jacobian, tolerance 1e-6, t_end 20.64 (82.6 ns).
Packet for the time-dependent R26 (r26_transient_worker.py) on the paper's sharp-lid grids: Kn 0.2 on 90x90 (beta 0),
Kn 0.05 on 60x60 (beta 1.25).  Slurm array 0-1, 32 CPUs (colour groups of the Jacobian in parallel), cpu-preempt, requeue,
checkpoint after every saved frame.  Source = pinned source/ and parallel_fd.py of the reviewed grid-sequencing packet."""
import hashlib, json, zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'R26_KN005_CORRECTED_OPERATOR_20261113/packet_gs/pkg_R26-KN020-GP64-FROM56/package.zip'
RUNTIME = '<cluster workspace>/MASS_REPAIR_SERIALREF_ISOLATED_RUNTIME_20261020/venv/bin/python'
sha = lambda b: hashlib.sha256(b).hexdigest(); job = 'R26-TRANSIENT-GAUSS'
z = zipfile.ZipFile(SRC); common = {n: z.read(n) for n in z.namelist() if n.startswith('source/') or n == 'parallel_fd.py'}
common['r26_transient_worker.py'] = (HERE / 'r26_transient_worker.py').read_bytes()
pins = {n: sha(b) for n, b in common.items() if n != 'r26_transient_worker.py'}
cases = {'KN020': dict(N=56, Kn=0.2, beta=0.0), 'KN005': dict(N=70, Kn=0.05, beta=1.25)}
members = {}
for c, p in cases.items():
    cfg = dict(p, dt=0.0645, t_end=20.64, frame_every=2, lid_time_ramp=0.25, lid_ramp_width=0.0316227766016838, central_jacobian=True, processes=32, invalid_penalty=1e8, newton_linf=1e-6,
               newton_max_lagged=15, newton_max_total=80, local_test=False, verbose=False, pins=pins)
    members[c + '/CONFIG.json'] = json.dumps(cfg, indent=1).encode()
    for n, b in common.items(): members[c + '/' + n] = b
members['run.sbatch'] = '\n'.join(['#!/bin/bash', '#SBATCH --partition=cpu-preempt', '#SBATCH --account=<account>', '#SBATCH --nodes=1',
    '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=32', '#SBATCH --mem=120G', '#SBATCH --time=2-00:00:00', '#SBATCH --array=0-1',
    '#SBATCH --signal=B:USR1@120', '#SBATCH --requeue', '#SBATCH --open-mode=append', 'set -euo pipefail', 'ulimit -c 0',
    'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1',
    'cases=(KN020 KN005)', 'cd "${cases[$SLURM_ARRAY_TASK_ID]}"', 'test -e output/DONE.json && exit 0',
    'exec %s -B r26_transient_worker.py >> worker.log 2>&1' % RUNTIME, '']).encode()
P = HERE / ('pkg_' + job); P.mkdir(exist_ok=False)
with zipfile.ZipFile(P / 'package.zip', 'x', zipfile.ZIP_DEFLATED) as zz:
    for n in sorted(members):
        info = zipfile.ZipInfo(n, date_time=(2026, 11, 25, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16; zz.writestr(info, members[n])
zb = (P / 'package.zip').read_bytes()
(P / 'PACKAGE_MANIFEST.json').write_text(json.dumps(dict(schema='R26_TRANSIENT_V1', zip=dict(bytes=len(zb), sha256=sha(zb)), cases=cases,
    job=dict(name=job, partition='cpu-preempt', account='<account>', ntasks=1, mem='120G', time='2-00:00:00', array='0-1', requeue=True, maximum_jobs=1),
    members={k: dict(bytes=len(v), sha256=sha(v)) for k, v in members.items()}), indent=1))
(P / 'run.sbatch').write_bytes(members['run.sbatch'])
t = (HERE.parent / 'DSMC_TRANSIENT_MOVIE_20261124/pkg_DSMC-MOVIE/unity_stage_submit.py').read_text(encoding='utf8')
for a, b in (('DSMC-MOVIE', job), ('DSMC_MOVIE_', 'R26_TRANSIENT_GAUSS_'), ("ntasks=32, mem='32G', time='12:00:00', array='0-15', partition='cpu,cpu-preempt'",
             "ntasks=1, cpus=32, mem='120G', time='2-00:00:00', array='0-1', partition='cpu-preempt'")):
    assert a in t, a; t = t.replace(a, b)
old = ("old = Path('<cluster workspace>/DSMC_LIDPROF_BUILD_f1d878ece1c648a5/src/spa_mpi')\n"
       "assert sha(old.read_bytes()) == 'e3f83ff2b9cbfa472637262dd614d3609d6a934c110d7c89ce49cfe593b8815f', 'OLD_SPARTA_BINARY'\n")
assert t.count(old) == 1, 'binary check'; t = t.replace(old, "assert Path(%r).exists(), 'RUNTIME_MISSING'\n" % RUNTIME)
t = t.replace('Submit the DSMC lid-profile build-and-test job ONCE', 'Submit the time-dependent R26 array job ONCE')
compile(t, 's', 'exec'); (P / 'unity_stage_submit.py').write_text(t, encoding='utf8')
print('built', job, sha(zb)[:16], len(zb))
