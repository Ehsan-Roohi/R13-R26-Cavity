"""Owner 2026-11-24 ("New short DSMC run"): movies of the start-up of flow and heat transfer in the cavity.  Sharp lid (the paper's
reference decks, lid line unchanged), Kn_Gu 0.05 and 0.2, 8 independent seeds each (Slurm array 0-15, 32 MPI ranks per task).
Gas starts at rest at 300 K and the lid moves at 100 m/s from step 0.  Direct SPARTA moments (nrho u v temp heatx heaty) averaged over
blocks of 500 steps (50 samples, stride 10), 400 frames to step 200000 (the warm-up length used for the production runs).
No raw56, no restarts (one-hour runs; a preempted task is rerun from step 0).  Binary = lid-profile build (bit-identical without keyword)."""
import hashlib, json, zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
SEED = Path('<revision workspace>/revision/DSMC_KN005_SEVEN_REALIZATION_TIME_CONVERGENCE_SOURCE_PREPARATION_20261024/WAVE240K_FINITE48H_SEVEN_JOBS_SOURCE_ONLY/seed_179847619')
BINARY = '<cluster workspace>/DSMC_LIDPROF_BUILD_f1d878ece1c648a5/src/spa_mpi'
BINARY_SHA = 'e3f83ff2b9cbfa472637262dd614d3609d6a934c110d7c89ce49cfe593b8815f'
sha = lambda b: hashlib.sha256(b).hexdigest()
GAS = {'005': ('3.4538410244383913e+25', '5.2701431647314318e+06', '0.05'), '020': ('8.6346025610959782e+24', '1.3175357911828579e+06', '0.2')}
ref = (SEED / 'in.wave').read_text()
assert 'surf_collide         lid diffuse 300 1.0 translate 100 0.0 0.0\n' in ref and 'timestep             1.0323529147672158e-12' in ref
DECK = """# DSMC lid-driven cavity start-up (Gaussian lid, reference deck of the paper's DSMC); Kn_Gu = {kn}; seed index {i}.
units                si
seed                 {seed}
dimension            2
boundary             s s p

create_box           0.0 9.9999999999999995e-07 0.0 9.9999999999999995e-07 -0.5 0.5
create_grid          160 160 1

global               nrho {nrho} fnum {fnum} temp 300
species              argon.species Ar
mixture              gas Ar nrho {nrho} temp 300 vstream 0.0 0.0 0.0

surf_collide         fixed diffuse 300 1.0
surf_collide         lid diffuse 300 1.0 translate 100 0.0 0.0 lidprofile gauss 0.031622776601683794
bound_modify         xlo xhi ylo collide fixed
bound_modify         yhi collide lid

collide              vss gas maxwell.vss
create_particles     gas n 0
timestep             1.0323529147672158e-12

stats 1000
stats_style step cpu np nattempt ncoll nscoll
compute flow grid all gas nrho u v
compute thermal thermal/grid all gas temp
compute heat eflux/grid all gas heatx heaty
fix movavg ave/grid all 10 50 500 c_flow[*] c_thermal[*] c_heat[*] ave one
dump movblocks grid all 500 mov.block.* id xc yc f_movavg[*]
dump_modify movblocks format float %.9g
run 200000
"""
members = {}; cases = []
for kn in ('005', '020'):
    for i in range(8):
        name = 'KN%s_S%d' % (kn, i); cases.append(name); seed = 825110003 + 104729 * len(cases)
        members[name + '/in.mov'] = DECK.format(kn=GAS[kn][2], i=i, seed=seed, nrho=GAS[kn][0], fnum=GAS[kn][1]).encode()
        members[name + '/argon.species'] = (SEED / 'argon.species').read_bytes(); members[name + '/maxwell.vss'] = (SEED / 'maxwell.vss').read_bytes()
assert len({m.split(b'\n')[2] for k, m in members.items() if k.endswith('in.mov')}) == 16
members['run.sbatch'] = '\n'.join(['#!/bin/bash -l', '#SBATCH --partition=cpu,cpu-preempt', '#SBATCH --account=<account>', '#SBATCH --nodes=1',
    '#SBATCH --ntasks=32', '#SBATCH --cpus-per-task=1', '#SBATCH --mem=32G', '#SBATCH --time=12:00:00', '#SBATCH --array=0-15', '#SBATCH --requeue',
    'set -euo pipefail', 'module load openmpi/5.0.3', 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
    'BIN=%s' % BINARY, 'echo "%s  $BIN" | sha256sum -c -' % BINARY_SHA,
    'cases=(%s)' % ' '.join(cases), 'cd "${cases[$SLURM_ARRAY_TASK_ID]}"',
    'if [ -e log.sparta ]; then mkdir -p "prev_${SLURM_RESTART_COUNT:-0}" && mv -f log.sparta screen.out screen.err mov.block.* "prev_${SLURM_RESTART_COUNT:-0}"/ 2>/dev/null || true; fi',
    'mpirun -n 32 "$BIN" -in in.mov > screen.out 2> screen.err', 'touch DONE', '']).encode()
P = HERE / 'pkg_DSMC-MOVIE-GAUSS'; P.mkdir(exist_ok=False)
with zipfile.ZipFile(P / 'package.zip', 'x', zipfile.ZIP_DEFLATED) as z:
    for name in sorted(members):
        info = zipfile.ZipInfo(name, date_time=(2026, 11, 24, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
        z.writestr(info, members[name])
zb = (P / 'package.zip').read_bytes()
(P / 'PACKAGE_MANIFEST.json').write_text(json.dumps(dict(schema='DSMC_MOVIE_PACKAGE_V1', zip=dict(bytes=len(zb), sha256=sha(zb)), binary=dict(path=BINARY, sha256=BINARY_SHA), cases=cases,
    job=dict(name='DSMC-MOVIE-GAUSS', partition='cpu,cpu-preempt', account='<account>', ntasks=32, mem='32G', time='12:00:00', array='0-15', requeue=True, maximum_jobs=1),
    members={k: dict(bytes=len(v), sha256=sha(v)) for k, v in sorted(members.items())}), indent=1))
for c in ('KN005_S0', 'KN020_S0'): (P / (c + '.in')).write_bytes(members[c + '/in.mov'])
(P / 'run.sbatch').write_bytes(members['run.sbatch'])
t = (HERE.parent / 'DSMC_LIDPROFILE_20261118' / 'pkg_DSMC-LIDPROF-RUN' / 'unity_stage_submit.py').read_text(encoding='utf8')
for a, b in (('DSMC-LIDPROF-RUN', 'DSMC-MOVIE-GAUSS'), ('DSMC_LIDPROF_RUN_', 'DSMC_MOVIE_GAUSS_'),
             ("ntasks=32, mem='32G', time='2-00:00:00', array='0-3'", "ntasks=32, mem='32G', time='12:00:00', array='0-15'"),
             ("requeue=False", "requeue=True")):
    assert a in t, a; t = t.replace(a, b)
compile(t, 's', 'exec'); (P / 'unity_stage_submit.py').write_text(t, encoding='utf8')
print('built', sha(zb)[:16], len(zb), cases)
