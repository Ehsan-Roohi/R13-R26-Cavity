"""Finite original-solver pilot. Every iterate remains provisional, never adopted."""
import os,sys,json,hashlib,time,resource,signal
from pathlib import Path
H=Path(__file__).resolve().parent
C=json.loads((H/'CONFIG.json').read_bytes())
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
resource.setrlimit(resource.RLIMIT_AS,(128*2**30,128*2**30))
assert __debug__ and os.environ['SLURM_JOB_ID'].isdigit()
assert int(os.environ['SLURM_CPUS_PER_TASK'])==32
assert int(os.environ['SLURM_MEM_PER_NODE'])==131072
O=H/'output';O.mkdir(exist_ok=True)
start=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def syncdir():
 fd=os.open(O,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def _json_default(value):
 if hasattr(value,'tolist'):return value.tolist()
 raise TypeError('Unsupported JSON type: '+type(value).__name__)

def publish(name,v):
 p=O/name;t=O/(name+'.tmp')
 with t.open('w',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False,default=_json_default);f.flush();os.fsync(f.fileno())
 os.replace(t,p);syncdir()
def deadline():
 if time.monotonic()-start>=C['work_seconds']:raise TimeoutError('FINITE_PILOT_DEADLINE')
def stop(*_):raise TimeoutError('PREEMPT_OR_TERMINATION')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
for name,pin in C['source_hashes'].items():assert sha(H/name)==pin,('SOURCE_CHANGED',name)
assert sha(H/'seed.npz')==C['seed_sha256']
for name,pin in C['parent_hashes'].items():assert sha(H/name)==pin,('PARENT_CHANGED',name)
# Verify the Slurm cgroup hard limit, not merely the process address-space cap.
cg=[s.split(':',2)[2] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')]
assert len(cg)==1,'CGROUP_V2_REQUIRED'
limits=[];p=Path('/sys/fs/cgroup')/cg[0].lstrip('/')
while p!=Path('/sys/fs/cgroup'):
 f=p/'memory.max'
 if f.is_file():
  v=f.read_text().strip()
  if v!='max':limits.append(int(v))
 p=p.parent
assert limits and min(limits)<=128*2**30,'AGGREGATE_MEMORY_CAP_REQUIRED'
publish('RESOURCE.json',{'job_id':os.environ['SLURM_JOB_ID'],'CPUs':32,'cgroup_limits':limits,'hard_AS_bytes':128*2**30})
sys.path.insert(0,str(H/'source'))
import numpy as np
from dataclasses import asdict
from r26_cases import jfm_maxwell_cavity_case
from r26_discretization import R26NodeBVP
from r26_fv_backend import compatible_fv_bulk_residual,wall_bounded_control_volume_weights
import r26_solver as solver
from r26_validation import global_balance_diagnostics
from scipy.sparse import save_npz
parent_case=jfm_maxwell_cavity_case(C['parent_N'],kn=C['Kn'],lid_speed_m_per_s=100,wall_temperature_K=300,grid_stretch_beta=C['beta'])
parent_problem=R26NodeBVP(parent_case,bulk_operator=compatible_fv_bulk_residual,mass_weights=wall_bounded_control_volume_weights(parent_case.x,parent_case.y))
parent=np.load(H/'parent.npy',allow_pickle=False)
if C['parent_encoded']:
 parent=parent.reshape(C['parent_N'],C['parent_N'],17).copy();parent[...,0]=np.exp(parent[...,0]);parent[...,3]=np.exp(parent[...,3])
reference=np.load(H/'parent_residual.npy',allow_pickle=False).ravel() if C['parent_residual_npy'] else np.fromfile(H/'parent_residual.f64',dtype='<f8')
pev=parent_problem.evaluate(parent)
diffs={key:float(np.max(abs(np.asarray(arr).ravel()-reference))) for key,arr in [('raw',pev.unscaled_residual),('scaled',pev.residual)]}
assert C['parent_reference_kind'] in ('raw','scaled')
assert diffs[C['parent_reference_kind']]<=1e-12*(1+float(np.max(abs(reference)))),'EXACT_PARENT_OPERATOR_REPLAY_FAILED'
publish('PARENT_REPLAY.json',{'comparison_maxabs':diffs,'reference_kind':C['parent_reference_kind'],'reference_sha256':C['parent_hashes'],'scaled_RMS':float(np.sqrt(np.mean(pev.residual**2))),'passed':True,'bitwise_equality_claimed':False})
del parent,pev,parent_problem,reference
case=jfm_maxwell_cavity_case(C['N'],kn=C['Kn'],lid_speed_m_per_s=100,wall_temperature_K=300,grid_stretch_beta=C['beta'])
problem=R26NodeBVP(case,bulk_operator=compatible_fv_bulk_residual,mass_weights=wall_bounded_control_volume_weights(case.x,case.y))
with np.load(H/'seed.npz',allow_pickle=False) as z:state=z['state'].copy()
assert state.shape==problem.shape and np.isfinite(state).all()
tr=solver.LogStateTransform(problem.shape)
base=problem.evaluate(state);last_phi=.5*float(np.sum(base.residual**2));accepted=0;telemetry_count=0
def raw_full_sse(ev):
 raw=np.asarray(ev.unscaled_residual).ravel().copy()
 row=int(np.ravel_multi_index((problem.mass_j,problem.mass_i,0),problem.shape))
 raw[row]=ev.diagnostics.held_out_continuity
 return float(raw@raw+ev.diagnostics.mass_error**2)
last_raw_sse=raw_full_sse(base)
publish('BASELINE.json',{'diagnostics':asdict(base.diagnostics),'phi':last_phi,'RMS':float(np.sqrt(np.mean(base.residual**2))),'restored_continuity_plus_mass_raw_SSE':last_raw_sse,'scientific_acceptance':False})
publish('BALANCE_PREFLIGHT.json',global_balance_diagnostics(state,case))
# Retain only a completed current-grid J on native scratch, never export it.
original_hybrid=solver.hybrid_mass_jacobian
J_count=0
def retained_hybrid(*args,**kwargs):
 global J_count
 J=original_hybrid(*args,**kwargs);J_count+=1
 p=O/('J_%02d_NATIVE_ONLY.npz'%J_count);tmp=O/(p.name+'.tmp.npz')
 save_npz(tmp,J,compressed=False)
 with tmp.open('rb') as f:os.fsync(f.fileno())
 os.replace(tmp,p);syncdir()
 center=np.asarray(args[1]);center_path=O/('J_%02d_CENTER.npy'%J_count)
 with center_path.open('xb') as f:np.save(f,center);f.flush();os.fsync(f.fileno())
 publish('J_%02d_COMPLETE.json'%J_count,{'J_file':p.name,'J_sha256':sha(p),'center_sha256':sha(center_path),'shape':list(J.shape),'nnz':J.nnz,'configuration_sha256':sha(H/'CONFIG.json'),'native_only':True,'export_allowed':False,'reuse_admitted':False})
 deadline();return J
solver.hybrid_mass_jacobian=retained_hybrid
def checkpoint(u,iteration,evaluations,_):
 global last_phi,last_raw_sse,accepted
 deadline();ev=problem.evaluate(u);f=np.asarray(ev.residual).ravel();phi=.5*float(f@f)
 bal=global_balance_diagnostics(u,case)
 assert np.isfinite(u).all() and np.isfinite(f).all() and np.all(u[...,0]>0) and np.all(u[...,3]>0)
 assert phi<last_phi,'FRESH_ACTUAL_MERIT_DECREASE_REQUIRED'
 assert bal['wall_effective_pressure_min']>0 and bal['thermodynamic_pressure_min']>0,'POSITIVE_PRESSURE_REQUIRED'
 name='checkpoint_%04d.npz'%(accepted+1);p=O/name;tmp=O/(name+'.tmp')
 assert not p.exists()
 with tmp.open('xb') as out:
  np.savez_compressed(out,state=u,encoded=tr.encode(u),residual=f,x=case.x,y=case.y,accepted=np.array(False),diagnostic_only=np.array(True),scientific_accepted=np.array(False))
  out.flush();os.fsync(out.fileno())
 os.replace(tmp,p);syncdir()
 raw_sse=raw_full_sse(ev)
 rec={'path':name,'sha256':sha(p),'configuration_sha256':sha(H/'CONFIG.json'),'source_hashes':C['source_hashes'],'iteration':iteration,'solver_F_evaluations':evaluations,'phi_before':last_phi,'phi_after':phi,'restored_continuity_plus_mass_raw_SSE_before':last_raw_sse,'restored_continuity_plus_mass_raw_SSE_after':raw_sse,'raw_SSE_decreased':raw_sse<last_raw_sse,'both_original_merits_independently_audited':False,'RMS':float(np.sqrt(np.mean(f*f))),'diagnostics':asdict(ev.diagnostics),'balances':bal,'solver_line_accepted':True,'scientific_accepted':False,'original_seven_TR_guards_evaluated':False,'selector_promoted':False}
 publish('checkpoint_%04d.json'%(accepted+1),rec);publish('LATEST_COMPLETE.json',rec)
 accepted+=1;last_phi=phi;last_raw_sse=raw_sse
def progress(phase,event):
 deadline()
 if phase!='residual':publish('PHASE.json',{'phase':phase,'event':event,'elapsed':time.monotonic()-start,'accepted_provisional':accepted})
def telemetry(record):
 global telemetry_count
 deadline();telemetry_count+=1;publish('telemetry_%04d.json'%telemetry_count,record)
try:
 result=solver.solve_r26_bvp(problem,state,options=solver.SolveOptions(method='colored_newton',residual_tolerance=1e-9,held_out_continuity_tolerance=1e-7,max_iterations=2,max_function_evaluations=1000),checkpoint_callback=checkpoint,progress_callback=progress,iteration_telemetry=telemetry)
 publish('RESULT.json',{'status':'FINITE_PILOT_RETURNED','solver_reported_converged':result.converged,'scientific_accepted':False,'selector_promoted':False,'original_seven_TR_guards_evaluated':False,'accepted_provisional':accepted,'diagnostics':asdict(result.diagnostics),'message':result.message,'elapsed':time.monotonic()-start})
except BaseException as exc:
 publish('RESULT.json',{'status':'FINITE_PILOT_STOPPED','exception':type(exc).__name__,'reason':str(exc),'accepted_provisional':accepted,'scientific_accepted':False,'selector_promoted':False,'elapsed':time.monotonic()-start})
 raise
