from pathlib import Path
import os,sys,json,hashlib,time,resource,signal
resource.setrlimit(resource.RLIMIT_AS,(128*2**30,128*2**30))
signal.signal(signal.SIGALRM,lambda *args:sys.exit(124));signal.alarm(5300)
root=Path(__file__).resolve().parent
c=json.loads((root/'CONFIG.json').read_text())
for n,h in c['pins'].items():
 assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
assert os.environ.get('SLURM_CPUS_PER_TASK')=='32'
sys.path.insert(0,str(root))
import numpy as np
import r13_maxwell_adapter as adapter
m=adapter.load_archived_solver();adapter.install_candidate(m)
cfg=m.RanaOriginalConfig(**c['configuration']);cfg.validate()
out=root/'output';out.mkdir(exist_ok=False)
def put(name,obj):
 p=out/(name+'.tmp');p.write_text(json.dumps(m.json_safe(obj),indent=2,allow_nan=False));os.replace(p,out/name)
put('START.json',{'epoch':time.time(),'configuration':c['configuration'],'job':os.environ['SLURM_JOB_ID']})
def observe(record):
 if record.get('line_search_status')=='accepted':
  frame=sys._getframe(1)
  assert frame.f_code is m.solve_rana_jfnk.__code__
  a=np.array(frame.f_locals['updated'][:-1],copy=True).reshape(cfg.ny,cfg.nx,17)
  assert np.isfinite(a).all() and (a[...,0]>0).all() and (a[...,3]>0).all()
  dest=out/('accepted_%04d.npy'%int(record['outer_iteration']))
  with dest.with_suffix('.tmp').open('wb') as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
  os.replace(dest.with_suffix('.tmp'),dest)
  put('LATEST.json',{'checkpoint':dest.name,'record':record,'scientific_acceptance':False})
 with (out/'history.jsonl').open('a') as f:f.write(json.dumps(m.json_safe(record),allow_nan=False)+'\n')
u,solver=m.solve_rana_original(cfg,callback=observe)
with (out/'state.tmp').open('wb') as f:np.save(f,u,allow_pickle=False)
os.replace(out/'state.tmp',out/'state.npy')
r=m.build_result(cfg,u,solver,state_file='state.npy')
r['state_sha256']=hashlib.sha256((out/'state.npy').read_bytes()).hexdigest()
r['independent_audit_pending']=True
put('report.json',r)
sys.exit(0 if r.get('passed_physical_consistency_gates') else 2)
