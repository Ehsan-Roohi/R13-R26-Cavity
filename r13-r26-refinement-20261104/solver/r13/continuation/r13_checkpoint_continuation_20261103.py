from pathlib import Path
import sys,json,time
exec(open('revision/grid_refinement_readonly_preflight.py',encoding='utf8').read().split("code='''")[0])
D=G/'R13_NEW_BOUNDARY_GRID_20261103';plan=json.loads((D/'PLAN.json').read_text());op,tag=sys.argv[1:3]
assert tag in ['KN005_N60','KN005_N80','KN020_N60','KN020_N80']
idx=plan['cases'].index(tag)
code='import json,hashlib,subprocess,shutil\nfrom pathlib import Path\nroot=Path('+repr(plan['stage'])+')\ntag='+repr(tag)+'\nidx='+str(idx)+'\n'
code+='''old=root/tag;s=root.with_name(root.name+'_CONT1')/tag
job=json.loads((root/'SUBMIT_RESULT.json').read_text())['stdout'].strip().split(';')[0]+'_'+str(idx)
q=subprocess.run(['sacct','-j',job,'-n','-P','-o','JobID,State,ExitCode,ElapsedRaw'],capture_output=True,text=True,check=True).stdout
rows=[x.split('|') for x in q.splitlines() if x.strip()]
assert len(rows)==3 and all(x[1].split()[0] in ('FAILED','TIMEOUT','CANCELLED','COMPLETED','PREEMPTED','OUT_OF_MEMORY','NODE_FAIL') for x in rows),q
assert {x[0] for x in rows}=={job,job+'.batch',job+'.extern'},q
main=next(x for x in rows if x[0]==job)
assert main[1]=='TIMEOUT' or (main[1]=='FAILED' and main[2]=='124:0'),'TIME_LIMIT_ONLY_RESTART_GUARD: '+q
assert not (old/'output/report.json').exists(),'FINAL_REPORT_REQUIRES_REVIEW'
latest=json.loads((old/'output/LATEST.json').read_text());assert latest['record']['line_search_status']=='accepted'
name=latest['checkpoint'];assert Path(name).name==name and name.startswith('accepted_') and name.endswith('.npy')
seed=old/'output'/name;seedhash=hashlib.sha256(seed.read_bytes()).hexdigest()
'''
if op=='prepare':
 code+='''s.mkdir(parents=True,exist_ok=False);c=json.loads((old/'CONFIG.json').read_text())
for n,h in c['pins'].items():
 assert hashlib.sha256((old/n).read_bytes()).hexdigest()==h
 shutil.copy2(old/n,s/n)
shutil.copy2(seed,s/'restart_state.npy');assert hashlib.sha256((s/'restart_state.npy').read_bytes()).hexdigest()==seedhash
w=s/'worker.py';t=w.read_text();needle='u,solver=m.solve_rana_original(cfg,callback=observe)';assert t.count(needle)==1
t=t.replace(needle,"initial=np.load(root/'restart_state.npy',allow_pickle=False)\\nassert initial.shape==(cfg.ny,cfg.nx,17) and np.isfinite(initial).all() and (initial[...,0]>0).all() and (initial[...,3]>0).all()\\nu,solver=m.solve_rana_original(cfg,initial=initial,callback=observe)")
t=t.replace("def observe(record):","def observe(record):\\n record=dict(record,global_outer_iteration=int(c['restart']['previous_accepted_iterations'])+int(record.get('outer_iteration',0)))")
observer="""original_preconditioner=m._jfnk_right_preconditioner
def monitored_preconditioner(*args,**kwargs):
 put('PHASE.json',{'phase':'preconditioner','event':'begin','epoch':time.time()})
 value=original_preconditioner(*args,**kwargs)
 put('PHASE.json',{'phase':'preconditioner','event':'end','epoch':time.time()})
 return value
m._jfnk_right_preconditioner=monitored_preconditioner
original_jv=m._jfnk_jacobian_vector
jv_calls=0
def monitored_jv(*args,**kwargs):
 global jv_calls
 value=original_jv(*args,**kwargs);jv_calls+=1
 if jv_calls%20==0:put('LINEAR_PROGRESS.json',{'Jv_calls':jv_calls,'epoch':time.time()})
 return value
m._jfnk_jacobian_vector=monitored_jv
"""
t=t.replace("initial=np.load(root/'restart_state.npy'",observer+"initial=np.load(root/'restart_state.npy'")
compile(t,str(w),'exec');w.write_text(t)
c['pins']['worker.py']=hashlib.sha256(w.read_bytes()).hexdigest();c['pins']['restart_state.npy']=seedhash
c['restart']={'previous_job':job,'previous_elapsed_seconds':int(main[3]),'previous_accepted_iterations':int(latest['record']['outer_iteration']),'source_checkpoint':str(seed),'sha256':seedhash,'scheduler':q,'scope':'State restart; solver adaptive history, lambda and initial scaling initialized by existing API, not bitwise trajectory continuation. Compare reassembled raw equation RMS across restart.','no_scientific_acceptance':True}
(s/'CONFIG.json').write_text(json.dumps(c,indent=2))
exec_line=next(x for x in (root/'run.sbatch').read_text().splitlines() if x.startswith('exec '))
shell='#!/bin/bash\\n#SBATCH --partition=cpu-preempt\\n#SBATCH --account=pi_roohie_umass_edu\\n#SBATCH --cpus-per-task=32\\n#SBATCH --mem=128G\\n#SBATCH --time=01:30:00\\n#SBATCH --no-requeue\\nset -euo pipefail\\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\\ncd "'+str(s)+'"\\n'+exec_line+'\\n'
(s/'run.sbatch').write_text(shell)
print(json.dumps({'prepared':str(s),'restart':c['restart']}))
'''
elif op=='submit':
 code+='''c=json.loads((s/'CONFIG.json').read_text());assert c['restart']['sha256']==seedhash
for n,h in c['pins'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
assert (s/'INDEPENDENT_REVIEW.json').exists(),'INDEPENDENT_REVIEW_REQUIRED'
review=json.loads((s/'INDEPENDENT_REVIEW.json').read_text());assert review['status']=='PASS'
assert review['configuration_sha256']==hashlib.sha256((s/'CONFIG.json').read_bytes()).hexdigest()
assert review['worker_sha256']==c['pins']['worker.py'] and review['restart_sha256']==seedhash
assert review['run_sbatch_sha256']==hashlib.sha256((s/'run.sbatch').read_bytes()).hexdigest()
with (s/'SUBMIT_INTENT.json').open('x') as f:f.write('{}')
p=subprocess.run(['sbatch','--parsable','--qos=normal','--chdir='+str(s),'--output='+str(s/'SLURM.out'),'--error='+str(s/'SLURM.err'),str(s/'run.sbatch')],capture_output=True,text=True,timeout=30)
r={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'stage':str(s)};(s/'SUBMIT_RESULT.json').write_text(json.dumps(r));print(json.dumps(r));assert p.returncode==0
'''
else:raise ValueError(op)
r=ns['call'](code,'R13_CONT1_'+op+'_'+str(time.time_ns()),timeout=70)
(D/('CONT1_'+tag+'_'+op+'_'+str(time.time_ns())+'.json')).write_text(json.dumps(r,indent=2));print(r['stdout']);assert r['exit_code']==0,r
