from pathlib import Path
import json,time,sys
exec(Path('revision/grid_refinement_readonly_preflight.py').read_text().split("code='''")[0])
D=G/'R13_NEW_BOUNDARY_GRID_20261103'
p=json.loads((D/'PLAN.json').read_text())
code='from pathlib import Path\nimport json,hashlib,subprocess,shutil\nbase=Path('+repr(p['stage'])+')\n'
code+='''
old=base.with_name(base.name+'_CONT1')/'KN005_N60'
s=base.with_name(base.name+'_CONT2')/'KN005_N60'
job='66631684'
q=subprocess.check_output(['sacct','-j',job,'-n','-P','-o','JobID,State,ExitCode,ElapsedRaw'],text=True)
rows=[x.split('|') for x in q.splitlines() if x.strip()]
assert len(rows)==3 and {x[0] for x in rows}=={job,job+'.batch',job+'.extern'}
assert next(x for x in rows if x[0]==job)[1]=='TIMEOUT'
assert all(x[1].split()[0] in ('TIMEOUT','CANCELLED','COMPLETED') for x in rows)
assert job not in subprocess.check_output(['squeue','-h','-u','roohie_umass_edu','-o','%i'],text=True).split()
latest=json.loads((old/'output/LATEST.json').read_text())
assert latest['record']['line_search_status']=='accepted'
seed=old/'output'/latest['checkpoint'];h=hashlib.sha256(seed.read_bytes()).hexdigest()
assert h=='860520725152d75e0f283f69a05c6e023987e5c9dcbc6eedaa8fb1af28870438'
assert latest['record']['global_outer_iteration']==3
c=json.loads((old/'CONFIG.json').read_text())
ws=subprocess.check_output(['ws_list'],text=True)
assert 'cavity-cost-native-20261030' in ws
assert shutil.disk_usage(old).free>2*1024**3
s.mkdir(parents=True,exist_ok=False)
for n,pin in c['pins'].items():
 assert hashlib.sha256((old/n).read_bytes()).hexdigest()==pin
 shutil.copy2(old/n,s/n)
shutil.copy2(seed,s/'restart_state.npy')
c['pins']['restart_state.npy']=h
c['restart'].update(previous_job=job,previous_elapsed_seconds=10820,previous_accepted_iterations=3,source_checkpoint=str(seed),sha256=h,scheduler=q)
(s/'CONFIG.json').write_text(json.dumps(c,indent=2))
shell=(old/'run.sbatch').read_text().replace(str(old),str(s)).replace('--time=01:30:00','--time=03:00:00')
assert '--time=03:00:00' in shell and '--cpus-per-task=32' in shell and '--mem=128G' in shell
(s/'run.sbatch').write_text(shell)
review={'previous_job':job,'previous_consumption_seconds':10820,'global_offset':3,'checkpoint_sha256':h,'worker_unchanged':hashlib.sha256((s/'worker.py').read_bytes()).hexdigest()==c['pins']['worker.py'],'physics_unchanged':c['configuration']==json.loads((old/'CONFIG.json').read_text())['configuration'],'reservation_seconds':10800,'single_job_only':True,'workspace_listing':ws,'stage':str(s)}
(s/'PREPARE_REVIEW.json').write_text(json.dumps(review,indent=2))
print(json.dumps(review))
'''
r=ns['call'](code,'R13_CONT2_PREPARE_'+str(time.time_ns()),timeout=70)
(D/'CONT2_KN005_N60_PREPARE.json').write_text(json.dumps(r,indent=2))
print(r['stdout']);assert r['exit_code']==0,r
