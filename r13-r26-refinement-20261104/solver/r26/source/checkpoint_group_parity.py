"""At most19 real residual calls: fixed nine grouped/singleton sample parity.
Checks declared-pattern scheduling at the actual checkpoint; not global proof
of sparsity completeness. Charged to existing cooperative clock and eval count.
"""
import numpy as np
from structured425_groups import validated_groups

def checkpoint_group_parity(counted,x,*,bounds,abs_step,sparsity,problem,objective,observe):
    local=sparsity.tolil(copy=True)
    mr=int(np.ravel_multi_index((problem.mass_j,problem.mass_i,0),problem.shape))
    local[mr,:]=0;local=local.tocsr();local.eliminate_zeros()
    groups=validated_groups(problem.shape,local,np)
    lo,hi=bounds;h=np.asarray(abs_step,dtype=float).copy()
    # Match forward two-point signed absolute steps; reject rather than invent
    # a smaller physical perturbation if either sign cannot fit originalbounds.
    h=np.where(x+h>hi,-h,h)
    if not np.isfinite(h).all() or np.any(h==0) or np.any(x+h<lo) or np.any(x+h>hi) or np.any(x+h==x):
        raise RuntimeError('CHECKPOINT_PARITY_STEPS_NOT_ADMISSIBLE')
    calls=0
    def checked(z):
        nonlocal calls
        if calls>=19:raise RuntimeError('CHECKPOINT_PARITY_CALL_CAP')
        before=objective.invalid_evaluations;calls+=1;value=counted(z)
        if objective.invalid_evaluations!=before or not np.isfinite(value).all():
            raise RuntimeError('CHECKPOINT_PARITY_INVALID_SAMPLE')
        return value
    f0=checked(x);records=[]
    ny,nx,_=problem.shape
    points=((ny//2,nx//2,0),(ny//2,0,1),(ny//2,nx-1,3),
            (0,nx//2,7),(ny-1,nx//2,8),
            (0,0,4),(0,nx-1,5),(ny-1,0,6),(ny-1,nx-1,9))
    representatives=[int(np.ravel_multi_index(v,problem.shape)) for v in points]
    labels=[int(groups[col]) for col in representatives]
    if len(set(representatives))!=9 or len(set(labels))!=9:
        raise RuntimeError('CHECKPOINT_PARITY_NINE_UNIQUE_EDGE_COLUMNS_REQUIRED')
    for point,col,label in zip(points,representatives,labels):
        cols=np.flatnonzero(groups==label)
        rows=local[:,col].nonzero()[0]
        if not len(rows):raise RuntimeError('CHECKPOINT_PARITY_EMPTY_SELECTED_SUPPORT')
        grouped=x.copy();grouped[cols]+=h[cols]
        singleton=x.copy();singleton[col]+=h[col]
        fg=checked(grouped);fs=checked(singleton)
        dx=singleton[col]-x[col]
        if dx==0 or not np.isfinite(dx):raise RuntimeError('CHECKPOINT_PARITY_ZERO_ACTUAL_DX')
        dg=(fg[rows]-f0[rows])/dx;ds=(fs[rows]-f0[rows])/dx
        defect=np.abs(dg-ds);allowance=1e-7+1e-5*np.maximum(np.abs(dg),np.abs(ds))
        if not np.isfinite(dg).all() or not np.isfinite(ds).all() or np.any(defect>allowance):
            observe({'event':'checkpoint_parity_failed','label':label,'column':col,'calls':calls})
            raise RuntimeError('CHECKPOINT_GROUP_SINGLETON_PARITY_REJECTED')
        records.append({'point':list(point),'label':label,'column':col,'actual_dx':float(dx),'row_count':len(rows),'group_multiplicity':len(cols),'max_abs_derivative_difference':float(np.max(defect))})
    if calls!=19:raise RuntimeError('EXACT_CHECKPOINT_PARITY_COUNT')
    observe({'event':'checkpoint_parity_complete','residual_calls':calls,'groups':425,'mass_row_excluded':mr,'records':records,'not_global_sparsity_completeness_proof':True})
