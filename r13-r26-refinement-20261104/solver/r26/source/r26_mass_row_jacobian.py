"""Candidate hybrid assembly; valid center and EVERY FD sample enforced.

No externally supplied f0 is accepted. A fresh guarded center replaces the
center evaluation already performed by the original approx_derivative call.
Historical rejected line-search calls do not poison a subsequent valid center.
"""
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import coo_matrix
from exact_mass_row import analytic_mass_entries
from structured425_groups import validated_groups

class InvalidMassRowAssembly(RuntimeError):pass

def hybrid_mass_jacobian(counted,encoded,*,method,abs_step,bounds,sparsity,
                         problem,transform,objective,fd_implementation=None):
    if method!='2-point':raise ValueError('original two-point convention required')
    if fd_implementation is None:fd_implementation=approx_derivative
    # fd_implementation is a dependency injection ONLY for tiny guard tests;
    # production call does not pass it and uses captured SciPy approx_derivative.
    def checked(z):
        before=objective.invalid_evaluations
        value=counted(z)
        if objective.invalid_evaluations!=before:
            raise InvalidMassRowAssembly('invalid penalty branch at center or FD perturbation')
        if not np.isfinite(value).all():raise InvalidMassRowAssembly('nonfinite residual')
        return value
    f0=checked(encoded)
    state=transform.decode(encoded)
    if state.shape!=problem.shape or problem.shape[-1]!=17:
        raise ValueError('original C-order R26 shape required')
    mr=int(np.ravel_multi_index((problem.mass_j,problem.mass_i,0),problem.shape))
    local=sparsity.tolil(copy=True);local[mr,:]=0;local=local.tocsr();local.eliminate_zeros()
    groups=validated_groups(problem.shape,local,np)
    fd=fd_implementation(checked,encoded,method=method,abs_step=abs_step,bounds=bounds,sparsity=(local,groups),f0=f0).tocoo()
    if not np.isfinite(fd.data).all():raise InvalidMassRowAssembly('nonfinite FD matrix')
    entries=analytic_mass_entries(state[...,0].ravel().tolist(),problem.mass_weights.ravel().tolist(),problem.case.scaling.mass)
    # Remove ANY row entries, including stored zeros or a caller's stale row.
    keep=fd.row!=mr
    cols=np.fromiter(entries.keys(),dtype=np.int64,count=len(entries))
    vals=np.fromiter(entries.values(),dtype=float,count=len(entries))
    return coo_matrix((np.concatenate((fd.data[keep],vals)),
        (np.concatenate((fd.row[keep],np.full(len(entries),mr,dtype=np.int64))),
         np.concatenate((fd.col[keep],cols)))),shape=fd.shape).tocsr()
