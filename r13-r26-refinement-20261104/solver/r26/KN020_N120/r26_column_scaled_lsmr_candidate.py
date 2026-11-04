"""Candidate only: not deployed to the active trajectory.

Invertible right scaling of A x=b; evaluate acceptance in original coordinates.
The minimum-norm solution can differ for rank-deficient A, so this is not a
bitwise or minimum-norm-equivalent replacement. Nonlinear guards remain needed.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator,lsmr
from time import monotonic

def solve_scaled(A,b,*,atol=1e-7,btol=1e-7,maxiter=None,seconds=None,row_weights=None):
    start=monotonic()
    def check_time():
        if seconds is not None and monotonic()-start >= seconds:
            raise TimeoutError('BOUNDED_LINEAR_SOLVE_EXPIRED')

    weighted = A if row_weights is None else A.multiply(row_weights[:, None]).tocsr()
    rhs = b if row_weights is None else row_weights*b
    norms=np.sqrt(np.asarray(weighted.power(2).sum(axis=0)).ravel())
    assert np.isfinite(norms).all()
    scale=np.ones_like(norms)
    np.divide(1.0,norms,out=scale,where=norms>0)
    assert np.isfinite(scale).all() and (scale>0).all()
    def matvec(z):
        check_time()
        return weighted@(scale*z)

    def rmatvec(y):
        check_time()
        return scale*(weighted.T@y)

    op=LinearOperator(A.shape,matvec=matvec,rmatvec=rmatvec,dtype=np.float64)
    check_time()
    result=lsmr(op,rhs,atol=atol,btol=btol,maxiter=maxiter)
    check_time()
    x=scale*result[0];r=A@x-b
    return x,{'iterations':int(result[2]),'istop':int(result[1]),
              'elapsed_seconds':monotonic()-start,
              'original_residual_l2':float(np.linalg.norm(r)),
              'original_normal_residual_l2':float(np.linalg.norm(A.T@r)),
              'scientific_acceptance':False}
