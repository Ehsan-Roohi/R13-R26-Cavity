"""Deterministic radius-two R26 FD groups, with actual-pattern validation.
No residual, FD-step, boundary, mass-row, line-search or chord-policy change.
"""

def validated_groups(shape, local, np):
    if len(shape)!=3 or shape[2]!=17 or min(shape[:2])<5:
        raise ValueError('requires at least 5x5 C-order17 grid')
    ny,nx,nc=map(int,shape); n=ny*nx*nc
    if local.shape!=(n,n):raise ValueError('pattern shape mismatch')
    j,i,k=np.indices(shape)
    groups=(((j%5)*5+i%5)*17+k).ravel(order='C').astype(np.int64)
    if len(np.unique(groups))!=425:raise ValueError('425 labels required')
    csr=local.tocsr(copy=True);csr.sum_duplicates();csr.eliminate_zeros()
    # Check every declared dependency, not merely the nominal stencil.
    # A nonlocal extension or an unremoved dense mass row fails closed.
    for r in range(n):
        cols=csr.indices[csr.indptr[r]:csr.indptr[r+1]]
        labels=groups[cols]
        if len(labels)!=len(np.unique(labels)):
            raise ValueError('FD color conflict at row '+str(r))
    return groups
