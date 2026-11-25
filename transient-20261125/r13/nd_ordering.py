"""Geometric nested-dissection ordering for the R13 JFNK preconditioner LU (owner 2026-11-18: "remove the solver ceiling").
The frozen bordered matrix has (ny*nx*17 + 1) unknowns: node (j, i) -> block j*nx + i of 17 variables, then one mass-border
row/column.  nd_factor(A, ny, nx) factors P A P^T with SuperLU (NATURAL column order, threshold pivoting) where P orders the
nodes by recursive bisection with separators two nodes wide (the stencil couples a node to its neighbours and, in the wall
rows, to the second neighbour) and puts the border last.  Fill grows like N^2 log N instead of N^3 (banded natural order),
which keeps the factor below SuperLU's 32-bit index limit on fine grids.  Any exact LU gives the same preconditioned
operator, so the converged solution is unchanged (same argument as for NATURAL vs COLAMD, NATURAL_CHECK_N40.json).
Same construction as nd_nodes/Factor of the R26 grid-sequencing worker (packet_gs/worker.py)."""
import numpy as np
import scipy.sparse.linalg as sla

NVAR = 17


def nd_permutation(ny, nx, width=2, leaf=6):
    def rec(j0, j1, i0, i1):
        nj, ni = j1 - j0, i1 - i0
        if (nj <= leaf and ni <= leaf) or min(nj, ni) <= width:
            return [(j, i) for j in range(j0, j1) for i in range(i0, i1)]
        if ni >= nj:
            m = i0 + (ni - width) // 2
            return rec(j0, j1, i0, m) + rec(j0, j1, m + width, i1) + [(j, i) for i in range(m, m + width) for j in range(j0, j1)]
        m = j0 + (nj - width) // 2
        return rec(j0, m, i0, i1) + rec(m + width, j1, i0, i1) + [(j, i) for j in range(m, m + width) for i in range(i0, i1)]
    order = rec(0, ny, 0, nx); assert len(order) == ny * nx == len(set(order))
    nodes = np.array([j * nx + i for j, i in order], dtype=np.int64)
    perm = (nodes[:, None] * NVAR + np.arange(NVAR)[None, :]).ravel()
    return np.concatenate([perm, [ny * nx * NVAR]])


class NDFactor:
    def __init__(self, matrix, ny, nx, pivot_threshold=0.1):
        n = matrix.shape[0]; assert matrix.shape == (n, n) and n == ny * nx * NVAR + 1, (matrix.shape, ny, nx)
        self.perm = nd_permutation(ny, nx); assert np.array_equal(np.sort(self.perm), np.arange(n))
        a = matrix.tocsr()[self.perm][:, self.perm].tocsc()
        self.lu = sla.splu(a, permc_spec='NATURAL', diag_pivot_thresh=pivot_threshold)
        self.shape = matrix.shape; self.nnz = int(self.lu.L.nnz + self.lu.U.nnz)

    def solve(self, b, trans='N'):
        b = np.asarray(b); x = np.empty_like(b, dtype=float); x[self.perm] = self.lu.solve(b[self.perm], trans=trans); return x

    def __getattr__(self, name):   # L, U, perm_r, ... of the permuted factor (diagnostics only)
        return getattr(self.lu, name)
