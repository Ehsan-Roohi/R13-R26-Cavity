"""Independent stdlib helper; no R26 import or production solver integration.

Caller supplies density decoded by the AUTHENTICATED LogStateTransform and
the actual normalized problem.mass_weights, in C-order node order. This is
the derivative of the smooth valid-state mass equation, NOT of invalid-penalty
branches. There are 17 interleaved variables per node, density at offset zero.
"""
import math

def analytic_mass_entries(density, normalized_weights, mass_scale, *, nvar=17):
    rho=tuple(float(x) for x in density)
    weights=tuple(float(x) for x in normalized_weights)
    scale=float(mass_scale)
    if nvar != 17 or not rho or len(rho)!=len(weights):
        raise ValueError('R26 C-order node geometry required')
    if not math.isfinite(scale) or scale<=0:
        raise ValueError('positive finite fixed mass scaling required')
    if any(not math.isfinite(x) or x<=0 for x in rho):
        raise ValueError('valid decoded positive finite density required')
    if any(not math.isfinite(x) or x<0 for x in weights):
        raise ValueError('actual normalized nonnegative weights required')
    if not math.isclose(math.fsum(weights),1.,rel_tol=5e-14,abs_tol=5e-14):
        raise ValueError('weights must already be normalized by problem')
    entries={}
    for node,(w,r) in enumerate(zip(weights,rho)):
        value=w*r/scale
        if not math.isfinite(value) or (w>0 and value==0):
            raise ValueError('derivative overflow or underflow')
        if w>0:entries[17*node]=value
    return entries

def without_mass_row(pattern, mass_row, dimension):
    """Return COO dependency set with ENTIRE replaced row removed.

    This removes local false-positive non-density entries as well as the
    global density border. Residual values/equations are not modified.
    """
    if type(dimension) is not int or dimension<1 or type(mass_row) is not int or not 0<=mass_row<dimension:
        raise ValueError('row/dimension')
    result=set()
    for row,col in pattern:
        if type(row) is not int or type(col) is not int or not(0<=row<dimension and 0<=col<dimension):
            raise ValueError('COO index')
        if row!=mass_row:result.add((row,col))
    return result

def install_mass_row(finite_difference_entries, mass_row, dimension, exact_entries):
    """Replace whole row in a sparse mapping; do not add to stale FD row."""
    result={}
    for (row,col),value in finite_difference_entries.items():
        if not(0<=row<dimension and 0<=col<dimension) or not math.isfinite(value):
            raise ValueError('finite difference matrix')
        if row!=mass_row:result[row,col]=value
    if type(mass_row) is not int or not 0<=mass_row<dimension:
        raise ValueError('mass row')
    for col,value in exact_entries.items():
        if type(col) is not int or not 0<=col<dimension or col%17 or not math.isfinite(value) or value<=0:
            raise ValueError('analytic density entry')
        result[mass_row,col]=value
    return result
