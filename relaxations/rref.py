from coreUtils import nintu
import numpy as np
from numba import njit, jit

from numpy import argmax, abs, ones, zeros, arange, outer, around, all

from math import log10

@njit
def rref(B, tol=1e-10, doCopy=True):
    A = B.copy() if doCopy else B
    rows, cols = A.shape
    r = 0
    n_piv_pos = 0
    pivots_pos = -ones((2,cols))
    row_exchanges = arange(rows)
    
    swaps = zeros((2,), nintu)
    swapsR = zeros((2,), nintu)
    
    m = 0.

    for c in range(cols):
        if __debug__:
            print("Now at row", r, "and col", c, "with matrix:");
            print(A)

        ## Find the pivot row:
        pivot = argmax(abs(A[r:rows, c])) + r
        m = float(abs(A[pivot, c]))
        if __debug__:
            print("Found pivot, ", m, " in row ", pivot)
        if m <= tol:
            ## Skip column c, making sure the approximately zero terms are
            ## actually zero.
            A[r:rows, c] = zeros(rows - r)
            if __debug__:
                print("All elements at and below (", r, ",", c, ") are zero.. moving on..")
        else:
            ## keep track of bound variables
            pivots_pos[0,n_piv_pos]=r
            pivots_pos[1,n_piv_pos]=c
            n_piv_pos += 1

            if pivot != r:
                ## Swap current row and pivot row
                swaps[0] = pivot
                swaps[1] = r
                swapsR = swaps[::-1]
                A[swaps, c:cols] = A[swapsR, c:cols]
                row_exchanges[swaps] = row_exchanges[swapsR]

                if __debug__:
                    print("Swap row", r, "with row", pivot, "Now:"); print(A)

            ## Normalize pivot row
            A[r, c:cols] = A[r, c:cols] / A[r, c];

            ## Eliminate the current column
            v = A[r, c:cols]
            ## Above (before row r):
            if r > 0:
                ridx_above = arange(r)
                A[ridx_above, c:cols] = A[ridx_above, c:cols] - outer(v, A[ridx_above, c]).T
                if __debug__:
                    print("Elimination above performed:"); print(A)
            ## Below (after row r):
            if r < rows - 1:
                ridx_below = arange(r + 1, rows)
                A[ridx_below, c:cols] = A[ridx_below, c:cols] - np.outer(v, A[ridx_below, c]).T
                if __debug__:
                    print("Elimination below performed:"); print(A)
            r += 1
        ## Check if done
        if r == rows:
            break
    return (A, pivots_pos[:, :n_piv_pos], row_exchanges)


@jit
def robustRREF(B, tolCalc_:float = 1e-6, tol0_:float = 1e-5, fullOut = False):
    U1, piv1, rows1 = rref(B, tol=tolCalc_, doCopy=True)
    U2, piv2, rows2 = rref(B, tol=tolCalc_/10, doCopy=True)

    # Round
    dec = -int(log10(tol0_))
    U1r = around(U1, decimals=dec)
    U2r = around(U2, decimals=dec)

    if not all(U1r==U2r):
        raise RuntimeError

    if fullOut:
        return U1r, piv1, rows1
    else:
        return U1r, piv1[1,:]