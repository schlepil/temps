from numba import jit, njit
import numpy as np

def matmul0(A,B):
    return A*B*B*A*B*B*B*A
@jit
def matmul1(A,B):
    return A*B*B*A*B*B*B*A
@njit
def matmul2(A,B):
    return A*B*B*A*B*B*B*A