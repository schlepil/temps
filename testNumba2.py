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

@njit
def funca(anArray):
    tmp=0.
    for k in range(anArray.size):
        tmp += (float(k)+anArray[k])
    return tmp

@njit
def funcb(anArray):
    tmp=0.
    for k,val in enumerate(anArray):
        tmp += (float(k)+val)
    return tmp