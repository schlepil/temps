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


def fEmpty():
    pass

def fEmptyNCall0():
    for _ in range(100000):
        pass

def fEmptyNCall1():
    [None for _ in range(100000)]

def fEmptyNCall2():
    for _ in range(100000):
        fEmpty()

def fEmptyNCall3():
    [fEmpty() for _ in range(100000)]