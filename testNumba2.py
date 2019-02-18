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

@njit
def funAcc0(A):
    for i in range(A.shape[0]):
            A[i]*=1.35641687
    return A

@njit
def funAcc(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i,j]
    return A

@njit
def funAcc1(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j]*=1.35641687
    return A

@njit
def funAcc2(A):
    for i in range(A.shape[0]):
            A[i, :]
    return A

@njit
def funAcc3(A):
    for i in range(A.shape[0]):
            A[i, :]*=1.356416876
    #I do not know why but this is insanely much faster then funAcc1 for A.shape[1] large and almmost as fast for A.shape[1] small
    return A


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