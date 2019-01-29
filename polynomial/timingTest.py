from coreUtils import *
from polynomial import *

from scipy.sparse import csr_matrix

from random import choices

from math import ceil

import subprocess

import time

#
# map1 = np.zeros((thisRepr.listOfMonomialsAsInt.max()+1,),dtype=nint)
#
# for i,j in zip(thisRepr.listOfMonomialsAsInt,thisRepr.varNums):
#     map1[i]=j
#
# map1[0,1]
# map1csr = csr_matrix(map1)
# dict1 = dict(zip(thisRepr.listOfMonomialsAsInt,thisRepr.varNums))
#
#
# map2 = np.zeros((thisRepr2.listOfMonomialsAsInt.max()+1,),dtype=nint)
#
# for i,j in zip(thisRepr2.listOfMonomialsAsInt,thisRepr2.varNums):
#     map2[i]=j
#
# map2csr = csr_matrix(map2)
# dict2 = dict(zip(thisRepr2.listOfMonomialsAsInt,thisRepr2.varNums))
#
#
# map2csr[0,300]
# map2csr._get_single_element(0,300)
#
# idx1 = choices(thisRepr.listOfMonomialsAsInt,k=20)
# idx2 = choices(thisRepr2.listOfMonomialsAsInt,k=20)
#
#
# @njit
# def sparseVecAccesNumba(indices,data,col):
#     lower = 0
#     upper = indices.size-1
#     mid = 0
#
#     while lower <= upper:
#         mid = (lower+upper)//2
#         if indices[mid] < col:
#             lower = mid+1
#         elif indices[mid] > col:
#             upper = mid-1
#         else:
#             break
#
#     return data[mid]
#
#
# def sparseVecAcces(aCSR,col):
#     return sparseVecAccesNumba(aCSR.indices,aCSR.data,col)
#
#
# sparseVecAcces(map1csr,300)
#
# def funCSR0(mat,idx):
#     for aidx in idx:
#         mat[0,aidx]
#     return None
# def funCSR1(mat,idx):
#     for aidx in idx:
#         mat._get_single_element(0,aidx)
#     return None
# def funCSR2(mat,idx):
#     ind = mat.indices
#     data = mat.data
#     for aidx in idx:
#         sparseVecAccesNumba(ind,data,aidx)
#     return None
# def funDict(dict,idx):
#     for aidx in idx:
#         dict[aidx]
#     return None
#
# if __name__ == "__main__":
#     funCSR0(map1csr, idx1)
#     funCSR1(map1csr,idx1)
#     funCSR2(map1csr,idx1)
#     funDict(dict1,idx1)
#
#     subprocess.call("""python3 -m timeit -s "from polynomial import funCSR0, map1csr, idx1" "funCSR0(map1csr, idx1)" """)
#
#
#

def funcEval(nDims, maxDegree):
    thisRepr = polynomialRepr(nDims,maxDegree)
    thisPoly = polynomials(thisRepr, np.random.rand(thisRepr.nMonoms,)-.5)
    
    C = np.random.rand(nDims,nDims)-.5
    C = ndot(C,C.T)+np.identity(nDims, nfloat)*.1
    
    Ci = inv(C)
    
    #Get the lin coord change
    thisPolyTrans = cp(thisPoly)
    thisPolyTrans.coeffs = thisPolyTrans.repr.doLinCoordChange(thisPolyTrans.coeffs, Ci)
    
    
    x = np.random.rand(nDims,)
    y = ndot(C,x.reshape((-1,1)))
    
    print(thisPoly.eval(x))
    print(thisPolyTrans.eval(y.squeeze()))

def getPoly(nDims, maxDegree):
    thisRepr = polynomialRepr(nDims,maxDegree)
    thisPoly = polynomials(thisRepr,np.random.rand(thisRepr.nMonoms,)-.5)

    C = np.random.rand(nDims,nDims)-.5
    C = ndot(C,C.T)+np.identity(nDims,nfloat)*.1
    Ci = inv(C)

    x = np.random.rand(nDims,)
    thisPoly.eval(x)
    thisPoly.repr.doLinCoordChange(np.copy(thisPoly.coeffs),Ci)
    
    return thisPoly

def evalRand(nDims, maxDegree):
    
    thisPoly = getPoly(nDims, maxDegree)
    thisPolyTrans = cp(thisPoly)
    
    for _ in range(100):
    
        C = np.random.rand(nDims,nDims)-.5
        C = ndot(C,C.T)+np.identity(nDims,nfloat)*.1
        Ci = inv(C)
    
        thisPolyTrans.coeffs = thisPolyTrans.repr.doLinCoordChange(thisPoly.coeffs,Ci)

        x = np.random.rand(nDims,)
        y = ndot(C,x.reshape((-1,1)))
        if __debug__:
            print(thisPoly.eval(x)-thisPolyTrans.eval(y.squeeze()))
        else:
            assert abs(thisPoly.eval(x)-thisPolyTrans.eval(y.squeeze())) < 1e-10
    
    return None
    
    
    

if __name__ == "__main__":
    T = time.time()
    evalRand(2,2)
    print("2x2 time ", time.time()-T)
    T = time.time()
    evalRand(6,4)
    print("6x4 time ",time.time()-T)
    #funcEval(2,2)
    #funcEval(6,4)