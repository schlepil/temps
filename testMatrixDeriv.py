import sympy as sy
import numpy as np

from sympy.abc import x,y,z
from sympy import sin,cos,tan

from random import random,choice

from copy import deepcopy as dp

from numba import jit,njit

#import testNumba

digits_ = 2


def list2intPy(alist):
    out = 0
    for k,ai in enumerate(reversed(alist)):
        out += ai*10**(digits_*k)
    return out


def int2listPy(aint,len,out=None):
    if out is None:
        out = len*[None]
    
    for k in range(len-1,0-1,-1):
        out[k],aint = divmod(aint,10**(digits_*k))
    
    return out


@njit
def list2intNumba(anArr):
    out = 0
    len = anArr.size
    
    for k in range(len):
        out += anArr[k]*10**(digits_*(len-1-k))
    
    return out


@njit
def int2listNumba(aInt,len,out=None):
    if out is None:
        out = np.empty((len,),dtype=np.int_)
    
    for k in range(len-1,0-1,-1):
        out[k],aint = divmod(aint,10**(digits_*k))
    
    return out


list2int = lambda aIn:list2intNumba(aIn)
int2list = lambda aIn,len,out=None:int2listNumba(aIn,len,out)

listOfDerives = [[0,0,0]]

for _ in range(3):
    listOfDerivesOld = dp(listOfDerives)
    for alist in listOfDerivesOld:
        for i in range(3):
            newList = dp(alist)
            newList[i] += 1
            if not newList in listOfDerives:
                listOfDerives.append(newList)

listOfDerives = list(map(lambda aDeriv:np.array(aDeriv,dtype=np.int_),listOfDerives))

M = sy.Matrix([[sin(x+y),y**2*x,z+y],[y**2*x,cos(x*2*z),sin(y**2)],[z+y,sin(y**2),cos(z)]])

derivDictM = {}

for aDeriv in listOfDerives:
    aDerivInt = list2int(aDeriv)
    derivDictM[aDerivInt] = {'eval':M,'str':"M"}
    
    for (N,var) in zip(aDeriv,[x,y,z]):
        for _ in range(N):
            derivDictM[aDerivInt]['eval'] = sy.diff(derivDictM[aDerivInt]['eval'],var)
            derivDictM[aDerivInt]['str'] = 'd'+str(var)+derivDictM[aDerivInt]['str']
    
    derivDictM[aDerivInt]['evalNP'] = sy.lambdify((x,y,z),derivDictM[aDerivInt]['eval'],modules=[{'ImmutableDenseMatrix':np.matrix},'numpy'])

Mi = M.inv()

derivDictMi = {}

for aDeriv in listOfDerives:
    derivDictMi[aDeriv] = {'eval':Mi,'str':"Mi"}
    
    for N,var in zip(aDeriv,[x,y,z]):
        derivDictMi[aDeriv]['eval'] = sy.diff(derivDictMi[aDeriv]['eval'],var)
        derivDictMi[aDeriv]['str'] = 'd'+str(var)+derivDictMi[aDeriv]['str']
    
    derivDictMi[aDerivInt]['evalNP'] = sy.lambdify((x,y,z),derivDictMi[aDerivInt]['eval'],modules=[{'ImmutableDenseMatrix':np.matrix},'numpy'])

Mf = sy.Function('M',commutative=False)(x,y,z)
Mfi = sy.Function('Mi',commutative=False)(x,y,z)

dxMfe = sy.Function('dxM',commutative=False)
dyMfe = sy.Function('dyM',commutative=False)
dzMfe = sy.Function('dzM',commutative=False)

dydxMfe = sy.Function('dydxM',commutative=False)
dzdxMfe = sy.Function('dzdxM',commutative=False)

dzdydxMfe = sy.Function('dzdydxM',commutative=False)

subsDict1 = dict([(sy.Derivative(Mfi,ax),-(Mfi*sy.diff(Mf,ax)*Mfi)) for ax in [x,y,z]])

subsList2 = ((sy.Derivative(Mf,x,y,z),"dzdydxM"),(sy.Derivative(Mf,x,z,y),"dzdydxM"),(sy.Derivative(Mf,y,x,z),"dzdydxM"),(sy.Derivative(Mf,y,z,x),"dzdydxM"),(sy.Derivative(Mf,z,x,y),"dzdydxM"),(sy.Derivative(Mf,z,y,x),"dzdydxM"),
             (sy.Derivative(Mf,x,y),"dydxM"),(sy.Derivative(Mf,y,x),"dydxM"),
             (sy.Derivative(Mf,x,z),"dzdxM"),(sy.Derivative(Mf,z,x),"dzdxM"),
             (sy.Derivative(Mf,y,z),"dzdxM"),(sy.Derivative(Mf,z,y),"dzdyM"),
             (sy.Derivative(Mf,x),"dxM"),(sy.Derivative(Mf,y),"dyM"),(sy.Derivative(Mf,z),"dzM"),
             (Mfi,"Mi"),(Mf,"M"))

subsList2 = tuple((str(a),b) for (a,b) in subsList2)

dxMfi = sy.diff(Mfi,x)
dxMfi_s = dxMfi.subs(subsDict1)

print(dxMfi_s)
print(sy.simplify(dxMfi_s))

dydxMfi = sy.diff(dxMfi_s,y)
dydxMfi_s = dydxMfi.subs(subsDict1)

print(dydxMfi_s)
print(sy.simplify(dydxMfi_s))

dzdydxMfi = sy.diff(dydxMfi_s,z)
dzdydxMfi_s = dzdydxMfi.subs(subsDict1)

print(dzdydxMfi_s)
print(sy.simplify(dzdydxMfi_s))

dxMfi_se = str(dxMfi_s)
dydxMfi_se = str(dydxMfi_s)
dzdydxMfi_se = str(dzdydxMfi_s)

for keyStr,targStr in subsList2:
    dxMfi_se_old = dp(dxMfi_se)
    dydxMfi_se_old = dp(dydxMfi_se)
    dzdydxMfi_se_old = dp(dzdydxMfi_se)
    dxMfi_se = dxMfi_se.replace(keyStr,targStr)
    dydxMfi_se = dydxMfi_se.replace(keyStr,targStr)
    dzdydxMfi_se = dzdydxMfi_se.replace(keyStr,targStr)
    if dzdydxMfi_se.find("Derivative(M,") >= 0:
        print(keyStr)

print(dxMfi_se)
print(dydxMfi_se)
print(dzdydxMfi_se)

# Compute and compare

for _ in range(10):
    # Draw one
    aDeriv = choice(listOfDerives)
    aDerivInt = list2int(aDeriv)

    # Random input
    xyz = list((np.random.rand(3)-.5)*10.)
    subsDictVal = dict(zip([x,y,z],xyz))
    
    A = np.random.rand(100,100)
    print(testNumba.myFun1(A))
    print("a")

