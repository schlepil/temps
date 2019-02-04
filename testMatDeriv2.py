import sympy as sy
import numpy as np

from sympy.abc import x,y,z
from sympy import sin,cos,tan

from random import random,choice

from copy import deepcopy as dp


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


