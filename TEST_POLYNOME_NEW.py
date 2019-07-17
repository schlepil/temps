import numpy as np
from scipy.sparse import coo_matrix
import polynomial as poly
import plotting as plot
from random import random
import relaxations as relax
from relaxations.lasserre import *
from relaxations.constraints import *
from relaxations.optUtils import *
from relaxations.rref import robustRREF
from scipy.optimize import minimize as sp_minimize
from scipy.optimize import NonlinearConstraint
from copy import copy, deepcopy

coeff=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
coeff2=np.array([1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0])
coeff3=np.array([1.,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
coeff4=np.array([1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0])
coeff5=np.array([1,0,1,0,0,0,0,0,0,0,0,0,0,0,0])

#coeff *= (np.random.rand(*coeff.shape)-0.5)

#contraint.getCstr(isSparse=True)
#print(contraint.getCstr(isSparse=True))
#Creation d'une "representation"
myrepr = poly.polynomialRepr(2,4)
# Creation d'un polynome a partir de la represetation
mypoly = poly.polynomial(myrepr,coeff)
mypoly2 = poly.polynomial(myrepr,coeff2)
mypoly3 = poly.polynomial(myrepr,coeff3)
mypoly4=poly.polynomial(myrepr,coeff4)
mypoly5=poly.polynomial(myrepr,coeff5)


#les ellips:
poly_pour_ellips=poly.polynomial(myrepr)
center = 2. * (np.random.rand(2, 1) - .5)
P = 1.5 * (np.random.rand(2, 2) - .5)
P = ndot(P.T, P) + .5 * nidentity(2)
poly_pour_ellips.setEllipsoidalConstraint(center, 1., P)

poly_pour_ellips2=poly.polynomial(myrepr)
center2 = 2. * (np.random.rand(2, 1) - .5)
P2 = 1.5 * (np.random.rand(2, 2) - .5)
P2 = ndot(P2.T, P2) + .5 * nidentity(2)
poly_pour_ellips2.setEllipsoidalConstraint(center2, 1., P2)

poly_pour_ellips3=poly.polynomial(myrepr)
center3 = 2. * (np.random.rand(2, 1) - .5)
P3 = 1.5 * (np.random.rand(2, 2) - .5)
P3 = ndot(P3.T, P3) + .5 * nidentity(2)
poly_pour_ellips3.setEllipsoidalConstraint(center3, 1., P3)


Relax=lasserreRelax(repr=myrepr)
contraint=lasserreConstraint(poly=mypoly2,baseRelax=Relax)#,nRelax=2)
contraint2=lasserreConstraint(poly=mypoly3,baseRelax=Relax)#,nRelax=2)
contraint3=lasserreConstraint(poly=mypoly4,baseRelax=Relax)#,nRelax=2)
contraint4=lasserreConstraint(poly=mypoly5,baseRelax=Relax)#,nRelax=2)
contraint5=lasserreConstraint(poly=poly_pour_ellips,baseRelax=Relax)
contraint6=lasserreConstraint(poly=poly_pour_ellips2,baseRelax=Relax)
contraint7=lasserreConstraint(poly=poly_pour_ellips3,baseRelax=Relax)
from relaxations import convexProg as cP
mycP=cP(repr= myrepr, objective=mypoly)
mycP.addCstr(Relax)
mycP.addCstr(contraint)
mycP.addCstr(contraint2)
mycP.addCstr(contraint3)
mycP.addCstr(contraint4)
mycP.addCstr(contraint5)
mycP.addCstr(contraint6)
mycP.addCstr(contraint7)
#print(mycP.constraints.s.cstrList[-1].getCstr())
sol=mycP.solve_cvxopt()

#print('sol',sol)
mycP.checkSol(sol=sol)
xsol=mycP.extractOptSol(sol=sol)
print("extractOptSol",xsol[0])
print("extractOptSol",xsol[0][1,0])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

x1,x2 = np.meshgrid(np.linspace(-1,1, 100), np.linspace(-1, 1,100))

#returnL = [x1,x2]
returnL=np.vstack((x1.flatten(), x2.flatten()))

returnLZ = myrepr.evalAllMonoms(returnL)
isValid = nones(returnL.shape[1],).astype(np.bool_)

for acstr in mycP.constraints.s.cstrList[1:]:
    isValid = np.logical_and(isValid, acstr.poly.eval2(returnLZ).squeeze()>=0.)

returnLnew = returnL[:, isValid]
returnLZnew = returnLZ[:, isValid]

#print(returnLnew[0].shape)
#print(returnLnew[1].shape)

#print(mycP.repr.evalAllMonoms(returnL).shape)
#print(y.T.shape)
xsol = list(xsol)
xsol[0] = xsol[0][:,[0]]
xsolnew=np.array([[xsol[0][0,0]+random()],[xsol[0][1,0]+random()]])

Amat = nzeros((len(mycP.constraints.s.cstrList[1:]), coeff.size), dtype=nfloat)

for i,acstr in enumerate(mycP.constraints.s.cstrList[1:]):
    Amat[i,:] = acstr.poly.coeffs.copy()

this_cstr = {'type': 'ineq', 'fun': lambda x:ndot(Amat,mycP.repr.evalAllMonoms(x.reshape((-1,1)))).squeeze()}


while True:
    isValid = True
    xsolnew = xsol[0] + (np.random.rand(*xsol[0].shape) - 0.5)

    if nall(this_cstr['fun'](xsolnew)>=-0.25):
        break

newsol=ndot(coeff,myrepr.evalAllMonoms(xsolnew))

#gx=lambda x: float(mycP.objective.eval2(x.reshape((-1,1))))
gx=lambda x: ndot(mycP.objective.coeffs,mycP.repr.evalAllMonoms(x))
#y=mycP.objective.eval2(returnLnew)
y=ndot(coeff,returnLZnew)
#bnds = ((-1, 1), (-1, 1))
#cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2}, {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},{'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
res=sp_minimize(gx, xsolnew, method='COBYLA', constraints=this_cstr)
ax=Axes3D(fig)
ax.plot_trisurf(returnLnew[0].T,returnLnew[1].T,y)
ax.scatter([xsol[0][0,0]], [xsol[0][1,0]], [sol['primal objective']], color="g", s=100)
ax.scatter([xsolnew[0]], [xsolnew[1]], [newsol], color="r", s=100)
ax.scatter([res.x[0]], [res.x[1]], [gx(res.x)], color="black", s=100, marker='x')
ax.set_xlabel("x1 label", color="r")
ax.set_ylabel("x2 label", color="g")
ax.set_zlabel("y label", color="b")

print(this_cstr['fun'](xsolnew))
print(this_cstr['fun'](res.x))


plt.show()
