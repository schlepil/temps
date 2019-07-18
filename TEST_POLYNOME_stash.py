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

coeff=np.array([1.,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
coeff2=np.array([1.,-1,0,0,0,0,0,0,0,0,0,0,0,0,0])
coeff3=np.array([1.,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
coeff4=np.array([1.,0,-1,0,0,0,0,0,0,0,0,0,0,0,0])
coeff5=np.array([1.,0,1,0,0,0,0,0,0,0,0,0,0,0,0])

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


c=[]
for i in range(returnL[0].size):
    if np.dot((np.array([[returnL[0,i]],[returnL[1,i]]])-center).T, np.dot(P,(np.array([[returnL[0,i]],[returnL[1,i]]]).reshape((2,1))-center))) >= 1.**2 :
        c.append(i)
    elif np.dot((np.array([[returnL[0,i]],[returnL[1,i]]])-center2).T, np.dot(P2,(np.array([[returnL[0,i]],[returnL[1,i]]]).reshape((2,1))-center2))) >= 1.**2 :
        c.append(i)
    elif np.dot((np.array([[returnL[0,i]],[returnL[1,i]]])-center3).T, np.dot(P3,(np.array([[returnL[0,i]],[returnL[1,i]]]).reshape((2,1))-center3))) >= 1.**2 :
        c.append(i)

returnLnew = np.delete(returnL,c,1)

#print(returnLnew[0].shape)
#print(returnLnew[1].shape)

#print(mycP.repr.evalAllMonoms(returnL).shape)
#print(y.T.shape)
xsol = list(xsol)
xsol[0] = xsol[0][:,[0]]
xsolnew=np.array([[xsol[0][0,0]+random()],[xsol[0][1,0]+random()]])

allCstr=[None for _ in range(len(mycP.constraints.s.cstrList[1:]))]
for i,_ in enumerate(mycP.constraints.s.cstrList[1:]):
    #gi={'type': 'ineq', 'fun': lambda x:float(Cstr.poly.eval2(x.reshape((-1,1))))}
    #print(f"New constraint {i} evaluated to {gi['fun'](xsolnew)}")
    allCstr[i] = {'type': 'ineq', 'fun': lambda x:ndot(deepcopy(mycP.constraints.s.cstrList[copy(i)+1].poly.coeffs),mycP.repr.evalAllMonoms(x))}
    #allCstr[i] = {'type': 'ineq', 'fun': lambda x: float(mycP.constraints.s.cstrList[copy(i) + 1].poly.eval2(x.reshape((-1,1))))}
    print(f"New constraint {i} evaluated to {allCstr[i]['fun'](xsolnew)}; {id(allCstr[i]['fun'])}")

#mycP.constraints.s.cstrList[copy(i)+1].poly.coeffs


while True:
    isValid = True
    xsolnew = xsol[0] + (np.random.rand(*xsol[0].shape) - 0.5)
    verify=[None for _ in range(len(mycP.constraints.s.cstrList[1:]))]
    tiehanhan=[None for _ in range(len(mycP.constraints.s.cstrList[1:]))]
    for i,acstr in enumerate(allCstr):
        print('QAQ', float(acstr['fun'](xsolnew)))
        tiehanhan[i]=float(acstr['fun'](xsolnew))
        verify[i]=(float(acstr['fun'](xsolnew))>0.)
        #isValid = isValid and (float(acstr['fun'](xsolnew))>0.)
        print('aoligei',verify)
        print('tiehanhan',tiehanhan)
    verify=narray(verify)
    
    #if isValid:
    if nall(verify):
        break

for acstr in allCstr:
    assert acstr['fun'](xsolnew)>0.

newsol=ndot(coeff,myrepr.evalAllMonoms(xsolnew))

for i,acstr in enumerate(allCstr):
    print(f"Constraint {i} evaluated to {acstr['fun'](xsolnew)}, {id(acstr['fun'])}")
#gx=lambda x: float(mycP.objective.eval2(x.reshape((-1,1))))
gx=lambda x: ndot(mycP.objective.coeffs,mycP.repr.evalAllMonoms(x))
#y=mycP.objective.eval2(returnLnew)
y=ndot(coeff,mycP.repr.evalAllMonoms(returnLnew))
#bnds = ((-1, 1), (-1, 1))
#cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2}, {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},{'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
res=sp_minimize(gx, xsolnew, method='SLSQP', constraints=allCstr)
ax=Axes3D(fig)
ax.plot_trisurf(returnLnew[0].T,returnLnew[1].T,y)
ax.scatter([xsol[0][0,0]], [xsol[0][1,0]], [sol['primal objective']], color="g", s=100)
ax.scatter([xsolnew[0]], [xsolnew[1]], [newsol], color="r", s=100)
ax.scatter([res.x[0]], [res.x[1]], [gx(res.x)], color="black", s=100)
ax.set_xlabel("x1 label", color="r")
ax.set_ylabel("x2 label", color="g")
ax.set_zlabel("y label", color="b")
for i,acstr in enumerate(allCstr):
    print(f"Constraint {i} evaluated to {acstr['fun'](res.x)}")
    if acstr['fun'](res.x) < 0:
        print("Nope")


plt.show()
