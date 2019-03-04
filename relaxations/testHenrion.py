#https://homepages.laas.fr/henrion/Papers/extract.pdf

from relaxations import *

thisRepr = polynomialRepr(3,4)

c0 = nzeros((thisRepr.nMonoms,),nfloat); c0[0] = -1; c0[1] = 1.; polyObj0 = polynomial(thisRepr, c0); polyObj0 *= polyObj0
c1 = nzeros((thisRepr.nMonoms,),nfloat); c1[0] = -1; c1[2] = 1.; polyObj1 = polynomial(thisRepr, c1); polyObj1 *= polyObj1
c2 = nzeros((thisRepr.nMonoms,),nfloat); c2[0] = -1; c2[3] = 1.; polyObj2 = polynomial(thisRepr, c2); polyObj2 *= polyObj2

polyObj = -1.*(polyObj0+polyObj1+polyObj2)

cstr0 = nzeros((thisRepr.nMonoms,),nfloat); cstr0[0] = 1; polyObjCstr0 = polynomial(thisRepr, cstr0)
polyCstr0 = polyObjCstr0-polyObj0
polyCstr1 = polyObjCstr0-polyObj1
polyCstr2 = polyObjCstr0-polyObj2

baseRelax = lasserreRelax(thisRepr)
cstr0 = lasserreConstraint(baseRelax, polyCstr0)
cstr1 = lasserreConstraint(baseRelax, polyCstr1)
cstr2 = lasserreConstraint(baseRelax, polyCstr2)

cvxProb = convexProg(thisRepr, objective=polyObj)

cvxProb.addCstr(baseRelax)
cvxProb.addCstr(cstr0)
cvxProb.addCstr(cstr1)
cvxProb.addCstr(cstr2)

sol = cvxProb.solve()
xSol = cvxProb.extractOptSol(sol)

print(sol)
print(xSol)
