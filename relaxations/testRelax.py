from coreUtils import *
from polynomial import *
from relaxations import *

def funcTestBase(nDims, maxDeg):
    thisRepr = polynomialRepr(nDims, maxDeg)
    thisRelax = lasserreRelax(thisRepr)

    x = np.random.rand(nDims).astype(nfloat)
    print(thisRelax.evalCstr(x))

    print("\n\n")
    print(thisRepr.listOfMonomialsAsInt)
    coeffs = nzeros((thisRepr.nMonoms,),dtype=nfloat)
    coeffs[0] = 1.1
    coeffs[1] = 2.2
    thisPoly = polynomial(thisRepr, coeffs)

    thisPolyCstr = lasserreConstraint(thisRelax, thisPoly)

    print(thisPolyCstr.evalCstr(x))

    
    #print(thisRepr.__dict__)

    return None

def funcTestFirstOpt(relaxOrder=4):
    thisRepr = polynomialRepr(2,relaxOrder)
    thisRelax = lasserreRelax(thisRepr)
    # x^2+y^2<=1 -> 1.-x^2-y^2>=0.
    #1,x,y,x**2,xy,y**2
    coeffs = nzeros((thisRepr.nMonoms,),dtype=nfloat)
    coeffs[0] = 1.
    coeffs[3] = -1.
    coeffs[5] = -1.
    thisPoly = polynomial(thisRepr, coeffs)
    thisPolyCstr = lasserreConstraint(thisRelax,thisPoly)
    
    #objective 1 : convex
    # obj = (1+x)^2 + (1+y)^2
    coeffs = nzeros((thisRepr.nMonoms,),dtype=nfloat)
    coeffs[0] = 1.
    coeffs[1] = 1.
    obj1 = polynomial(thisRepr, np.copy(coeffs))
    coeffs = nzeros((thisRepr.nMonoms,),dtype=nfloat)
    coeffs[0] = 1.
    coeffs[2] = 1.
    obj2 = polynomial(thisRepr, np.copy(coeffs))
    
    obj = obj1**2+obj2**2
    
    thisOpt = convexProg(thisRepr, objective=obj)
    
    thisOpt.addCstr(thisRelax)
    sol = thisOpt.solve()
    print("Analytic solution is 0. obtained {0}".format(sol['obj']))
    print("Analytic minimizer is (-1,-1) obtained {0}".format(sol['x_np'][0,1:3]))
    print("relax mat is \n {0}".format(thisRelax.evalCstr(sol['x_np'])))
    print("With eigvals \n {0} \n and eigvec \n{1}".format(*eigh(thisRelax.evalCstr(sol['x_np']))))
    print("Delta constraints is \n {0}".format(thisRelax.evalCstr(sol['x_np'])-thisRelax.evalCstr(narray([-1.,-1.],dtype=nfloat))) )
    print("With eigvals \n {0} \n and eigvec \n {1}".format(*eigh(thisRelax.evalCstr(sol['x_np'])-thisRelax.evalCstr(narray([-1.,-1.],dtype=nfloat)))))
    
    thisOpt.addCstr(thisPolyCstr)
    sol = thisOpt.solve()
    Osq2 = -1./2.**0.5
    print("Analytic solution is 0.17157287525381 obtained {0}".format(sol['primal objective']))
    print("Analytic minimizer is (-1/sq2,-1/sq2) obtained {0}".format(sol['x_np'][0,1:3]))
    print("relax mat is \n {0}".format(thisRelax.evalCstr(sol['x_np'])))
    print("With eigvals \n {0} \n and eigvec \n{1}".format(*eigh(thisRelax.evalCstr(sol['x_np']))))
    print("Delta constraints is \n {0}".format(thisRelax.evalCstr(sol['x_np'])-thisRelax.evalCstr(narray([Osq2,Osq2],dtype=nfloat))))
    print("With eigvals \n {0} \n and eigvec \n {1}".format(*eigh(thisRelax.evalCstr(sol['x_np'])-thisRelax.evalCstr(narray([Osq2,Osq2],dtype=nfloat)))))
    
    obj2 = -obj
    thisOpt.removeCstr('s', 1)
    thisOpt.objective = obj2
    sol = thisOpt.solve()
    print("Solution should be unbounded")
    print(sol)

    thisOpt.addCstr(thisPolyCstr)
    sol = thisOpt.solve()
    Osq2 = 1./2.**0.5
    print("Analytic solution is -5.82842712474619 obtained {0}".format(sol['primal objective']))
    print("Analytic minimizer is (1/sq2,1/sq2) obtained {0}".format(sol['x_np'][0,1:3]))
    print("relax mat is \n {0}".format(thisRelax.evalCstr(sol['x_np'])))
    print("With eigvals \n {0} \n and eigvec \n{1}".format(*eigh(thisRelax.evalCstr(sol['x_np']))))
    print("Delta constraints is \n {0}".format(thisRelax.evalCstr(sol['x_np'])-thisRelax.evalCstr(narray([Osq2,Osq2],dtype=nfloat))))
    print("With eigvals \n {0} \n and eigvec \n {1}".format(*eigh(thisRelax.evalCstr(sol['x_np'])-thisRelax.evalCstr(narray([Osq2,Osq2],dtype=nfloat)))))


def funcTestFirstOpt2(relaxOrder=4):
    thisRepr = polynomialRepr(2, relaxOrder)
    thisRelax = lasserreRelax(thisRepr)
    # x^2+y^2<=1 -> 1.-x^2-y^2>=0.
    # 1,x,y,x**2,xy,y**2
    coeffs = nzeros((thisRepr.nMonoms,), dtype=nfloat)
    coeffs[0] = 1.
    coeffs[3] = -1.
    coeffs[5] = -1.
    thisPoly = polynomial(thisRepr, coeffs)
    thisPolyCstr = lasserreConstraint(thisRelax, thisPoly)

    # objective 1 : convex
    # obj = (1+x)^2 + (1+y)^2
    coeffs = nzeros((thisRepr.nMonoms,), dtype=nfloat)
    coeffs[0] = 1.
    coeffs[1] = 1.
    obj1 = polynomial(thisRepr, np.copy(coeffs))
    coeffs = nzeros((thisRepr.nMonoms,), dtype=nfloat)
    coeffs[0] = 1.
    coeffs[2] = 1.
    obj2 = polynomial(thisRepr, np.copy(coeffs))
    
    obj = obj1**2+obj2**2
    
    thisOpt = convexProg(thisRepr, objective=obj)
    
    thisOpt.addCstr(thisRelax)
    sol = thisOpt.solve()
    print("Analytic solution is 0. obtained {0}".format(sol['primal objective']))
    print("Analytic minimizer is (-1,-1) obtained {0}".format(sol['x_np'][0, 1:3]))
    print("relax mat is \n {0}".format(thisRelax.evalCstr(sol['x_np'])))
    print("With eigvals \n {0} \n and eigvec \n{1}".format(*eigh(thisRelax.evalCstr(sol['x_np']))))
    print("Delta constraints is \n {0}".format(thisRelax.evalCstr(sol['x_np'])-thisRelax.evalCstr(narray([-1., -1.], dtype=nfloat))))
    print("With eigvals \n {0} \n and eigvec \n {1}".format(*eigh(thisRelax.evalCstr(sol['x_np'])-thisRelax.evalCstr(narray([-1., -1.], dtype=nfloat)))))
    
    thisOpt.addCstr(thisPolyCstr)
    sol = thisOpt.solve()
    Osq2 = -1./2.**0.5
    print("Analytic solution is 0.17157287525381 obtained {0}".format(sol['primal objective']))
    print("Analytic minimizer is (-1/sq2,-1/sq2) obtained {0}".format(sol['x_np'][0, 1:3]))
    print("relax mat is \n {0}".format(thisRelax.evalCstr(sol['x_np'])))
    print("With eigvals \n {0} \n and eigvec \n{1}".format(*eigh(thisRelax.evalCstr(sol['x_np']))))
    print("Delta constraints is \n {0}".format(thisRelax.evalCstr(sol['x_np'])-thisRelax.evalCstr(narray([Osq2, Osq2], dtype=nfloat))))
    print("With eigvals \n {0} \n and eigvec \n {1}".format(*eigh(thisRelax.evalCstr(sol['x_np'])-thisRelax.evalCstr(narray([Osq2, Osq2], dtype=nfloat)))))
    
    obj2 = -obj
    thisOpt.removeCstr('s', 1)
    thisOpt.objective = obj2
    sol = thisOpt.solve()
    print("Solution should be unbounded")
    print(sol)
    
    thisOpt.addCstr(thisPolyCstr)
    sol = thisOpt.solve()
    Osq2 = 1./2.**0.5
    print("Analytic solution is -5.82842712474619 obtained {0}".format(sol['primal objective']))
    print("Analytic minimizer is (1/sq2,1/sq2) obtained {0}".format(sol['x_np'][0, 1:3]))
    print("relax mat is \n {0}".format(thisRelax.evalCstr(sol['x_np'])))
    print("With eigvals \n {0} \n and eigvec \n{1}".format(*eigh(thisRelax.evalCstr(sol['x_np']))))
    print("Delta constraints is \n {0}".format(thisRelax.evalCstr(sol['x_np'])-thisRelax.evalCstr(narray([Osq2, Osq2], dtype=nfloat))))
    print("With eigvals \n {0} \n and eigvec \n {1}".format(*eigh(thisRelax.evalCstr(sol['x_np'])-thisRelax.evalCstr(narray([Osq2, Osq2], dtype=nfloat)))))


if __name__ == "__main__":
    funcTestBase(2,4)
    funcTestFirstOpt(4)
    funcTestFirstOpt(6)
    funcTestFirstOpt(8)