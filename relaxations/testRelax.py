from coreUtils import *
from polynomial import *
from relaxations import *
from constraints import *

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
    thisPoly = polynomials(thisRepr, coeffs)

    thisPolyCstr = lasserreConstraint(thisRelax, thisPoly)

    print(thisPolyCstr.evalCstr(x))

    
    #print(thisRepr.__dict__)

    return None

def funcTestFirstOpt():
    thisRepr = polynomialRepr(2,4)
    thisRelax = lasserreRelax(thisRepr)
    # x^2+y^2<=1 -> 1.-x^2-y^2>=0.
    #1,x,y,x**2,xy,y**2
    coeffs = nzeros((thisRepr.nMonoms,),dtype=nfloat)
    coeffs[0] = 1.
    coeffs[3] = -1.
    coeffs[5] = -1.
    thisPoly = polynomials(thisRepr,coeffs)
    thisPolyCstr = lasserreConstraint(thisRelax,thisPoly)
    
    #objective 1 : convex
    # obj = (1+x)^2 + (1+y)^2
    coeffs = nzeros((thisRepr.nMonoms,),dtype=nfloat)
    coeffs[0] = 1.
    coeffs[1] = 1.
    obj1 = polynomials(thisRepr, np.copy(coeffs))
    coeffs = nzeros((thisRepr.nMonoms,),dtype=nfloat)
    coeffs[0] = 1.
    coeffs[2] = 1.
    obj2 = polynomials(thisRepr,np.copy(coeffs))
    
    obj = obj1**2+obj2**2
    
    
    



if __name__ == "__main__":
    funcTestBase(2,4)