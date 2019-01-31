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
    thisPoly = polynomials(thisRepr, coeffs)

    thisPolyCstr = lasserreConstraint(thisRelax, thisPoly)

    print(thisPolyCstr.evalCstr(x))

    
    #print(thisRepr.__dict__)

    return None


if __name__ == "__main__":
    funcTestBase(2,4)