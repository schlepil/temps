from coreUtils import *
from polynomial import *
from relaxations import *

def funcTestBase(nDims, maxDeg):
    thisRepr = polynomialRepr(nDims, maxDeg)
    lasserreRelax(thisRepr)
    
    #print(thisRepr.__dict__)

    return None


if __name__ == "__main__":
    funcTestBase(2,4)