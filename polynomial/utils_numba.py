from coreUtils import *

@njit(cache=True)
def linChangeNumba(firstIdxPerMonom,cIdx,cpIdx,multiplier,numElem,idxList,c,cp,Aflat):
    nMonoms = firstIdxPerMonom.size
    
    idxListShape0 = idxList.shape[0]
    idxListShape1 = idxList.shape[1]

    # Loop over old coeffs/monoms
    for k in range(nMonoms):
        tmpVal = c[cIdx[k]]
        if tmpVal != 0:
            # old monomial has nonzero coeffs -> take into account
            startIdx = firstIdxPerMonom[k]
            if k+1 == nMonoms:
                # Last monomial loop till end
                stopIdx = idxListShape1
            else:
                # Other -> Loop till nex
                stopIdx = firstIdxPerMonom[k+1]
            
            for j in range(startIdx,stopIdx):
                # Do the actual computation
                tmpVal2 = c[cIdx[k]]*multiplier[j]
                for i in range(numElem[j]):
                    tmpVal2 *= Aflat[idxList[i,j]]
                cp[cpIdx[j]] += tmpVal2
    
    return cp


@njit
def list2int(anArr: np.ndarray,digits: int = 1):
    """
    Computes an integer corresponding to the list of exponents.
    digits determines the maximally occuring degree: digits=1 -> maximal degree = 9
    [2,6,4] -> 264 with digits=1
    [2,6,4] -> 020604 with digits=2

    :param anArr: array representing the exponents
    :type anArr: np.ndarray
    :param digits: Determines the maximal degree
    :type digits: int
    :return: int representing the components
    :rtype: int
    """
    
    out = 0
    multi0 = 1
    multi1 = 10**digits
    
    for k in range(anArr.size-1,-1,-1):
        out += anArr[k]*multi0
        multi0 *= multi1
    
    return out


@njit
def int2list(aInt: int,nVars: int,digits: int = 1,out: np.ndarray = None) -> np.ndarray:
    """
    Inverse operation of list2int
    :param aInt:
    :param nVars:
    :param digits:
    :param out:
    :return:
    """
    if out is None:
        out = np.empty((nVars,),dtype=nint)
    
    for k in range(nVars-1,0-1,-1):
        out[k],aInt = divmod(aInt,10**(digits*k))
    
    return out

@njit
def polyMul(c0,c1,idxMat):
    cp = np.zeros(c0.shape, dtype=nfloat)
    
    #Loop
    for i,ac0 in range(c0.size):
        if ac0 != 0.:
            for j,ac1 in range(c1.size):
                if ac1!=0:
                    cp[idxMat[i,j]] += ac0*ac1
    
    return cp