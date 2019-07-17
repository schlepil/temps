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


@njit(cache=True)
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


@njit(cache=True)
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


# idxParent, idxVar = getIdxAndParent(aMonom, self.repr.listOfMonomialsPerDeg[k-1], digits)
@njit(cache=True)
def getIdxAndParent(aMonom: int,aMonomList: narray,nVars: int,digits: int):


    # TODO change to additionally take varnum2varnumParents
#    if __debug__: #numba .43
#        assert isinstance(aMonom,(int, nint, nintu))
#        assert isinstance(aMonomList,np.ndarray)
    
    found = False
    
    deltaMonom = nzeros((nVars,),dtype=nintu)
    for idxVar in range(nVars):
        deltaMonom[:] = 0
        deltaMonom[idxVar] = 1
        deltaMonomAsInt = list2int(deltaMonom,digits=digits)
        oldMonomAsInt = aMonom-deltaMonomAsInt
        
        for idxParent,aParent in enumerate(aMonomList):
            if aParent == oldMonomAsInt:
                found = True
                break
        if found:
            break
    
    return idxVar, idxParent
    
@njit(cache=True)
def polyMul(c0,c1,idxMat):
    cp = np.zeros(c0.shape, dtype=nfloat)
    
    #Loop
    for i,ac0 in enumerate(c0):
        if ac0 != 0.:
            for j,ac1 in enumerate(c1):
                if ac1!=0:
                    cp[idxMat[i,j]] += ac0*ac1
    
    return cp


@njit(cache=True)
def polyMulExp(c0,c1,cout,idxMat,idxMax0,idxMax1):
    
    # Loop
    for i,ac0 in enumerate(c0[:idxMax0]):
        if ac0 != 0.:
            for j,ac1 in enumerate(c1[idxMax1]):
                if ac1 != 0:
                    cout[idxMat[i,j]] += ac0*ac1

    return cp

# self.coeffs = quadraticForm_Numba(Q,qMonoms, h, hMonom, self.repr.idxMat, np.zeros((self.repr.nMonoms,), dtype=nfloat))
@njit(cache=True)
def quadraticForm_Numba(Q,qMonoms, h, hMonom, idxMat, coeffsOut):
    """
    Here no special form is assumed for Q
    :param Q:
    :param qMonoms:
    :param h:
    :param hMonom:
    :param idxMat:
    :param coeffsOut:
    :return:
    """
    #print('hMonom', hMonom)
    #print('h', h)
    for aHMonom, aH in zip(hMonom, h):
        #print('aHMonom',aHMonom)
        #print('aH', aH)
        coeffsOut[aHMonom] += aH
        #print('coeffsOut[aHMonom] += aH',coeffsOut)
    #print('qMonoms',qMonoms)
    #print('Q',Q)
    #print('idxMat', idxMat)
    for i,imonom in enumerate(qMonoms):
        for j, jmonom in enumerate(qMonoms):
           # print('i',i)
            #print('j',j)
            coeffsOut[idxMat[imonom,jmonom]] += Q[i,j]
            #print('idxMat[imonom, jmonom',idxMat[imonom, jmonom])
            #print('coeffsOut[idxMat[imonom,jmonom]] += Q[i,j]', coeffsOut)
    return coeffsOut


@njit(cache=True)
def evalMonomsNumba1(x, varNum2varNumParents):

    z = nempty((varNum2varNumParents.shape[0],1))

    #Init
    z[0,0] = 1.
    z[1:1+x.shape[0],:] = x
    #Loop
    #for k, (p0, p1) in enumerate(varNum2varNumParents):
    for k in range(varNum2varNumParents.shape[0]):
        p0 = varNum2varNumParents[k, 0]
        p1 = varNum2varNumParents[k, 1]
        if k < (+1 + x.shape[0]):
            continue
        z[k,0] = z[p0,0]*z[p1,0]
    return z

@njit(cache=True)
def evalMonomsNumbaN(x, varNum2varNumParents):

    z = nempty((varNum2varNumParents.shape[0],x.shape[1]))

    #Init
    z[0,:] = 1.
    z[1:1+x.shape[0],:] = x
    #Loop
    #for k, (p0,p1) in enumerate(varNum2varNumParents):
    for k in range(varNum2varNumParents.shape[0]):
        p0=varNum2varNumParents[k,0]
        p1=varNum2varNumParents[k,1]
        if k<(+1+x.shape[0]):
            continue
        z[k,:] = z[p0,:]*z[p1,:]
    return z

#@njit
# def evalMonomsNumba(x, varNum2varNumParents):
#     if len(x.shape)>1:
#       print('its 1')
#       if (x.shape[1]==1) :
#          z =  evalMonomsNumba1(x, varNum2varNumParents)
#       else:
#          z = evalMonomsNumbaN(x, varNum2varNumParents)
#     else:
#         print('its 2')
#         x = x.reshape(-1, 1)
#         print('its 2',x.shape)
#         z = evalMonomsNumba1(x, varNum2varNumParents)
#     return z
def evalMonomsNumba(x, varNum2varNumParents):
      if (x.shape[1]==1) :
         z =  evalMonomsNumba1(x, varNum2varNumParents)
      else:
         z = evalMonomsNumbaN(x, varNum2varNumParents)
      return z
