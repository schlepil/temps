from coreUtils import *

from itertools import permutations, combinations

def getInverseTaylorStrings(Mstr='M',Mistr='Mi',fstr='f',derivStrings=['x','y','z'], doCompile:bool=False):
    """
    Compute strings corresponding to the Taylor approximation of
    Mi*f
    So
    d/dx [Mi*f] = d/dx [Mi]*f + Mi*d/dx[f] = -Mi*d/dx[M]*Mi*f + Mi*d/dx[f]
    :param Mstr:
    :param Mistr:
    :param fstr:
    :param derivStrings:
    :return:
    """
    
    nDerivs = len(derivStrings)
    derivSyms = [sy.symbols(aStr) for aStr in derivStrings]
    
    M = sy.Function(Mstr,commutative=False)(*derivSyms)
    Mi = sy.Function(Mistr,commutative=False)(*derivSyms)
    f = sy.Function(fstr,commutative=False)(*derivSyms)
    
    subsDict = dict([(sy.Derivative(Mi,ax),-(Mi*sy.diff(M,ax)*Mi)) for ax in derivSyms])
    
    subsList = [(str(M),Mstr),(str(Mi),Mistr),(str(f),fstr)]
    #Succesively append all derivatives
    idxL = list(range(nDerivs))
    allDerivs = []
    allDerivStr = []
    for nDeriv in range(1,nDerivs+1):
        thisDerivs = list(combinations(idxL,nDeriv))
        allDerivs.extend(thisDerivs)
        for aDeriv in thisDerivs:
            thisStr = "".join(derivStrings[aIdx] for aIdx in reversed(aDeriv))
            allDerivStr.append(thisStr)
            for aDerivPermut in permutations(aDeriv, nDeriv):
                thisDList = [derivSyms[aIdx] for aIdx in aDerivPermut]
                subsList.append( (str(sy.Derivative(M,*thisDList)), thisStr+Mstr) )
                subsList.append( (str(sy.Derivative(f,*thisDList)), thisStr+fstr) )
    
    subsList.reverse() #Replace complex terms first
    
    allDerivsList = [Mi*f]
    for aSym in derivSyms:
        allDerivsList.append( sy.diff(allDerivsList[-1],aSym) )
        allDerivsList[-1] = allDerivsList[-1].subs(subsDict)

    allDerivsListStr = []
    for aDeriv in allDerivsList:
        allDerivsListStr.append(str(aDeriv))
        for keyStr,targStr in subsList:
            allDerivsListStr[-1] = allDerivsListStr[-1].replace(keyStr,targStr)
        if doCompile:
            allDerivsListStr[-1] = compile(allDerivsListStr[-1], 'string', 'eval')
    
    # TODO take into account that M and Mi are symmetric, use np multidot
    
    if __debug__:
        print(allDerivsListStr)
    
    return {'funcstr':allDerivsListStr, 'derivStr':derivStrings, 'allDerivs':dict(zip(allDerivStr,allDerivs))}