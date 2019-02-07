from coreUtils import *

from polynomial import polynomialRepr

from polynomial.utils_numba import getIdxAndParent

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


def compTaylorExp(f:sy.Matrix, fStr:str, q:sy.Symbol, isMat:bool, maxTaylorDeg:int, repr:polynomialRepr)->dict:
    """
    Computes the Taylor expansion of a given formula
    :param f:
    :param fStr:
    :param q:
    :param maxTaylorDeg:
    :param repr:
    :return:
    """
    if __debug__:
        assert repr.nDims == q.shape[0]*q.shape[1]

    #Definitions
    # array2mat = [{'ImmutableDenseMatrix':np.matrix},'numpy']
    array2mat = [{'ImmutableDenseMatrix': np.array}, 'numpy']
    # array2mat = ['numpy']


    n,m = f.shape

    if not isMat:
        if __debug__:
            assert m==1, 'only vectors allowed'
        #Vector case
        taylorFD = {fStr+'0': sy.Matrix(nzeros((n, 1)))}

        taylorFD[fStr+'0'] += f

        lastMat = f
        for k in range(1, maxTaylorDeg + 1):
            thisMat = sy.Matrix(nzeros((n, len(repr.listOfMonomialsPerDeg[k]))))

            # Get the derivative corresponding to each monomial
            for j, aMonom in enumerate(repr.listOfMonomialsPerDegAsInt[k]):
                idxDeriv, idxParent = getIdxAndParent(aMonom, repr.listOfMonomialsPerDegAsInt[k - 1], repr.nDims, repr.digits)

                # Derive the (sliced) vector
                thisMat[:, j] = sy.diff(lastMat[:, idxParent], q[idxDeriv, 0]) / float(k)

            # save
            taylorFD["{1:s}{0:d}".format(k, fStr)] = thisMat
            # Iterate
            lastMat = thisMat

        # for all 0-N
        totCols = sum([len(aList) for aList in repr.listOfMonomialsPerDeg[:maxTaylorDeg + 1]])
        taylorFD[fStr+"Taylor"] = sy.Matrix(nzeros((n, totCols)))
        cCols = 0
        for k in range(0, maxTaylorDeg + 1):
            aKey = "{1}{0:d}".format(k,fStr)
            aVal = taylorFD[aKey]

            taylorFD[fStr+"Taylor"][:, cCols:cCols + aVal.shape[1]] = aVal
            cCols += aVal.shape[1]
        
        #Create each
        for k in range(0,maxTaylorDeg+1):
            totCols = sum([len(aList) for aList in repr.listOfMonomialsPerDeg[:k+1]])
            taylorFD[fStr+"Taylor_to_{0:d".format(k)] = sy.Matrix(nzeros((n,totCols)))
            
            

        tempDict = dict()

        for aKey, aVal in taylorFD.items():
            tempDict[aKey + "_eval"] = sy.lambdify(q, aVal, modules=array2mat)

        taylorFD.update(tempDict)
    else:
        taylorFD = {fStr+'0': [sy.Matrix(nzeros((n, m)))]}

        taylorFD[fStr+'0'][0] += f

        lastList = [f]

        for k in range(1, maxTaylorDeg + 1):
            thisList = [sy.Matrix(nzeros((n, m))) for _ in
                        range(repr.listOfMonomialsPerDegAsInt[k].size)]

            # Get the derivative corresponding to each monomial
            for j, aMonom in enumerate(repr.listOfMonomialsPerDegAsInt[k]):
                idxDeriv, idxParent = getIdxAndParent(aMonom, repr.listOfMonomialsPerDegAsInt[k - 1], repr.nDims, repr.digits)

                thisList[j] = sy.diff(lastList[idxParent], q[idxDeriv, 0]) / float(k)

            # save
            taylorFD["{1}{0:d}".format(k, fStr)] = thisList
            # Iterate
            lastList = thisList

        taylorFD[fStr+"Taylor"] = []
        for k in range(0, maxTaylorDeg + 1):
            aKey = "{1}{0:d}".format(k, fStr)
            aVal = taylorFD[aKey]

            taylorFD[fStr+"Taylor"].extend(aVal)

        # array2mat = [{'ImmutableDenseMatrix': np.matrix}, 'numpy']
        array2mat = [{'ImmutableDenseMatrix': np.array}, 'numpy']
        # array2mat = ['numpy']

        tempDict = {}
        for aKey, aVal in taylorFD.items():
            tempDict[aKey + "_eval"] = [sy.lambdify(q, aMat, modules=array2mat) for aMat in aVal]
        taylorFD.update(tempDict)

    return taylorFD


