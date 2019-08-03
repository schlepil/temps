from coreUtils import *

from polynomial import polynomialRepr

from polynomial.utils_numba import getIdxAndParent

from itertools import permutations, combinations

try:
    from scipy.misc import factorial
except ImportError:
    from scipy.special import factorial


def getTaylorWeights(allMonom:List):
    return 1./narray( [nprod(factorial(aMonom)) for aMonom in allMonom] , dtype=nfloat)


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


def compParDerivs(f:sy.Matrix, fStr:str, q:sy.Symbol, isMat:bool, maxPDerivDeg:int, repr:polynomialRepr)->dict:
    """
    Computes the partial derivatives needed for taylor expansion of a given formula
    :param f:
    :param fStr:
    :param q:
    :param maxPDerivDeg:
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

    #TODO
    # Check if derive_by_array is better

    if not isMat:
        if __debug__:
            assert m==1, 'only vectors allowed'
        #Vector case
        pDerivFD = {f"{fStr}0": sy.Matrix(nzeros((n, 1)))}

        pDerivFD[f"{fStr}0"] += f

        lastMat = f
        for k in range(1, maxPDerivDeg + 1):
            thisMat = sy.Matrix(nzeros((n, len(repr.listOfMonomialsPerDeg[k]))))

            # Get the derivative corresponding to each monomial
            for j, aMonom in enumerate(repr.listOfMonomialsPerDegAsInt[k]):
                idxDeriv, idxParent = getIdxAndParent(aMonom, repr.listOfMonomialsPerDegAsInt[k - 1], repr.nDims, repr.digits)

                # Derive the (sliced) vector
                thisMat[:, j] = sy.diff(lastMat[:, idxParent], q[idxDeriv, 0])

            # save
            pDerivFD[f"{fStr:s}{k:d}"] = thisMat
            # Iterate
            lastMat = thisMat

        # for all 0-N
        totCols = sum([len(aList) for aList in repr.listOfMonomialsPerDeg[:maxPDerivDeg + 1]])
        pDerivFD[f"{fStr}PDeriv"] = sy.Matrix(nzeros((n, totCols)))
        cCols = 0
        for k in range(0, maxPDerivDeg + 1):
            aKey = f"{fStr}{k:d}"
            aVal = pDerivFD[aKey]

            pDerivFD[f"{fStr}PDeriv"][:, cCols:cCols + aVal.shape[1]] = aVal
            cCols += aVal.shape[1]

        #Create each taylor expansion of up to degree k
        totCols = 0
        for k in range(0,maxPDerivDeg+1):
            totCols += len( repr.listOfMonomialsPerDeg[k])
            pDerivFD[f"{fStr}PDeriv_to_{k:d}"] = pDerivFD[f"{fStr}PDeriv"][:,:totCols] #Matrix

        # Lambdify all expression
        # TODO check if creating a numba/cython file is more efficient
        tempDict = dict()
        for aKey, aVal in pDerivFD.items():
            tempDict[f"{aKey}_eval"] = sy.lambdify(q, aVal, modules=array2mat)

        pDerivFD.update(tempDict)
    else:
        pDerivFD = {f"{fStr}0": [sy.Matrix(nzeros((n, m)))]}

        pDerivFD[f"{fStr}0"][0] += f

        lastList = [f]

        for k in range(1, maxPDerivDeg + 1):
            thisList = [sy.Matrix(nzeros((n, m))) for _ in
                        range(repr.listOfMonomialsPerDegAsInt[k].size)]

            # Get the derivative corresponding to each monomial
            for j, aMonom in enumerate(repr.listOfMonomialsPerDegAsInt[k]):
                idxDeriv, idxParent = getIdxAndParent(aMonom, repr.listOfMonomialsPerDegAsInt[k - 1], repr.nDims, repr.digits)

                thisList[j] = sy.diff(lastList[idxParent], q[idxDeriv, 0])

            # save
            pDerivFD[f"{fStr}{k:d}"] = thisList
            # Iterate
            lastList = thisList

        pDerivFD[f"{fStr}PDeriv"] = []
        for k in range(0, maxPDerivDeg + 1):
            aKey = f"{fStr}{k:d}"
            aVal = pDerivFD[aKey]

            pDerivFD[f"{fStr}PDeriv"].extend(aVal)
        
        #Create each
        totMonoms = 0
        for k in range(0, maxPDerivDeg + 1):
            totMonoms += len( repr.listOfMonomialsPerDeg[k])
            pDerivFD[f"{fStr}PDeriv_to_{k:d}"] = pDerivFD[f"{fStr}PDeriv"][:totMonoms]
            pDerivFD[f"{fStr}PDeriv_to_{k:d}_MAT"] = sy.Array.zeros(n,m,totMonoms).as_mutable()#tensor.array.MutableDenseNDimArray.zeros(totMonoms,n,
            # m)#Array.zeros(totMonoms,n,m)
            for j,aVal in enumerate(pDerivFD[f"{fStr}PDeriv_to_{k:d}"]):
                #pDerivFD[f"{fStr}PDeriv_to_{k:d}_MAT"][j,:,:] = aVal #TODO I guess __setitem__ does not support slicing which would really sucks
                for idxI in range(aVal.shape[0]):
                    for idxJ in range(aVal.shape[1]):
                        pDerivFD[f"{fStr}PDeriv_to_{k:d}_MAT"][idxI, idxJ, j] = aVal[idxI, idxJ]

        # array2mat = [{'ImmutableDenseMatrix': np.matrix}, 'numpy']
        array2mat = [{'ImmutableDenseMatrix': np.array}, 'numpy']
        # array2mat = ['numpy']

        #Lambdify all expression
        #TODO check if creating a numba/cython file is more efficient
        tempDict = {}
        for aKey, aVal in pDerivFD.items():
            # Fix for matrices
            if isinstance(aVal, list):
                tempDict[f"{aKey}_eval"] = [sy.lambdify(q, aMat, modules=array2mat) for aMat in aVal]
            elif isinstance(aVal, (sy.array.DenseNDimArray)):
                assert aKey.find("_MAT") != -1
                # Some hack as n-dimensional arrays are not properly lambdified
                thisShape = aVal.shape
                thisSize = nprod(thisShape)
                aValM = sy.Matrix(aVal.reshape((thisSize)))
                tempDict[f"{aKey}_eval_flat"] = sy.lambdify(q, aValM, modules=array2mat)
                tempDict[f"{aKey}_eval"] = lambda *args: tempDict[f"{aKey}_eval_flat"](*args).reshape(thisShape)
            else:
                raise RuntimeError
        pDerivFD.update(tempDict)

    return pDerivFD