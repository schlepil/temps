from coreUtils import *
from polynomial import *

# Implements base relaxations and constraints


@njit(cache=True)
def populateCstr(cMat, firstIdxPerK, posInCMat, whichCoeff, c):
    """
    Constructs the constraint matrix corresponding to a semi-algebraique constraint
    cMat = (nMonomsHalf**2, nMonoms) Each column of the constraint matrix corresponds to the flattened symmetric coefficient matrix corresponding to the a monom/variable
    :param cMat: Init to zero assumed
    :param firstIdxPerK:  First Idx for each column
    :param posInCMat: j-th element of the k-th row
    :param whichCoeff: Which coefficient of the polynomial (in c) is associated
    :param c: coefficients of the constraint poly
    :return: cMat
    """

    for k in range(cMat.shape[0]):
        for idx in range(firstIdxPerK[k], firstIdxPerK[k+1]):
            cMat[k,posInCMat[idx]] += c[whichCoeff[idx]]

    return cMat

#We will only look at the upper triangular matrix as they are all symmetric

def polyListEval(listOfMonomials:"List of np.arrays", x:np.array):
    if __debug__:
        lenExp = listOfMonomials[0].size
        assert all([aMonom.size==lenExp for aMonom in listOfMonomials])
        assert x.size==lenExp

    allMonoms = np.vstack(listOfMonomials)
    x = x.reshape((-1,))

    monomVals = nprod(npower(x, allMonoms), axis=1)
    monomVals.resize((1, monomVals.size))

    return monomVals


def evalCstrMat(listOfMonomials, cstrMat, cstrRHS, x):
    if x.size == len(listOfMonomials):
        monomVals = x
    else:
        monomVals = polyListEval(listOfMonomials, x)
    
    dim = int(cstrMat.shape[0]**.5)
    
    return nsum(nmultiply(cstrMat, monomVals),axis=1).reshape((dim,dim))-cstrRHS.reshape((dim,dim))

class lasserreRelax:
    def __init__(self, repr:polynomialRepr, emptyClass=False):
        
        assert repr.maxDeg%2 == 0, "The representation has to be of even degree"
        
        self.repr = repr
        
        self.degHalf = repr.maxDeg//2

        #Forward. Note: Do not exchange the repr
        self.digits = self.repr.digits
        self.listOfMonomials = self.repr.listOfMonomials
        self.listOfMonomialsAsInt = self.repr.listOfMonomialsAsInt
        self.listOfMonomialsPerDeg = self.repr.listOfMonomialsPerDeg
        self.maxDeg = self.repr.maxDeg
        self.nMonoms = self.repr.nMonoms


        if not emptyClass:
            self.compute()

    def compute(self):
        
        self.monomsListH = []
        for k in range(self.degHalf+1):
            self.monomsListH.extend(self.repr.listOfMonomialsPerDeg[k])
        # to int
        self.monomsListHInt = narray(lmap(lambda aList: list2int(aList, self.digits), self.monomsListH), dtype=nint).reshape(((-1,)))
        
        self.degHalfMonoms = len(self.monomsListH)
        
        # Monomial relaxation matrix
        self.monomMatrix = self.monomsListHInt.reshape((-1,1))+self.monomsListHInt.reshape((1,-1))
        
        # Count and store sparse occurences
        self.occurencesPerVarnum = np.zeros((self.nMonoms,), nint)
        # The constraints saved in coordinate form
        # Later on always keep in mind that the zero variable is the constant term
        # For the moment store the matrices in column
        self.constraintMat = variableStruct()
        # All matrices are symmetric -> the data is equivalent in fortran or c ordering
        allCMat = nzeros((self.monomMatrix.size, self.nMonoms), dtype=nfloat)
        for k, aMonomInt in enumerate(self.listOfMonomialsAsInt):
            thisCMat = (self.monomMatrix == aMonomInt).astype(nfloat)
            self.occurencesPerVarnum[k] = np.sum(thisCMat)
            allCMat[:,k] = thisCMat.flatten('F')
        
        # Lasserre -> constraintMat >= 0 ; cone programming A.x <= b
        # -> Inverse sign

        self.constraintMat.allCMat = -allCMat
        self.constraintMat.allCMatCOO = sparse.coo_matrix(allCMat)
        self.constraintMat.i = self.constraintMat.allCMatCOO.row.astype(nintu)
        self.constraintMat.col = self.constraintMat.allCMatCOO.col.astype(nintu)
        self.constraintMat.data = self.constraintMat.allCMatCOO.data.astype(nfloat)
        self.constraintMat.firstIdxPerMonom = np.hstack((0, np.cumsum(self.occurencesPerVarnum))).astype(nintu)
        self.constraintMat.rhs = np.zeros((self.constraintMat.allCMat.shape[0],1),dtype=nfloat)

        self.constraintMat.shape = (self.monomMatrix.size, self.nMonoms)

    def getCstr(self, isSparse=False):

        assert isSparse in [False, 'coo', 'csr', 'csc'], "unqualified format"

        if isSparse is False:
            return self.constraintMat.allCMat,self.constraintMat.rhs
        elif isSparse == 'coo':
            return self.constraintMat.allCMatCOO,self.constraintMat.rhs
        elif isSparse == 'csr':
            return self.constraintMat.allCMatCOO.tocsr(),self.constraintMat.rhs
        else:
            return self.constraintMat.allCMatCOO.tocsc(),self.constraintMat.rhs
    
    def evalCstr(self, x:np.array):
        if __debug__:
            assert (x.size == self.repr.nDims) or (x.size == self.repr.nMonoms), "Wrong dimension, only single point evaluation"
        if x.size == self.repr.nMonoms:
            thisC = x
        else:
            thisC = polyListEval(self.repr.listOfMonomials, x)
        return evalCstrMat(self.listOfMonomials, *self.getCstr(isSparse=False), thisC)


class lasserreConstraint:
    def __init__(self, baseRelax:lasserreRelax, poly:polynomial, nRelax:int=None):
        """
        This constraint relaxes the actual constraint
        g(x)>=0.
        
        :param baseRelax:
        :param poly:
        :param nRelax:
        """
        
        assert poly.getMaxDegree()<poly.repr.maxDeg, "Increase relaxation order of the base relaxation"
        assert (nRelax is None) or ((poly.maxDeg+nRelax) <= poly.repr.maxDeg), "Decrease relaxation order of this constraint"
        assert poly.getMaxDegree() < baseRelax.maxDeg, "Inconsistent"

        self.baseRelax = baseRelax
        self.poly = poly
        self.nRelax = nRelax if nRelax is not None else 2*((self.baseRelax.repr.maxDeg-poly.getMaxDegree())//2)

        self.polyDeg = poly.getMaxDegree()

        #shortcuts
        self.repr = self.baseRelax.repr
        self.listOfMonomials = self.baseRelax.listOfMonomials
        self.listOfMonomialsAsInt = self.baseRelax.listOfMonomialsAsInt
        self.monom2num = self.baseRelax.repr.monom2num
        self.num2monom = self.baseRelax.repr.num2monom

        self.precompute()


    def precompute(self):

        assert self.nRelax%2 == 0, "Relax must be even"

        self.cstrMatDef = variableStruct(firstIdxPerK=[0], posInCMat=[], whichCoeff=[])

        maxIdxPoly = self.repr.varNumsUpToDeg[self.polyDeg].size

        maxIdxRelax = self.repr.varNumsUpToDeg[self.nRelax//2].size

        cstrMonoms = self.listOfMonomialsAsInt[:maxIdxRelax]
        cstrMatRelax = cstrMonoms.reshape((-1,1))+cstrMonoms.reshape((1,-1))

        #Multiply each entry of the cstrMatrix with the polynomial and store the abstraction
        k = -1
        for i in range(cstrMatRelax.shape[0]):
            for j in range(cstrMatRelax.shape[1]):
                # Entry mat[i,j]
                # New k -> Create or increment
                k += 1
                self.cstrMatDef.firstIdxPerK.append(0)

                cMatMonom = cstrMatRelax[i,j]
                # Multiply with polynomial
                for idxC, monomP in enumerate(self.poly.repr.listOfMonomialsAsInt[:maxIdxPoly]):
                    self.cstrMatDef.firstIdxPerK[-1] +=1
                    self.cstrMatDef.whichCoeff.append(idxC) #Save this coefficient
                    self.cstrMatDef.posInCMat.append(self.monom2num[monomP+cMatMonom]) #variable number associated to the product monomial
        # Here again: Lasserre cstrMatRelax >= 0 -> inverse sign for cone prog
        # This is done when constructing the constraint
        
        # Done, transform to array
        self.cstrMatDef.firstIdxPerK, self.cstrMatDef.posInCMat, self.cstrMatDef.whichCoeff = narray(self.cstrMatDef.firstIdxPerK, dtype=nintu), narray(self.cstrMatDef.posInCMat, dtype=nintu), narray(self.cstrMatDef.whichCoeff, dtype=nintu)
        self.cstrMatDef.firstIdxPerK = np.cumsum(self.cstrMatDef.firstIdxPerK)
        self.cstrMatDef.cstrMatRelax = cstrMatRelax
        self.cstrMatDef.shapeMatRelax = cstrMatRelax.shape
        self.cstrMatDef.shapeCstr = (cstrMatRelax.size, self.repr.nMonoms)
        self.cstrMatDef.rhs = np.zeros((cstrMatRelax.size,1),dtype=nfloat)
        return None
    
    def getCstr(self, isSparse=False):
        if __debug__:
            assert isSparse in [False, 'coo', 'csr', 'csc'], "unrecongnized"

        cstrMat = np.zeros(self.cstrMatDef.shapeCstr, dtype=nfloat)

        cstrMat = populateCstr(cstrMat, self.cstrMatDef.firstIdxPerK, self.cstrMatDef.posInCMat, self.cstrMatDef.whichCoeff, self.poly.coeffs.ravel())
        cstrMat = -cstrMat #The constraint is g(x) >= 0, but optimization convention is A.x <_k b or more precisely A.x + gamma = b with gamma >=_k 0
        
        if isSparse is False:
            return cstrMat, self.cstrMatDef.rhs
        elif isSparse == 'coo':
            return sparse.coo_matrix(cstrMat), self.cstrMatDef.rhs
        elif isSparse == 'csr':
            return sparse.csr_matrix(cstrMat), self.cstrMatDef.rhs
        else:
            return sparse.csc_matrix(cstrMat), self.cstrMatDef.rhs
    
    def evalCstr(self, x:np.array):
        # evalCstrMat(listOfMonomials,cstrMat,x):
        if x.size == self.repr.nMonoms:
            thisC = x
        else:
            thisC = polyListEval(self.listOfMonomials, x)
        return evalCstrMat(self.listOfMonomials, *self.getCstr(isSparse=False), thisC)
        
    
    










        
        
        
        