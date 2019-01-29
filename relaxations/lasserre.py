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
            cMat[k,posInCMat[idx]] = c[whichCoeff[idx]]

    return cMat

#We will only look at the upper triangular matrix as they are all symmetric

def evalCtrMat(listOfMonomials,cstrMat,x):
    
    allMonoms = np.vstack(listOfMonomials)
    x=x.reshape((-1,))
    
    monomVals = nprod(npower(x,allMonoms),axis=1)
    monomVals.resize((1,monomVals.size))
    
    dim = int(cstrMat.shape[0]**.5)
    
    return nsum(nmultiply(cstrMat, monomVals),axis=1).reshape((dim,dim))

class lasserreRelax:
    def __init__(self, repr:polynomialRepr, emptyClass=False):
        
        assert repr.maxDeg%2 == 0, "The representation has to be of even degree"
        
        self.repr = repr
        
        self.degHalf = repr.maxDeg//2
        
        if not emptyClass:
            self.compute()
    
    #Forward
    @property
    def digits(self):
        return self.repr.digits
    
    @property
    def maxDeg(self):
        return self.repr.maxDeg

    @property
    def listOfMonomialsAsInt(self):
        return self.repr.listOfMonomialsAsInt

    @property
    def nMonoms(self):
        return self.repr.nMonoms

#    @property
#    def(self):
#        return self.repr.

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

        self.constraintMat.allCMat = allCMat
        self.constraintMat.allCMatCOO = sparse.coo_matrix(allCMat)
        self.constraintMat.i = self.constraintMat.allCMatCOO.row.astype(nintu)
        self.constraintMat.col = self.constraintMat.allCMatCOO.col.astype(nintu)
        self.constraintMat.data = self.constraintMat.allCMatCOO.data.astype(nfloat)
        self.constraintMat.firstIdxPerMonom = np.hstack((0, np.cumsum(self.occurencesPerVarnum))).astype(nintu)

        self.constraintMat.shape = (self.monomMatrix.size, self.nMonoms)

    def getCstr(self, isSparse=False):

        assert isSparse in [False, 'coo', 'csr', 'csc'], "unqualified format"

        if isSparse is False:
            return self.constraintMat.allCMat
        elif isSparse == 'coo':
            return self.constraintMat.allCMatCOO
        elif isSparse == 'csr':
            return self.constraintMat.allCMatCOO.tocsr()
        else:
            return self.constraintMat.allCMatCOO.tocsc()
    
    def evalCstr(self, x:np.array):
        if __debug__:
            assert x.size == self.repr.nDims, "Wrong dimension, only single point evaluation"
        return evalCtrMat(self.listOfMonomials, self.getCstr(sparse=False),x.flatten())


class lasserreConstraint:
    def __init__(self, baseRelax:lasserreRelax, poly:polynomials, nRelax:int=None):
        assert poly.getMaxDegree()<poly.repr.maxDeg, "Increase relaxation order of the base relaxation"
        assert (nRelax is None) or ((poly.maxDeg+nRelax) <= poly.repr.maxDeg), "Decrease relaxation order of this constraint"
        assert poly.getMaxDegree() == poly.maxDeg, "Inconsistent"

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

        maxIdxPoly = 0
        for k in range(self.polyDeg+1):
            maxIdxPoly += len(self.baseRelax.repr.listOfMonomialsPerDeg[k])

        maxIdxRelax = 0
        for k in range(self.nRelax//2+1):
            maxIdxRelax += len(self.baseRelax.repr.listOfMonomialsPerDeg[k])

        cstrMonoms = self.listOfMonomialsAsInt[:maxIdxRelax]
        cstrMatRelax = cstrMonoms.reshape((-1,1))+cstrMonoms.reshape((1,-1))

        #Multiply each entry of the cstrMatrix with the polynomial and store the abstraction
        k = -1
        for i in range(cstrMatRelax.shape[0]):
            for j in range(cstrMatRelax.shape[1]):
                # Entry mat[i,j]
                # New k -> Create or increment
                k += 1
                self.cstrMatDef.firstIdxPerK.append([0])

                cMatMonom = cstrMatRelax[i,j]
                # Multiply with polynomial
                for idxC, monomP in enumerate(self.poly.repr.listOfMonomialsAsInt[:maxIdxPoly]):
                    self.cstrMatDef.firstIdxPerK[-1] +=1
                    self.cstrMatDef.whichCoeff.append(idxC) #Save this coefficient
                    self.cstrMatDef.posInCMat.append(self.monom2num(monomP+cMatMonom)) #variable number associated to the product monomial
        # Done, transform to array
        self.cstrMatDef.firstIdxPerK, self.cstrMatDef.posInCMat, self.cstrMatDef.whichCoeff = narray(self.cstrMatDef.firstIdxPerK, dtype=nintu), narray(self.cstrMatDef.posInCMat, dtype=nintu), narray(self.cstrMatDef.whichCoeff, dtype=nintu)
        self.cstrMatDef.cstrMatRelax = cstrMatRelax
        self.cstrMatDef.shapeMatRelax = cstrMatRelax.shape
        self.cstrMatDef.shapeCstr = (cstrMatRelax.size, self.repr.nMonoms)
        return None
    
    def getCstr(self, c:np.array=None, isSparse=False):
        if __debug__:
            assert isSparse in [False, 'coo', 'csr', 'csc'], "unrecongnized"

        cstrMat = np.zeros(self.cstrMatDef.shapeCstr, dtype=nfloat)
        
        c = self.poly.coeffs if c is None else c
        assert c.size == self.repr.nMonoms, "coefficients must be given for all monomials"

        cstrMat = populateCstr(cMat, self.cstrMatDef.firstIdxPerK, posInCMat, whichCoeff, c)
        
        if isSparse is False:
            return cstrMat
        elif isSparse == 'coo':
            return sparse.coo_matrix(cstrMat)
        elif isSparse == 'csr':
            return sparse.csr_matrix(cstrMat)
        else:
            return sparse.csc_matrix(cstrMat)
    
    def evalCstr(self, x:np.array):
        
    
    










        
        
        
        