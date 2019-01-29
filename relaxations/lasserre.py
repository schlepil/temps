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
        self.constraintMat = variableStruct(i=None, j=None, data=None)
        # All matrices are symmetric -> the data is equivalent in fortran or c ordering
        allCMat = nzeros((self.monomMatrix.size, self.nMonoms), dtype=nfloat)
        for k, aMonomInt in enumerate(self.listOfMonomialsAsInt):
            thisCMat = (self.monomMatrix == aMonomInt).atype(nfloat)
            self.occurencesPerVarnum[k] = np.sum(thisCMat)
            allCMat[:,k] = thisCMat.flatten('F')
        allCMatC00 = sparse.coo_matrix(allCMat)

        self.constraintMat.i = narray(self.constraintMat.i, dtype=nintu)
        self.constraintMat.j = narray(self.constraintMat.j, dtype=nintu)
        self.constraintMat.data = narray(self.constraintMat.data, dtype=nfloat)
        self.constraintMat.firstIdxPerMonom = np.hstack((0, np.cumsum(self.occurencesPerVarnum))).astype(nintu)

        self.constraintMat.shape = (self.monomMatrix.size, self.nMonoms)

    def getCstr(self, sparse=False):

        assert sparse in [False, 'coo', 'csr', 'csc']

        thisMat = sparse.coo_matrix((self.constraintMat.data, (self.constraintMat.i,self.constraintMat.j)), dtype=nfloat, shape=self.constraintMat.shape)

        if sparse is False:
            return thisMat.todense()
        elif sparse == 'coo':
            return thisMat
        elif sparse == 'csr':
            return thisMat.tocsr()
        else:
            return thisMat.tocsc()


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
        self.cstrMatDef.cstrMat = cstrMatRelax
        self.cstrMatDef.shape = cstrMatRelax.shape
        return None










        
        
        
        