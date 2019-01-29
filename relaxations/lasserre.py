from coreUtils import *
from polynomial import *

# Implements base relaxations and constraints

#We will only look at the upper triangular matrix as they are all symmetric

class lasserreRelax:
    def __init__(self, repr:polynomialRepr, emptyClass=False):
        
        assert repr.maxDeg%2 == 0, "The representation has to be of even degree"
        
        self.repr = repr
        
        self.degHalf = repr.maxDeg/2
        
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
        for k in range(self.degHalf):
            self.monomsListH.extend(self.repr.listOfMonomials[k])
        # to int
        self.monomsListHInt = narray(lmap(lambda aList: list2int(aList, self.digits), self.monomsListH), dtype=nint).reshape(((-1,)))
        
        self.degHalfMonoms = len(self.monomsListH)
        
        # Monomial relaxation matrix
        self.monomMatrix = self.monomsListH.reshape((-1,1))+self.monomsListH.reshape((1,-1))
        
        # Count and store sparse occurences
        self.occurencesPerVarnum = np.zeros((self.nMonoms,), nint)
        self.
        for k, aMonomInt in enumerate(self.listOfMonomialsAsInt):
        
        
        
        