from coreUtils import *
from polynomial.utils import *

from multivar_horner.multivar_horner import MultivarPolynomial, HornerMultivarPolynomial

class polynomial():
    def __init__(self, repr:polynomialRepr, coeffs:np.ndarray=None, alwaysFull:bool=True):
        
        self.repr = repr
        
        self._isUpdate=False
        
        if coeffs is None:
            self._coeffs = nzeros((self.repr.nMonoms,), dtype=nfloat)
        else:
            # Must be convertible
            self._coeffs = narray(coeffs).astype(nfloat).reshape((self.repr.nMonoms,))
        
        self._coeffs.setflags(write=False)
        
        self._alwaysFull = alwaysFull

        self._evalPoly = None

        self.maxDeg = self.getMaxDegree()
        
        if __debug__:
            assert self.maxDeg<=self.repr.maxDeg

    
    def __copy__(self):
        return polynomial(self.repr, np.copy(self._coeffs), self._alwaysFull)
    
    def __deepcopy__(self, memodict={}):
        return polynomial(dp(self.repr, memodict), np.copy(self._coeffs), cp(self._alwaysFull))
    
    @property
    def coeffs(self):
        return self._coeffs
    
    @coeffs.setter
    def coeffs(self, newCoeffs):
        try:
            self._coeffs = narray(newCoeffs).astype(nfloat).reshape((self.repr.nMonoms,))
        except:
            print("Could not be converted -> skip")
        self._coeffs.setflags(write=False)
        # Update degree
        self.maxDeg = self.getMaxDegree() #monomial are of ascending degree -> last nonzero coeff determines degree
        self._isUpdate = False
    
    
    def __add__(self, other):
        if __debug__:
            assert isinstance(other, polynomial)
            assert self.repr == other.repr
        return polynomial(self.repr, self._coeffs + other._coeffs, alwaysFull=self._alwaysFull)
    
    def __iadd__(self, other):
        if __debug__:
            assert isinstance(other, polynomial)
            assert self.repr == other.repr
        self._coeffs+=other._coeffs
        self.maxDeg=self.getMaxDegree()
        return self
    
    def __sub__(self, other):
        if __debug__:
            assert isinstance(other, polynomial)
            assert self.repr == other.repr
        return polynomial(self.repr, self._coeffs - other._coeffs, alwaysFull=self._alwaysFull)
    
    def __isub__(self, other):
        if __debug__:
            assert isinstance(other, polynomial)
            assert self.repr == other.repr
        self._coeffs-=other._coeffs
        self.maxDeg=self.getMaxDegree()
        return self
    
    def __neg__(self):
        return polynomial(self.repr, -self._coeffs, alwaysFull=self._alwaysFull)
    
    def __mul__(self, other):
        if isinstance(other, (float,int)):
            return polynomial(self.repr, float(other) * self._coeffs, alwaysFull=self._alwaysFull)
        else:
            if __debug__:
                assert isinstance(other, polynomial)
                assert self.repr == other.repr
                assert self.maxDeg+other.maxDeg<=self.repr.maxDeg
            
            c = polyMul(self._coeffs, other._coeffs, self.repr.idxMat)
            return polynomial(self.repr, c, alwaysFull=self._alwaysFull)
    
    def __rmul__(self, other):
        # Commute
        return self.__mul__(other)
    
    def __imul__(self, other):
        self._coeffs.setflags(write=True)
        if isinstance(other,(float,int)):
            self._coeffs *= float(other)
        else:
            if __debug__:
                assert isinstance(other, polynomial)
                assert self.repr == other.repr
                assert self.maxDeg+other.maxDeg <= self.repr.maxDeg
        
            self._coeffs = polyMul(self._coeffs,other._coeffs,self.repr.idxMat)
            self.maxDeg = self.maxDeg+other.maxDeg
        self._coeffs.setflags(write=False)
        return self
    
    def __pow__(self, power:int, modulo=None):
        if __debug__:
            assert self.maxDeg*power<=self.repr.maxDeg
        #very simplistic
        newPoly = self*self
        
        for _ in range(power-2):
            newPoly*=self
        return newPoly
        
    
    def round(self,atol=1e-12):
        self._coeffs[np.abs(self._coeffs)<=atol]=0.
        self.maxDeg=self.getMaxDegree()

    def getMaxDegree(self):
        try:
            return self.repr.listOfMonomials[int(np.argwhere(self._coeffs != 0.)[-1])].sum()
        except IndexError:
            #all zero
            return 0


    def computeInternal(self, full:bool=False):
        # This computes and stores the horner scheme
        # If not told otherwise it will try to seek a sparse layout, but only takes into account explicit zeros
        
        #shortcut of always full
        if self._alwaysFull:
            if self._evalPoly is None:
                self._evalPoly = MultivarPolynomial(self._coeffs.reshape((-1, 1)), np.require(narray(self.repr.listOfMonomials), dtype=nint))
            else:
                self._evalPoly.coefficients = self.coeffs.reshape((-1,1))
            self._isUpdate = True
            return
                
        
        if full:
            thisExp = np.require(narray(self.repr.listOfMonomials), dtype=nint)
            thisCoeffs = self._coeffs
        else:
            thisExp = []
            thisCoeffs = []
            for aExp, aCoeff in zip(self.repr.listOfMonomials, self._coeffs):
                if aCoeff != 0:
                    thisExp.append(aExp)
                    thisCoeffs.append(aCoeff)
            thisExp = np.require(narray(thisExp), dtype=nint)
            thisCoeffs = np.require(narray(thisCoeffs), dtype=nfloat)
            thisCoeffs.resize((thisCoeffs.size,1))

        self._evalPoly = MultivarPolynomial(thisCoeffs,thisExp)
        self._isUpdate = True
        
    def setQuadraticForm(self, Q:np.ndarray, qMonoms:np.ndarray=None, h=None, hMonoms=None):
        if __debug__:
            assert Q.shape[0] == Q.shape[1]

        qMonoms = self.repr.varNumsPerDeg[1] if qMonoms is None else qMonoms
        hMonoms = self.repr.varNumsUpToDeg[1] if ((h is not None) and (hMonoms is None)) else hMonoms
        
        if ((h is None) and (hMonoms is None)):
            h=nzeros((1,), dtype=nfloat)
            hMonoms = self.repr.varNumsPerDeg[0]
        
        self._coeffs = quadraticForm_Numba(Q,qMonoms, h, hMonoms, self.repr.idxMat, np.zeros((self.repr.nMonoms,), dtype=nfloat))
        self.maxDeg = self.getMaxDegree()
        return None
    
    def eval(self, x:np.array):
        if not self._isUpdate:
            self.computeInternal()
        
        return self._evalPoly.eval(x)

    def eval2(self, x:np.array, deg:int = None):
        assert self._coeffs.size == self.repr.nMonoms
        
        if deg is None:
            nMonoms_ = self.repr.varNumsUpToDeg[self.maxDeg].size
        else:
            assert deg <= self.repr.maxDeg
            nMonoms_ = len(self.repr.varNumsUpToDeg[deg])
            
        coeffsEval = np.reshape(self._coeffs, (1,self.repr.nMonoms))[[0],:nMonoms_]
        
        if x.shape[0] >= nMonoms_:
            z = x[:nMonoms_, :]
        else:
            z = evalMonomsNumba(x, self.repr.varNum2varNumParents[:nMonoms_, :])
        
        return ndot(coeffsEval, z)


class polyFunction():
    
    def __init__(self,repr:polynomialRepr, shape):
        self.repr = repr
        self.shape_ = narray(shape, ndmin=1)
        self.f = np.ndarray(self.shape_, dtype=object)
        
    def setVal(self, new:np.ndarray):
        assert self.f.shape == new.shape
        
        self.f = new
        return None
    
    def __getitem__(self, key):
        return self.f[key]
    
    def __setitem__(self, key, item):
        self.f[key] = item
        return None
        
    def eval(self, x:np.ndarray):
        """
        if x is 2d -> append an axis
        :param x:
        :return:
        """
        
        xs1 = x.shape[1]
        prodSpace = [[slice(0,xs1)]] + [range(a) for a in self.shape_]

        z = x if (x.shape[0] == self.repr.nMonoms) else evalMonomsNumba(x, self.repr.varNum2varNumParents)
        
        y = nzeros([xs1]+list(self.shape_), dtype=nfloat)
        
        for aIdxL in itertools.product(*prodSpace):
            y[aIdxL]= self.f[aIdxL[1:]].eval2(z) # TODO check if this is standard
        
        if xs1==1:
            y = y[0,:,:]
        
        return y #[nPt, *shape_]