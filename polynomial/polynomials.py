from coreUtils import *
from polynomial.utils import *

from multivar_horner.multivar_horner import MultivarPolynomial, HornerMultivarPolynomial

class polynomials():
    def __init__(self, repr:polynomialRepr, coeffs:np.ndarray=None, maxDeg:int=None, alwaysFull:bool=True):
        
        self.repr = repr
        
        self._isUpdate=False
        
        if coeffs is None:
            self._coeffs = np.zeros((self.repr.nMonoms,), dtype=nfloat)
        else:
            # Must be convertible
            self._coeffs = narray(coeffs).astype(nfloat).reshape((self.repr.nMonoms,))
        
        self._coeffs.setflags(write=False)
        
        self._alwaysFull = alwaysFull

        self._evalPoly = None

        if maxDeg is None:
            self.maxDeg = self.getMaxDegree()
        
        if __debug__:
            assert maxDeg<=self.repr.maxDeg

    
    def __copy__(self):
        return polynomials(self.repr, np.copy(self._coeffs), self._alwaysFull)
    
    def __deepcopy__(self, memodict={}):
        return polynomials(dp(self.repr, memodict), np.copy(self._coeffs), cp(self._alwaysFull))
    
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
        self.maxDeg = self.getMaxDegre() #monomial are of ascending degree -> last nonzero coeff determines degree
        self._isUpdate = False
    
    
    def __add__(self, other):
        if __debug__:
            assert isinstance(other, polynomials)
            assert self.repr == other.repr
        return polynomials(self.repr, self._coeffs+other._coeffs,allwaysFull=self._alwaysFull)
    
    def __iadd__(self, other):
        if __debug__:
            assert isinstance(other, polynomials)
            assert self.repr == other.repr
        self._coeffs+=other._coeffs
        self.maxDeg=self.getMaxDegree()
        return None
    
    def __sub__(self, other):
        if __debug__:
            assert isinstance(other, polynomials)
            assert self.repr == other.repr
        return polynomials(self.repr, self._coeffs-other._coeffs,allwaysFull=self._alwaysFull)
    
    def __isub__(self, other):
        if __debug__:
            assert isinstance(other, polynomials)
            assert self.repr == other.repr
        self._coeffs-=other._coeffs
        self.maxDeg=self.getMaxDegree()
        return None
    
    def __mul__(self, other):
        if __debug__:
            assert isinstance(other, polynomials)
            assert self.repr == other.repr
            assert self.maxDeg+other.maxDeg<=self.repr.maxDeg
        
        c = polyMul(self._coeffs, other._coefs, self.repr.idxMat)
        return polynomials(self.repr, c, self.maxDeg+other.maxDeg, alwaysFull=self._alwaysFull)
    
    def __rmul__(self, other):
        # Commute
        return self.__mul__(other)
    
    def __imul__(self, other):
        if __debug__:
            assert isinstance(other,polynomials)
            assert self.repr == other.repr
            assert self.maxDeg+other.maxDeg <= self.repr.maxDeg
    
        self._coeffs = polyMul(self._coeffs,other._coefs,self.repr.idxMat)
        self.maxDeg = self.maxDeg+other.maxDeg
        return None
    
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
        return self.repr.listOfMonomials[int(np.argwhere(self._coeffs != 0.)[-1])].sum()

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
    
    def eval(self, x:np.array):
        if not self._isUpdate:
            self.computeInternal()
        
        return self._evalPoly.eval(x)
        