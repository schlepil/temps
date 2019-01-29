from coreUtils import *
from polynomial.utils import *

from multivar_horner.multivar_horner import MultivarPolynomial, HornerMultivarPolynomial

class polynomials():
    def __init__(self, repr:polynomialRepr, coeffs:np.ndarray=None, maxDeg:int=None, alwaysFull:bool=True):
        
        self.repr = repr
        
        self._isUpdate=False
        
        if coeffs is None:
            self.__coeffs = np.zeros((self.repr.nMonoms,), dtype=nfloat)
        else:
            # Must be convertible
            self.__coeffs = narray(coeffs).astype(nfloat).reshape((self.repr.nMonoms,))
        
        self.__coeffs.setflags(write=False)
        
        self._alwaysFull = alwaysFull

        self._evalPoly = None

        if maxDeg is None:
            self.maxDeg = self.repr.maxDeg
        else:
            assert maxDeg<=self.repr.maxDeg
            self.maxDeg = maxDeg

    
    def __copy__(self):
        return polynomials(self.repr, np.copy(self.__coeffs), self._alwaysFull)
    
    def __deepcopy__(self, memodict={}):
        return polynomials(dp(self.repr, memodict), np.copy(self.__coeffs), cp(self._alwaysFull))
    
    @property
    def coeffs(self):
        return self.__coeffs
    
    @coeffs.setter
    def coeffs(self, newCoeffs):
        try:
            self.__coeffs = narray(newCoeffs).astype(nfloat).reshape((self.repr.nMonoms,))
        except:
            print("Could not be converted -> skip")
        self.__coeffs.setflags(write=False)
        # Update degree
        self.maxDeg = self.getMaxDegre() #monomial are of ascending degree -> last nonzero coeff determines degree
        self._isUpdate = False

    def getMaxDegree(self):
        return self.repr.listOfMonomials[np.argwhere(self.__coeffs != 0.)[-1]].sum()

    def computeInternal(self, full:bool=False):
        # This computes and stores the horner scheme
        # If not told otherwise it will try to seek a sparse layout, but only takes into account explicit zeros
        
        #shortcut of always full
        if self._alwaysFull:
            if self._evalPoly is None:
                self._evalPoly = MultivarPolynomial(self.__coeffs.reshape((-1,1)),np.require(narray(self.repr.listOfMonomials), dtype=nint))
            else:
                self._evalPoly.coefficients = self.coeffs.reshape((-1,1))
            self._isUpdate = True
            return
                
        
        if full:
            thisExp = np.require(narray(self.repr.listOfMonomials), dtype=nint)
            thisCoeffs = self.__coeffs
        else:
            thisExp = []
            thisCoeffs = []
            for aExp, aCoeff in zip(self.repr.listOfMonomials, self.__coeffs):
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
        