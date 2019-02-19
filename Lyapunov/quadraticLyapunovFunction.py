from Lyapunov.core import *
from Lyapunov.utils_numba import *

from control import lqr

class quadraticLyapunovFunction(LyapunovFunction):
    """
    Lyapunov function of the form x'.P.x
    """
    def __init__(self,dynSys: dynamicalSystem,P: np.ndarray=None, alpha:float=None):
        
        if __debug__:
            if P is not None:
                assert (P.shape[0] == P.shape[1]) and (P.shape[0] == dynSys.nq)
                assert alpha is not None
            else:
                assert alpha is None
            
        super(quadraticLyapunovFunction,self).__init__(dynSys)
        
        self.P_=None
        self.alpha_=None
        self.C_=None
        self.Ci_=None

        if P is not None:
            self.P = P
            self.alpha = alpha


        self.n = self.dynSys.nq

    @property
    def Ps(self):
        return self.P_/self.alpha_
    @property
    def P(self):
        return self.P_
    @P.setter
    def P(self, newP):
        self.P_ = newP
        if self.alpha_ is not None:
            self.C_ = cholesky(self.P_/self.alpha_)
            self.Ci_ = inv(self.C_)
        
    @property
    def alpha(self):
        return self.alpha_
    @alpha.setter
    def alpha(self, newAlpha):
        self.alpha_ = newAlpha
        if self.P_ is not None:
            self.C_ = cholesky(self.P_/self.alpha_)
            self.Ci_ = inv(self.C_)
    
    def evalV(self, x:np.ndarray, kd:bool=True):
        x = x.reshape((self.n,-1))
        return nsum(ndot(self.C_, x)**2,axis=0,keepdims=kd)
    
    def evalVd(self, x:np.ndarray, dx:np.ndarray, kd:bool=True):
        x = x.reshape((self.n,-1))
        dx = dx.reshape((self.n, -1))
        return nsum(nmultiply(x, ndot((2.*self.P_), dx)),axis=0,keepdims=kd)
    
    def sphere2Ellip(self, x):
        x = x.reshape((self.n, -1))
        return ndot(self.Ci_,x)
    
    def ellip2Sphere(self, x):
        x = x.reshape((self.n, -1))
        return ndot(self.C_, x)
    
    def lqrP(self, Q:np.ndarray, R:np.ndarray, x:np.ndarray=None, A:np.ndarray=None, B:np.ndarray=None, N:np.ndarray=None):
        """
        Solves lqr for
        xd = A.x + B.u
        :param Q:
        :param R:
        :param A:
        :param B:
        :param t:
        :param N:
        :return:
        """
        if __debug__:
            assert (A is None) and (B is None) and (x is not None)
        
        if A is None:
            A = self.dynSys.getTaylorApprox(x,1,1)[0] # TODO change getTaylorApprox
            B = self.dynSys.getTaylorApprox(x,0,0)[1][0,:,:] # Only zero order matrix
        
        #solve lqr
        if N is None:
            K, P, _ = lqr(A, B, Q, R)
        else:
            K, P, _ = lqr(A, B, Q, R, N)
        
        return P,K
        

    def getObjectivePoly(self,x0: np.ndarray = None,dx0: np.ndarray = None,fTaylor: np.ndarray = None,gTaylor: np.ndarray = None,uOpt: np.ndarray = None,idxCtrl: np.ndarray = None,t: float = 0.,taylorDeg: int = 3):
        if __debug__:
            assert (fTaylor is None) == (gTaylor is None)
            assert (x0 is None) != (fTaylor is None)
            assert taylorDeg is not None
            assert uOpt is not None
    
        if dx0 is None:
            dx0 = self.refTraj.getDX(t)
        if fTaylor is None:
            fTaylor,gTaylor = self.dynSys.getTaylorApprox(x0,taylorDeg)
    
        objPoly = polynomial(this.repr)
        objPoly.coeffs[:] = 0.
        
        objPoly.coeffs = evalPolyLyap_Numba(self.P, self.repr.varNumsPerDeg[1], fTaylor, self.repr.varNumsUpToDeg[taylorDeg+1], gTaylor, self.repr.varNumsUpToDeg[taylorDeg+1], dx0, nrequire(np.repeat(self.repr.varNumsPerDeg[1], self.nq), dtype=nintu), uOpt, uMonom, self.idxMat, objPoly.coeffs)
        
        return objPoly

    def getObjectiveAsArray(self, fTaylor: np.ndarray = None, gTaylor: np.ndarray = None, taylorDeg: int = 3,
                         u: np.ndarray = None, uMonom: np.ndarray = None, x0: np.ndarray = None, dx0: np.ndarray = None, t: float = 0.):

        """
        Unified call, returns an array of polynomial coefficients
        First line : Coeffs of system dynamics and (possibly) time-dependent shape [everything besides control]
        second line : Coeffs for first control input, with input given as the polynomial
        third line : Coeffs for second control input, with input given as the polynomial
        ...
        :param x0:
        :param dx0:
        :param fTaylor:
        :param gTaylor:
        :param u:
        :param idxCtrl:
        :param t:
        :param taylorDeg:
        :return:
        """

        if __debug__:
            assert (fTaylor is None) == (gTaylor is None)
            assert (x0 is None) != (fTaylor is None)
            assert taylorDeg is not None
            assert u is not None

        if dx0 is None:
            dx0 = self.refTraj.getDX(t)
        if fTaylor is None:
            fTaylor, gTaylor = self.dynSys.getTaylorApprox(x0, taylorDeg)

        outCoeffs = nzeros((1+self.nu, self.nMonoms), nfloat)

        outCoeffs = evalPolyLyapAsArray_Numba(self.P, self.repr.varNumsPerDeg[1], fTaylor, self.repr.varNumsUpToDeg[taylorDeg], gTaylor,
                                              self.repr.varNumsUpToDeg[taylorDeg], dx0, nrequire(np.tile(self.repr.varNumsPerDeg[0], (self.nq,)), dtype=nintu),
                                              u, uMonom, self.idxMat, outCoeffs) #dx0 only depends on time, not on the position

        return outCoeffs
    
    def getCstrWithDeg(self, gTaylor:np.ndarray, taylorDeg:int, deg:int, which:np.ndarray=None, alwaysFull:bool=True):
        """
        Returns the polynomial constraints resulting from the taylor approximation of the dyn sys / or the pure polynomial dynamics
        deg: either 1 or 3 : linear or polynomial constraint
        which: defines for which control input, if not given then all are computed
        :param gTaylor:
        :param deg:
        :param which:
        :return:
        """
        if __debug__:
            assert deg in (1,3)
            assert taylorDeg>=deg
        
        if which is None:
            which = np.arange(0,self.nu)
        
        if alwaysFull:
            coeffsOut = nzeros((len(which), self.repr.nMonoms))
        
        if deg == 1:
            PG0 = ndot(self.P, gTaylor[0,:,:])
            if alwaysFull:
                coeffsOut[:,len(self.repr.varNumsPerDeg[0]):len(self.repr.varNumsPerDeg[0])+len(self.repr.varNumsPerDeg[1])] = PG0.T[which,:]
            else:
                coeffsOut = PG0.T[which,:] #Linear constraint each row of the returned matrix corresponds to the normal vector of a separating hyperplane
        else:
            # Return full matrix, each row corresponds to the coefficients of one polynomial constraint (for all monomials in the representation)
            # Due the matrix multiplication of P and each g
            PG = nmatmul(self.P, gTaylor)#gTaylor is g[monom,i,j]
            compPolyCstr_Numba(self.repr.varNumsPerDeg[1], PG, self.repr.varNumsUpToDeg[taylorDeg], which, self.idxMat, coeffsOut) #Expects coeffs
            # to be zero
        
        return coeffsOut


class piecewiseQuadraticLyapunovFunction(LyapunovFunction):
    """
    Lyapunov function of the form x'.P.x
    """

    def __init__(self, dynSys: dynamicalSystem, P0: np.ndarray, alpha: float):

        if __debug__:
            if P0 is not None:
                assert (P0.shape[0] == P0.shape[1]) and (P0.shape[0] == dynSys.nq)
                assert alpha is not None
            else:
                assert alpha is None

        super(type(self), self).__init__(dynSys)

        self.P0 = P0
        self.alpha = alpha

        self.n = self.dynSys.nq

    def setPlanesAndComp(self, sepNormals:np.ndarray, addComSize:List[Tuple[np.ndarray]]):

        assert sepNormals.shape[0] == len(addComSize)

        #def __init__(self,dynSys: dynamicalSystem,P: np.ndarray=None, alpha:float=None):

        self.sepNormals = sepNormals #Smaller then zero is first component

        allProds = product([0,1], repeat=len(addComSize))
        self.allprods = narray(list(allProds), dtype=nintu)

        self.lyapVList = {}
        self.prodKeyList = []

        for aprod in self.allprods:
            # get the P
            thisP = self.P0.copy()
            for k,asign in enumerate(aprod):
                thisP += ndot(sepNormals[[k],:].T, sepNormals[[k],:]*addComSize[k][asign])
            #create the key and the object
            thisKey = list2int(narray(aprod, dtype=nintu),digits=1)
            self.prodKeyList.append(thisKey)
            self.lyapVList[thisKey] = quadraticLyapunovFunction(self.dynSys, thisP, self.alpha)

        self.prodKeyList = narray(self.prodKeyList, dtype=nintu)


    def getIdxMat(self, x):
        return (ndot(self.sepNormals, x) <= 0.).astype(nintu)


    def evalV(self, x: np.ndarray, kd: bool = True):
        x = x.reshape((self.n, -1))
        idxMat = self.getIdxMat(x)
        out = nempty((x.shape[1],), dtype=nfloat)
        for k,akey in enumerate(self.prodKeyList):
            thisIdx = nall(idxMat == np.tile(self.allprods[[k], :].T, (1, x.shape[1])), axis=0)
            if nany(thisIdx):
                out[thisIdx] = self.lyapVList[akey].evalV(x[:,thisIdx], kd=False)

        if kd:
            out.resize((1,x.shape[1]))

        return out

    def evalVd(self, x: np.ndarray, dx: np.ndarray, kd: bool = True):
        x = x.reshape((self.n, -1))
        dx = dx.reshape((self.n, -1))
        idxMat = self.getIdxMat(x)
        out = nempty((x.shape[1],), dtype=nfloat)
        for k, akey in enumerate(self.prodKeyList):
            thisIdx = nall(idxMat == np.tile(self.allprods[[k], :].T, (1, x.shape[1])), axis=0)
            if nany(thisIdx):
                out[thisIdx] = self.lyapVList[akey].evalVd(x[:, thisIdx], dx[:, thisIdx], kd=False)

        if kd:
            out.resize((1, x.shape[1]))

        return out


    def sphere2Ellip(self, x):

        x = x.reshape((self.n, -1))
        idxMat = self.getIdxMat(x)
        for k, akey in enumerate(self.prodKeyList):
            thisIdx = nall(idxMat == np.tile(self.allprods[[k], :].T, (1, x.shape[1])), axis=0)
            if nany(thisIdx):
                x[:,thisIdx] = self.lyapVList[akey].sphere2Ellip(x[:, thisIdx])

        return x

    def ellip2Sphere(self, x):

        x = x.reshape((self.n, -1))
        idxMat = self.getIdxMat(x)
        for k, akey in enumerate(self.prodKeyList):
            thisIdx = nall(idxMat == np.tile(self.allprods[[k], :].T, (1, x.shape[1])), axis=0)
            if nany(thisIdx):
                x[:, thisIdx] = self.lyapVList[akey].Ellip2Sphere(x[:, thisIdx])

        return x




