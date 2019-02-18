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
                assert (P.shape[0] == P.shape[1]) and (P.shape[0] == self.dynSys.nq)
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
        x = x.reshape((self.P_.shape[0],-1))
        return nsum(ndot(self.C_, x)**2,axis=0,keepdims=kd)
    
    def evalVd(self, x:np.ndarray, dx:np.ndarray, kd:bool=True):
        x = x.reshape((self.P_.shape[0],-1))
        dx = dx.reshape((self.P_.shape[0], -1))
        return nsum(nmultiply(x, ndot((2.*self.P_), dx)),axis=0,keepdims=kd)
    
    def sphere2Ellip(self, x):
        x = x.reshape((self.P_.shape[0], -1))
        return ndot(self.Ci_,x)
    
    def ellip2Sphere(self, x):
        x = x.reshape((self.P_.shape[0], -1))
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
    
        objPoly = polynomials(this.repr)
        objPoly.coeffs[:] = 0.
        
        objPoly.coeffs = evalPolyLyap_Numba(self.P, self.repr.varNumsPerDeg[1], fTaylor, self.repr.varNumsUpToDeg[taylorDeg+1], gTaylor, self.repr.varNumsUpToDeg[taylorDeg+1], dx0, nrequire(np.repeat(self.repr.varNumsPerDeg[1], self.nq), dtype=nintu), uOpt, uMonom, self.idxMat, objPoly.coeffs)
        
        return objPoly
    
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
            compPolyCstr_Numba(self.repr.varNumsPerDeg[1], PG, self.repr.varNumsUpToDeg[taylorDeg+1], which, self.idxMat, coeffsOut)
        
        return coeffsOut
            
        
        

