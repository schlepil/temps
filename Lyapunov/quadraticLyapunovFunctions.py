from Lyapunov.core import *
from Lyapunov.utils_numba import *

import polynomial

from control import lqr

if coreOptions.doPlot:
    import plotting as plot


#Indexing

# Interpolators
# Common signature
##  self.interpolate(tIn, P_, C, Plog, t_, returnPd) ##

def getLowerNUpperInd(tIn, t):
    tInd = np.interp(tIn, t, np.arange(0, t.size), left=0.01, right=t.size-1.01)
    tIndL = narray(nmaximum(0, np.floor(tInd)), dtype=nint, ndmin=1)
    tIndU = narray(nminimum(np.ceil(tInd), t.size-1), dtype=nint, ndmin=1)
    
    if t.size > 1:
        for i in np.argwhere(tIndL==tIndU):
            if tIndL[i]>0:
                tIndL[i]-=1
            elif tIndU[i]<t.size-1:
                tIndU[i] += 1
            else:
                print(f"{t}\n{tIn}\n{tIndL}\n{tIndU}")
                raise RuntimeError
    return tIndL, tIndU
    

# Cartesian interpolation
def standardInterpol(tIn:np.ndarray, P:np.ndarray, C:np.ndarray, Plog:np.ndarray, t:np.ndarray, returnPd:bool):
    tIn = narray(tIn)
    #interpolation bondaries
    tIndL, tIndU = getLowerNUpperInd(tIn, t)
    
    if t.size == 1:
        if __debug__:
            assert nall(tIn.squeeze() == t[0])
        alpha = nzeros((1,), dtype=nfloat)
    else:
        alpha = (tIn-t[tIndL])/(t[tIndU]-t[tIndL])
    
    alpha = np.transpose(np.broadcast_to(alpha, (P.shape[1], P.shape[2], tIn.size)), (2,1,0))
    
    if not returnPd:
        return (P[tIndU,:,:]*alpha + P[tIndL,:,:]*(1.-alpha)).squeeze()
    else:
        return (P[tIndU, :, :]*alpha+P[tIndL, :, :]*(1.-alpha)).squeeze(), ((P[tIndU, :, :]-P[tIndL, :, :])/(np.transpose(np.broadcast_to(t[tIndU]-t[tIndL], (P.shape[1],P.shape[2], tIn.size)), (2,1,0)))).squeeze()

# Cartesian no deriv
def standardInterpolNoDeriv(tIn:np.ndarray, P:np.ndarray, C:np.ndarray, Plog:np.ndarray, t:np.ndarray, returnPd:bool):
    P = standardInterpol(tIn, P, C, Plog, t, False)
    if returnPd:
        return P, np.zeros_like(P)
    else:
        return P

def geodesicInterpol(tIn:np.ndarray, P:np.ndarray, C:np.ndarray, Plog:np.ndarray, t:np.ndarray, returnPd:bool): #(Pn, alphan, Pn1, alphan1, t, t0, t1):
    # let dist(A,B) be the Frobenius norm of A-B, norm(A_B) then
    # P(t) = P_n.exp((Phi.Pn.Cni)*t).C_n is the geodesic minimizing
    # int ds with ds = norm(Cni.dP.Cni)

    tIn = narray(tIn)

    # interpolation bondaries
    tIndL, tIndU = getLowerNUpperInd(tIn, t)
    
    # Following https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Jayasumana_Kernel_Methods_on_2013_CVPR_paper.html
    # and Serge Lang, Fundamentals of Differential Geometry, p 326

    if t.size == 1:
        if __debug__:
            assert nall(tIn.squeeze() == t[0])
        alpha = nzeros((1,), dtype=nfloat)
        dT = 1.
    else:
        alpha = (tIn-t[tIndL])/(t[tIndU]-t[tIndL])
    
    
    Pt = np.zeros((tIn.size, P.shape[1], P.shape[2]), dtype=nfloat)
    Pdt = np.zeros_like(Pt)
    for k in range(tIn.size):
        Pt[k,:,:] = expm((1.-alpha[k])*Plog[tIndL[k],:,:]+alpha[k]*Plog[tIndU[k],:,:])
        # TODO vectorize
        Pdt[k,:,:] = ndot((Plog[tIndU[k],:,:]-Plog[tIndL[k],:,:])/dT[k], Pt[k,:,:])
    
    if returnPd:
        if t.size == 1:
            raise UserWarning('Cannot compute derivative based on one point')
        return Pt, Pdt
    else:
        return Pt

def geodesicInterpolNoDeriv(tIn:np.ndarray, P:np.ndarray, C:np.ndarray, Plog:np.ndarray, t:np.ndarray, returnPd:bool):
    P = geodesicInterpolNoDeriv(tIn, P, C, Plog, t, False)
    
    if returnPd:
        return P, np.zeros_like(P)
    else:
        return P

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
            
        super(quadraticLyapunovFunction,self).__init__(2, dynSys)
        
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
    
    def getPnPdot(self, t: np.ndarray, returnPd = True):
        P = self.Ps
        try:
            t = float(t)
        except:
            if __debug__:
                assert isinstance(t, np.ndarray)
            P = np.tile(P, (t.size, P.shape[0], P.shape[1]))
        if returnPd:
            return P, np.zeros_like(P)
        else:
            return P
    
    def getZone(self, t):
        P,Pd = self.getPnPdot(t, True)
        alpha = 1. if len(P.shape) == 2 else nones((P.shape[0],), nfloat)
        return [P,alpha, Pd]
        
    def getLyap(self, t):
        P = self.getPnPdot(t, False)
        return (P,1.)

    def plot(self, ax: "plot.plt.axes", t: float = 0.0, x0:np.ndarray=None, opts={}):
    
        opts_ = {'plotStyle':'proj', 'linewidth':1., 'color':[0.0, 0.0, 1.0, 1.0],
                 'faceAlpha':0.5, 'linestyle':'-',
                 'plotAx':np.array([0, 1])}
        opts_.update(opts)
    
        P = self.Ps
        x = self.dynSys.refTraj.getX(t) if x0 is None else x0
    
        return plot.plotEllipse(ax, x, P, 1., **opts_)
    
    def evalV(self, x:np.ndarray, kd:bool=True):
        x = x.reshape((self.n,-1))

        return nsum(ndot(self.C_, x)**2,axis=0,keepdims=kd)
    
    def evalVd(self, x:np.ndarray, dx:np.ndarray, kd:bool=True):
        x = x.reshape((self.n,-1))
        dx = dx.reshape((self.n, -1))
        return nsum(nmultiply(x, ndot((2.*self.P_), dx)),axis=0,keepdims=kd)

    def convAng(self, x: np.ndarray, xd: np.ndarray, kd: bool = True):

        """
        Returns the angle between the normal to the surface and the velocity
        :param x:
        :param xd:
        :param t:
        :param kd:
        :return:
        """
        assert (x.shape == xd.shape)

        # Get the quadratic function
        P = self.Ps_

        # Project to get normal directions at x
        n = neinsum( "in,ij->jn", x, P) #TODO check if correct

        #Normalize
        n /= (norm(n,axis=0,keepdims=True)+coreOptions.floatEps)
        xdn = xd / (norm(xd,axis=0,keepdims=True)+coreOptions.floatEps)

        return np.arccos(nsum(-(nmultiply(n,xdn)), axis=0, keepdims=kd))
    
    def sphere2Ellip(self, x):
        x = x.reshape((self.n, -1))
        return ndot(self.Ci_,x)
    
    def ellip2Sphere(self, y):
        y = y.reshape((self.n, -1))
        return ndot(self.C_, y)
    
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
            AA=self.dynSys.getTaylorApprox(x,1,1)
            A = AA[0] # TODO change getTaylorApprox
            print('Hiiiii, its A',A)
            print("self.dynSys.getTaylorApprox(x,1,1)",AA)
            print('',AA[0].shape,AA[1].shape)
            BB = self.dynSys.getTaylorApprox(x,0,0) # Only zero order matrix
            print('self.dynSys.getTaylorApprox(x,0,0)', BB)
            print('',BB[0].shape,BB[1].shape)
            B=BB[1][0,:,:]
            print('Hiiiii, its B', B)
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
        
        objPoly.coeffs = evalPolyLyap_Numba(self.P, self.repr.varNumsPerDeg[1], fTaylor, self.repr.varNumsUpToDeg[taylorDeg+1], gTaylor, self.repr.varNumsUpToDeg[taylorDeg+1], dx0, nrequire(np.repeat(self.repr.varNumsPerDeg[1], self.nq), dtype=nintu), uOpt, uMonom, self.idxMat, objPoly.coeffs, np.zeros_like(self.P))
        
        return objPoly

    def getObjectiveAsArray(self, fTaylor: np.ndarray = None, gTaylor: np.ndarray = None, taylorDeg: int = 3,
                         u: np.ndarray = None, uMonom: np.ndarray = None, x0: np.ndarray = None, dx0: np.ndarray = None, t: float = 0., P=None, Pdot=None):
        
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
        # evalPolyLyapAsArray_Numba(P, monomP, f, monomF, g, monomG, dx0, monomDX0, u, monomU, idxMat, coeffsOut, Pdot)
        P = self.P if P is None else P
        if Pdot is not None:
            raise UserWarning
        Pdot = np.zeros_like(self.P) if Pdot is None else Pdot
        
        outCoeffs = evalPolyLyapAsArray_Numba(P, self.repr.varNumsPerDeg[1], fTaylor, self.repr.varNumsUpToDeg[taylorDeg], gTaylor,
                                              self.repr.varNumsUpToDeg[taylorDeg], dx0, nrequire(np.tile(self.repr.varNumsPerDeg[0], (self.nq,)), dtype=nintu),
                                              u, uMonom, self.idxMat, outCoeffs, Pdot) #dx0 only depends on time, not on the position

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


class quadraticLyapunovFunctionTimed(LyapunovFunction):
    """
    Lyapunov function of the form x'.P(t).x <= alpha(t)
    """

    def __init__(self, dynSys: dynamicalSystem, P:np.ndarray=None, alpha:np.ndarray = None, t:np.ndarray = None):
        
        if __debug__:
            if P is not None:
                assert (P.shape[1] == P.shape[2]) and (P.shape[1] == dynSys.nq)
                assert alpha is not None
                assert P.shape[0] == alpha.size
                assert alpha.size == t.size
            else:
                assert alpha is None
        
        super(type(self), self).__init__(2, dynSys)
        
        self.P_ = None
        self.alpha_ = None
        self.C_ = None
        self.Ci_ = None
        self.Plog_ = None
        self.Ps_ = None
        
        if P is not None:
            self.alpha = alpha
            self.P = P
            self.t_ = t
            
        
        self.n = self.dynSys.nq
        
        self.interpolate = None #Callable for interpolation
        
        self.opts_ = {'zoneCompLvl':3}
        self.optsCtrlDict_ = {'minConvRate':-0.}
    
    @property
    def Ps(self):
        return self.P_/np.transpose(np.broadcast_to(self.alpha_, (self.n, self.n, self.alpha_.size)), (2,1,0))
    
    @property
    def P(self):
        return self.P_
    
    @P.setter
    def P(self, newP):
        if __debug__:
            assert (self.P_ == newP.shape) or (self.P_ is None)
        self.P_ = newP
        self.compute_()
    
    @property
    def alpha(self):
        return self.alpha_
    
    @alpha.setter
    def alpha(self, newAlpha):
        if __debug__:
            assert (newAlpha.size == self.P_.shape[0]) or (newAlpha.size == self.alpha_.size) or (self.alpha_ is None)
        self.alpha_ = newAlpha.reshape((-1,))
        self.compute_()
    
    @property
    def t(self):
        return self.t_
    @t.setter
    def t(self, newt):
        if __debug__:
            assert (t.size == self.alpha_.size) or (t.size == self.t_.size) or (self.t_ is None)
        self.t_ = t.reshape((-1,))
    
    def getAlpha(self, idx=None):
        if idx is None:
            return self.alpha
        else:
            return self.alpha[idx]
    
    def setAlpha(self, newAlpha, idx=None, returnInfo=False):
        if idx is None:
            alphaFromTo = [dp(self.alpha_), dp(newAlpha)]
            self.alpha_ = newAlpha
        else:
            alphaFromTo = [dp(self.alpha_[idx]), dp(newAlpha)]
            self.alpha_[idx] = newAlpha
            self.computeK_(idx)
        return None if not returnInfo else alphaFromTo
    
    def reset(self):
        self.P_ = None
        self.alpha_ = None
        self.C_ = None
        self.Ci_ = None
        self.Plog_ = None
        self.Ps_ = None
        
        return None
    
    def register(self, t, PnAlpha):
        
        if self.P_ is None:
            self.P_ = PnAlpha[0].reshape((1,self.dynSys.nq,self.dynSys.nq))
            self.alpha_ = narray([PnAlpha[1]])
            self.t_ = narray([t])
        else:
            if t in self.t_:
                if __debug__:
                    print(f"Replacing item at {t}")
                ind = np.argwhere(t == self.t_)
                self.t_[ind] = t
                self.alpha_[ind] = PnAlpha[1]
                self.P_[ind,:,:] = PnAlpha[0]
            else:
                if __debug__:
                    print(f"RWInserting item")
                # Keep ordering
                try:
                    ind = np.where(self.t_ > t)[0][0]
                except IndexError:
                    # Larger than all occuring values
                    ind = self.t_.size

                self.t_ = np.hstack((self.t_[:ind], t, self.t_[ind:]))
                self.P_ = np.vstack((self.P_[:ind,:,:], PnAlpha[0].reshape((1,self.P_.shape[1],self.P_.shape[2])), self.P_[ind:,:,:]))
                self.alpha_ = np.hstack((self.alpha_[:ind], PnAlpha[1], self.alpha_[ind:]))
                # Set to None to force reallocate
                self.Plog_ = self.C_ = self.Ci_ = None
        
        # Take the update into account
        self.compute_()
    
    def compute_(self):
        if ((self.P_ is not None) and (self.alpha_ is not None)):
            self.alpha_ = nrequire(self.alpha_, dtype=nfloat)
            self.t_ = nrequire(self.t_, dtype=nfloat)
            self.P_ = nrequire(self.P_, dtype=nfloat)
            
            self.Ps_ = self.Ps
            self.Plog_ = nzeros(self.P_.shape, dtype=nfloat) if self.Plog_ is None else self.Plog_
            self.C_ = nzeros(self.P_.shape, dtype=nfloat) if self.C_ is None else self.C_
            self.Ci_ = nzeros(self.P_.shape, dtype=nfloat) if self.Ci_ is None else self.Ci_
            for i in range(self.P_.shape[0]):
                self.Plog_[i,:,:] = logm(self.P_[i,:,:]/self.alpha_[i])
                self.C_[i,:,:] = cholesky(self.P_[i,:,:]/self.alpha_[i])
                self.Ci_[i,:,:] = inv(self.C_[i,:,:])
        return None
    
    def computeK_(self, k_:int):
        for k in narray(k_, ndmin=1):
            self.Ps_[k, :, :] = self.P_[k, :, :]/self.alpha_[k]
            self.Plog_[k, :, :] = logm(self.P_[k, :, :]/self.alpha_[k])
            self.C_[k, :, :] = cholesky(self.P_[k, :, :]/self.alpha_[k])
            self.Ci_[k, :, :] = inv(self.C_[k, :, :])
        return None
        
    
    def getPnPdot(self, t:np.ndarray, returnPd=True):
        return self.interpolate(t, self.Ps_, self.C_, self.Plog_, self.t_, returnPd)
    
    def getZone(self, t):
        P,Pd = self.getPnPdot(t, True)
        alpha = 1. if len(P.shape) == 2 else nones((P.shape[0],), nfloat)
        return [P,alpha, Pd]
    
    def zone2Cstr(self,aZone, offset:np.ndarray=None):
        """
        Transforms a aZone into a polynomial constraint
        :param aZone:
        :param offset:
        :return:
        """
        thisPoly = polynomial.polynomial(self.dynSys.repr)
        thisPoly.setEllipsoidalConstraint(center=nzeros((self.dynSys.nq,1),dtype=nfloat) if offset is None else offset, radius=aZone[1], P=aZone[0])
        return thisPoly
    
    def getLyap(self, t):
        P = self.getPnPdot(t, False)
        return (P,1.)
    
    def getCtrlDict(self, t:float, fTaylorApprox=None, gTaylorApprox=None,returnZone=True, taylorDeg=None, maxCtrlDeg=2, opts={}):
        
        assert ((0<=maxCtrlDeg) and (maxCtrlDeg<=2))
        assert (taylorDeg is None) or (taylorDeg <= self.dynSys.maxTaylorDeg)
        
        #opts_ = {'minConvRate':0.}
        opts_ = dp(self.optsCtrlDict_)
        recursiveExclusiveUpdate(opts_, opts)
        
        ctrlDict = {}  # Return value
    
        thisPoly = polynomial.polynomial(self.dynSys.repr)  # Helper
    
        allU = [self.dynSys.ctrlInput.getMinU(t), self.dynSys.ctrlInput.refTraj.getU(t), self.dynSys.ctrlInput.getMaxU(t)]
        allDeltaU = [allU[0]-allU[1], allU[2]-allU[1]]
        
        #Get the zone
        zone = self.getZone(t)
        # Get the taylor if neccessary
        if ((fTaylorApprox is None) or (gTaylorApprox is None)):
            fTaylorApproxTmp, gTaylorApproxTmp = self.dynSys.getTaylorApprox(self.refTraj.getX(t))
            fTaylorApprox = fTaylorApproxTmp if fTaylorApprox is None else fTaylorApprox
            gTaylorApprox = gTaylorApproxTmp if gTaylorApprox is None else gTaylorApprox
        # Optimal control
        objectiveStar = self.getObjectiveAsArray(fTaylorApprox, gTaylorApprox, self.dynSys.maxTaylorDeg, np.ones((self.dynSys.nu, 1)), self.repr.varNumsPerDeg[0], dx0=self.dynSys.ctrlInput.refTraj.getDX(t), t=t, zone=zone)
        # Parse
        ctrlDict[-1] = {0:objectiveStar[0, :]}
        # Add minimal exponential convergence rate
        thisPoly.setQuadraticForm(nidentity(self.dynSys.nq))
        ctrlDict[-1][0] -= thisPoly.coeffs*opts_['minConvRate']
    
        for k in range(self.dynSys.nu):
            if __debug__:
                assert abs(objectiveStar[k+1, 0]) <= 1e-9
            #Each part of the total convergence depends linearly on the input
            #ctrlDict[k] = {-1:objectiveStar[k+1, :]*allDeltaU[0][k, 0], 1:objectiveStar[k+1, :]*allDeltaU[1][k, 0]}  # Best is minimal or maximal
            # However this does nolonger depend on the reference control input as the reference velocity is explicitly taken into account now
            # TODO check up
            ctrlDict[k] = {-1:objectiveStar[k+1, :]*allU[0][k, 0], 1:objectiveStar[k+1, :]*allU[2][k, 0]}  # Best is minimal or maximal
    
        # Linear control based on separating hyperplanes
        ctrlDict['PG0'] = ndot(zone[0], gTaylorApprox[0, :, :])
        uCtrlLin, uMonomLin = self.dynSys.ctrlInput.getU(2*np.ones((self.dynSys.nu,), dtype=nint), 0., P=zone[0], PG0=ctrlDict['PG0'], alpha=zone[1], monomOut=True)
        # Attention here the resulting polynomial coefficients are already scaled correctly (no multiplication with deltaU necessary)
        objectivePolyCtrlLin = self.getObjectiveAsArray(fTaylorApprox, gTaylorApprox, self.dynSys.maxTaylorDeg, uCtrlLin, uMonomLin, dx0=self.dynSys.ctrlInput.refTraj.getDX(t), t=t, zone=zone)
    
        # Parse
        for k in range(self.dynSys.nu):
            ctrlDict[k][2] = objectivePolyCtrlLin[k+1, :]  # set linear
    
        ctrlDict['sphereProj'] = False
        
        if returnZone:
            return ctrlDict, zone
        else:
            return ctrlDict
        

    def plot(self, ax: "plot.plt.axes", t: float = 0.0, x0:np.ndarray=None, opts={}):
    
        opts_ = {'pltStyle':'proj', 'linewidth':1., 'color':[0.0, 0.0, 1.0, 1.0],
                 'faceAlpha':0.5, 'linestyle':'-',
                 'plotAx':np.array([0, 1])}
        
        for aKey in opts_.keys():
            try:
                opts_[aKey] = opts[aKey]
            except KeyError:
                pass
        
    
        P = self.getPnPdot(t, False)
        x = self.dynSys.ctrlInput.refTraj.getX(t) if x0 is None else x0
    
        return plot.plotEllipse(ax, x, P, 1., **opts_)
    
    def evalV(self, x: np.ndarray, t:np.ndarray, kd: bool = True):
        t = narray(t, ndmin=1, dtype=nfloat)

        assert (t.size == x.shape[1]) or (t.size == 1)

        allP = self.getPnPdot(t, False)
        x = x.reshape((self.n, -1))
        if t.size == 1:
            if x.shape[1] <= 10:
                V = neinsum("in,ij,jn->n", x, allP, x)
            else:
                C = cholesky(allP)
                V = nsum(ndot(C, x)**2,axis=0,keepdims=kd)
        else:
            V = neinsum("in,nij,jn->n", x,allP,x)
        
        if kd:
            V.resize((1,x.shape[1]))
        else:
            V.resize((x.shape[1],))
        
        return V
    
    def evalVd(self, x: np.ndarray, xd: np.ndarray, t:np.ndarray, kd: bool = True):
        """
        x and xd are derivation variables with respect to reference
        V = x.T.P.x
        Vd = 2x.T.P.xd + x.T.Pd.x
        :param x:
        :param dx:
        :param t:
        :param kd:
        :return:
        """

        t = narray(t, ndmin=1, dtype=nfloat)

        x = x.reshape((self.n, -1))
        xd = xd.reshape((self.n, -1))

        assert (t.size == x.shape[1]) or (t.size == 1)
        assert (x.shape == xd.shape)

        allP, allPd = self.getPnPdot(t)

        if t.size == 1:
            allP = np.tile(allP, (x.shape[1],1,1))
            allPd = np.tile(allPd, (x.shape[1],1,1))
        
        Vd = neinsum("in,nij,jn->n", 2.*x, allP, xd) + neinsum("in,nij,jn->n", x, allPd+self.optsCtrlDict_['minConvRate']*allP, x)
        
        if kd:
            Vd.resize((1,x.shape[1]))
        
        return Vd

    def convAng(self, x: np.ndarray, xd: np.ndarray, t: np.ndarray, kd: bool = True):

        """
        Returns the angle between the normal to the surface and the velocity
        :param x:
        :param xd:
        :param t:
        :param kd:
        :return:
        """

        t = narray(t, ndmin=1, dtype=nfloat)

        assert (x.shape == xd.shape)
        assert ((t.size == 1) or (x.shape[1] == t.size))

        # Get the quadratic function
        P = self.getPnPdot(t, returnPd=False)

        if t.size == 1:
            P = np.tile(P, (x.shape[1],1,1))

        # Project to get normal directions at x
        n = neinsum( "in,nij->jn", x, P) #TODO check if correct

        #Normalize
        n /= (norm(n,axis=0,keepdims=True)+coreOptions.floatEps)
        xdn = xd / (norm(xd,axis=0,keepdims=True)+coreOptions.floatEps)

        return np.arccos(nsum(-(nmultiply(n,xdn)), axis=0, keepdims=kd))

    
    def sphere2Ellip(self, t, x):
        t = narray(t, dtype=nfloat, ndmin=1)
        assert (t.size==1) or (t.size == x.shape[1])
        
        P = self.getPnPdot(t,False)
        
        if t.size == 1:
            y = ndot(inv(cholesky(P)), x)
        else:
            y = np.empty_like(x)
            for k in range(x.shape[1]):
                y[:,[k]] = ndot(inv(cholesky(P[k,:,:])), x[:,[k]])
        
        return y
    
    def ellip2Sphere(self, t, y):
        t = narray(t, dtype=nfloat, ndmin=1)
        assert (t.size == 1) or (t.size == y.shape[1])
    
        P = self.getPnPdot(t, False)
    
        if t.size == 1:
            x = ndot(cholesky(P), y)
        else:
            x = np.empty_like(y)
            for k in range(y.shape[1]):
                x[:, [k]] = ndot(cholesky(P[k, :, :]), y[:, [k]])
    
        return x
    
    def sphere2lvlSet(self, *args, **kwargs):
        return self.sphere2Ellip(*args, **kwargs)
    
    def lvlSet2Sphere(self, *args, **kwargs):
        return self.ellip2Sphere(*args, **kwargs)
    
    def lqrP(self, Q: np.ndarray, R: np.ndarray, x: np.ndarray = None, A: np.ndarray = None, B: np.ndarray = None, N: np.ndarray = None):
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
            A = self.dynSys.getTaylorApprox(x, 1, 1)[0]  # TODO change getTaylorApprox
            B = self.dynSys.getTaylorApprox(x, 0, 0)[1][0, :, :]  # Only zero order matrix

        # solve lqr
        if N is None:
            K, P, _ = lqr(A, B, Q, R)
        else:
            K, P, _ = lqr(A, B, Q, R, N)
        
        return P, K
    
    def getObjectivePoly(self, x0: np.ndarray = None, dx0: np.ndarray = None, fTaylor: np.ndarray = None, gTaylor: np.ndarray = None, uOpt: np.ndarray = None, idxCtrl: np.ndarray = None, t: float = 0., taylorDeg: int = 3):
        if __debug__:
            assert (fTaylor is None) == (gTaylor is None)
            assert (x0 is None) != (fTaylor is None)
            assert taylorDeg is not None
            assert uOpt is not None
        raise RuntimeError
        if dx0 is None:
            dx0 = self.refTraj.getDX(t)
        if fTaylor is None:
            fTaylor, gTaylor = self.dynSys.getTaylorApprox(x0, taylorDeg)
        
        objPoly = polynomial(this.repr)
        objPoly.coeffs[:] = 0.
        
        objPoly.coeffs = evalPolyLyap_Numba(self.P, self.repr.varNumsPerDeg[1], fTaylor, self.repr.varNumsUpToDeg[taylorDeg+1], gTaylor, self.repr.varNumsUpToDeg[taylorDeg+1], dx0, nrequire(np.repeat(self.repr.varNumsPerDeg[1], self.nq), dtype=nintu), uOpt, uMonom, self.idxMat, objPoly.coeffs, np.zeros_like(self.P))
        
        return objPoly
    
    def getObjectiveAsArray(self, fTaylor: np.ndarray = None, gTaylor: np.ndarray = None, taylorDeg: int = 3,
                            u: np.ndarray = None, uMonom: np.ndarray = None, x0: np.ndarray = None, dx0: np.ndarray = None, t: float = 0., zone=None):
        
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
            assert (x0 is None) != (t is None)
        
        if (x0 is None) == (fTaylor is None):
            if x0 is None:
                x0 = self.refTraj.getX(t)
            else:
                x0 = None
            
        
        
        if __debug__:
            assert (fTaylor is None) == (gTaylor is None)
            assert (x0 is None) != (fTaylor is None)
            assert taylorDeg is not None
            assert u is not None
        
        if zone is None:
            zone = self.getZone(t)
        
        if dx0 is None:
            dx0 = self.refTraj.getDX(t)
        if fTaylor is None:
            fTaylor, gTaylor = self.dynSys.getTaylorApprox(x0, taylorDeg)
        
        outCoeffs = nzeros((1+self.nu, self.nMonoms), nfloat)
        
        
        P,alphaTemp,Pdot = zone
        P /= alphaTemp
        
        # evalPolyLyapAsArray_Numba(P, monomP, f, monomF, g, monomG, dx0, monomDX0, u, monomU, idxMat, coeffsOut, Pdot)
        outCoeffs = evalPolyLyapAsArray_Numba(P, self.repr.varNumsPerDeg[1], fTaylor, self.repr.varNumsUpToDeg[taylorDeg], gTaylor,
                                              self.repr.varNumsUpToDeg[taylorDeg], dx0, nrequire(np.tile(self.repr.varNumsPerDeg[0], (self.nq,)), dtype=nintu),
                                              u, uMonom, self.idxMat, outCoeffs, Pdot)  # dx0 only depends on time, not on the position
        
        return outCoeffs
    
    def getCstrWithDeg(self, TorZone: Union[float, zone], gTaylor: np.ndarray, deg: int, which: np.ndarray = None, alwaysFull: bool = True):
        """
        Returns the polynomial constraints resulting from the taylor approximation of the dyn sys / or the pure polynomial dynamics
        deg: either 1 or 3 : linear or polynomial constraint
        which: defines for which control input, if not given then all are computed
        :param gTaylor:
        :param deg:
        :param which:
        :return:
        """
        
        try:
            t = float(TorZone)
            P = self.getPnPdot(t, False)
        except TypeError:
            if isinstance(TorZone, (list, tuple)):
                P = TorZone[0]/TorZone[1]
            else:
                P = TorZone
            if __debug__:
                assert P.shape == (self.nq, self.nq)
        
        taylorDeg = [len(aLNbr) for aLNbr in self.repr.varNumsUpToDeg].index( gTaylor.shape[0])
        
        if which is None:
            which = np.arange(0, self.nu)
        
        if alwaysFull:
            coeffsOut = nzeros((len(which), self.repr.nMonoms), dtype=nfloat)
        
        if deg == 1:
            PG0 = ndot(P, gTaylor[0, :, :])
            if alwaysFull:
                coeffsOut[which, self.repr.varNumsPerDeg[1][0]:self.repr.varNumsPerDeg[1][-1]+1] = PG0.T[which, :] #TODO change indexing this is not correct
            else:
                coeffsOut = PG0.T[which, :]  # Linear constraint each row of the returned matrix corresponds to the normal vector of a separating hyperplane
        else:
            # Return full matrix, each row corresponds to the coefficients of one polynomial constraint (for all monomials in the representation)
            # Due the matrix multiplication of P and each g
            PG = nmatmul(self.P, gTaylor)  # gTaylor is g[monom,i,j]
            compPolyCstr_Numba(self.repr.varNumsPerDeg[1], PG, self.repr.varNumsUpToDeg[taylorDeg], which, self.idxMat, coeffsOut)  # Expects coeffs to be zero
        
        return coeffsOut

    def getPolyCtrl(self, mode, t, x0=None, uRef=None, gTaylor=None, zone=None, alwaysfull=True):
        
        if x0 is None:
            x0 = self.refTraj.getX(t)
        if uRef is None:
            uRef = self.refTraj.getU(t)
        
        if gTaylor is None:
            _, gTaylor = self.dynSys.getTaylorApprox(x0)
        
        if zone is None:
            zone = self.getZone(t)

        assert mode[0] == 1, 'Only linear separation is currently implemented'

        #ctrlCoeffs_, ctrlMonoms_ = self.dynSys.ctrlInput.getU(mode[1], t, monomOut=True)
        ctrlCoeffs_, ctrlMonoms_ = self.dynSys.ctrlInput.getU2(mode[1], t, zone=zone, gTaylor=gTaylor, x0=x0, monomOut=True, uRef=uRef)
        if alwaysfull:
            ctrlCoeffs = nzeros((self.nu, self.repr.nMonoms), dtype=nfloat)
            ctrlCoeffs[:, ctrlMonoms_] = ctrlCoeffs_
        else:
            ctrlCoeffs = ctrlCoeffs_

        return ctrlCoeffs

    def getOptIdx(self, TorZone: Union[float, zone], gTaylor: np.ndarray, dX: np.ndarray, deg:int, x0=None):
        """
        
        :param TorZone: either a float to denote a time point or a zone (here (P, alpha, Pdot)
        :param gTaylor: Taylor expansion
        :param dX: Points to be evaluated
        :param deg: -1 means true nonlinear input dynamics
        :param x0: reference point. If None, dX is interpreted as offset, if given then dX is "absolute"
        :return:
        """
        
        try:
            t = float(TorZone)
            P = self.getPnPdot(t, False)
        except TypeError:
            if isinstance(TorZone, (list, tuple)):
                P = TorZone[0]/TorZone[1]
            else:
                P = TorZone
            if __debug__:
                assert P.shape == (self.nq, self.nq)
        
        if x0 is not None:
            dX = dX-x0
        
        if deg == -1:
            assert x0 is not None
            # Use true nonlinear dyn
            allPG = self.dynSys.gEval(dX) #[nPt, *g.shape]
            allPG = nmatmul(P, allPG)
            Y = neinsum( 'ji,ijk->ki', dX, allPG )#[nu, nPt] stacked matrix multiplication of ndot( dX[:,[k].T, allPG[k,:,:] )
        
        else:
            # First get the coefficients
            polyCoeffs = self.getCstrWithDeg(TorZone, gTaylor, deg)
            # polyCoeffs is [nu, nMonoms]
            Y = ndot(polyCoeffs, self.repr.evalAllMonoms(dX)) # [nu, npt]
        
        idx = -(Y>0.).astype(nint)
        idx[idx==0] = 1
        
        return idx
    
    def getCtrl(self, t, mode, dX:np.ndarray, x0=None, uRef=None, zone=None):
        """
        
        :param t: time point
        :param mode: what control law is to be applied; mode[0] -> Degree of separation surface; mode[1] -> Degree of control; Currently mode[1]
        has to be in [0,1,2] <-> [uref, uOptDiscrete, Linear feedback]
        :param dX: Offset between points and reference point
        :param x0: reference point
        :param zone: zone describing the Lyapunov levelset
        :return:
        """
        if zone is None:
            zone = self.getZone(t)
            
        if x0 is None:
            x0 = self.dynSys.ctrlInput.refTraj.getX(t)
        if uRef is None:
            uRef = self.dynSys.ctrlInput.refTraj.getU(t)

            
        fTaylor, gTaylor = self.dynSys.getTaylorApprox(x0)

        # Get the polynomial control laws
        ctrlDict = {}
        for aDeg in np.unique(mode[1]):
            if aDeg==1:
                ctrlDict[1] = self.getPolyCtrl((mode[0], 1*nones((self.nu,),dtype=nint)), t, x0=x0, uRef=uRef, gTaylor=gTaylor, alwaysfull=True)
                ctrlDict[-1] = self.getPolyCtrl((mode[0], -1*nones((self.nu,),dtype=nint)), t, x0=x0, uRef=uRef, gTaylor=gTaylor, alwaysfull=True)
            else:
                ctrlDict[aDeg] = self.getPolyCtrl((mode[0], aDeg*nones((self.nu,),dtype=nint)), t, x0=x0, uRef=uRef, gTaylor=gTaylor, alwaysfull=True)

        # Get the polyvalues
        dZ = self.repr.evalAllMonoms(dX)
        
        # Get the indices for optimal control
        idxOptPlus = self.getOptIdx(zone, gTaylor, dX, mode[0]) == 1
        
        # Compute
        U = nzeros((self.nu, dX.shape[1]), dtype=nfloat)
        
        for i in range(self.dynSys.nu):
            if mode[1][i] == 1:
                # Min/Max
                U[i, idxOptPlus[i, :]] = ndot(ctrlDict[1][[i],:], dZ[:,idxOptPlus[i, :]])
                U[i, np.logical_not(idxOptPlus[i, :])] = ndot(ctrlDict[-1][[i],:], dZ[:,np.logical_not(idxOptPlus[i, :])])
            else:
                U[i, :] = ndot(ctrlDict[mode[1][i]][[i],:], dZ)

        U = self.dynSys.ctrlInput(U)

        return U

    def analyzeSolSphereLinCtrl(self, thisSol, ctrlDictIn, opts):
        nq_ = self.nq
        nu_ = self.nu
    
        thisPoly = polynomial.polynomial(self.repr)  # Helper
    
        probDict_ = thisSol['origProb']['probDict']
    
        probList = []
    
        thisProbLin = thisSol['origProb']
        probList.append(thisProbLin)
    
        # Create the subset to exclude
        # There might be multiple critical points, in this case they have the same value for the primal objective
        # TODO -> attention to notation of ySol and xSol; Currently not really coherent
        allY = thisSol['ySol']
    
        # Test if any of the points is infeasible relying on optimal control
        PG0n = ctrlDictIn['PG0']/(norm(ctrlDictIn['PG0'], axis=0, keepdims=True)+coreOptions.floatEps)
        for k in range(allY.shape[1]):
            # TODO change to optimal
            thisY = allY[:, [k]]
            yPlaneDist = ndot(thisY.T, PG0n).reshape((nu_,))
            yPlaneSign = np.sign(yPlaneDist)
            yPlaneSign[yPlaneSign == 0] = 1
            yPlaneSign = np.require(yPlaneSign, dtype=nint)
        
            thisCoeffs = ctrlDictIn[-1][0].copy()
        
            for i in range(nu_):
                thisCoeffs += ctrlDictIn[i][-yPlaneSign[i]]
        
            # Get the poly and eval
            thisPoly.coeffs = -thisCoeffs
        
            if thisPoly.eval2(thisY) < opts['numericEpsPos']:
                # This point is not stabilizable using optimal control input
                if __debug__:
                    print(f"""Point \n{thisY}\n is not stabilizable with eps {opts["numericEpsPos"]}""")
                return []
    
        # Base prob
        thisProbBase = {'probDict':{'nPt':-1, 'solver':opts['solver'], 'dimsNDeg':(self.dynSys.nq, self.repr.maxDeg), 'nCstrNDegType':[]}, 'cstr':[]}
        # Copy the base constraint that confines to sphere
        thisProbBase['probDict']['nCstrNDegType'].append(probDict_['nCstrNDegType'][0])
        thisProbBase['cstr'].append(thisProbLin['cstr'][0].copy())
    
        thisProbBase['probDict']['resPlacement'] = thisSol['probDict']['resPlacement']
        for k in range(allY.shape[1]):
            thisY = allY[:, [k]]
        
            thisProb = dp(thisProbBase)
            thisProb['probDict']['critPointOrigin'] = {'y':thisY.copy(), 'strictSep':0, 'currU':thisSol['probDict']['u'] }
            
            thisProb['probDict']['isTerminal'] = 0  # Convergence can be improved but only by increasing the computation load
            thisProb['strictSep'] = 0
        
            thisCtrlType = nzeros((nu_, 1), dtype=nint)  # [None for _ in range(nu_, 1)]
        
            # Decide what to do for each critical point
            yPlaneDist = ndot(thisY.T, PG0n).reshape((nu_,))
        
            minDist =.9 # Has to be always strictly smaller than one otherwise linear control prob becomes infeasible  # np.Inf
        
            for i, iDist in enumerate(yPlaneDist):
                if iDist < -opts['minDistToSep']:
                    # Negative -> use maximal control input
                    thisCtrlType[i, 0] = 1
                    minDist = min(minDist, abs(iDist)) # Ensure that limiting sphere and control separation surface do not intersect
                elif iDist < opts['minDistToSep']:
                    # Linear control as iDist is self.opts['minDistToSep'] <= iDist < self.opts['minDistToSep']
                    thisCtrlType[i, 0] = 2
                else:
                    # Large positive -> use minimal control input
                    thisCtrlType[i, 0] = -1
                    minDist = min(minDist, abs(iDist)) # Ensure that limiting sphere and control separation surface do not intersect
                
            # Any point within the separation sphere can at most have the distance 2*minDist to any of the
            # separating hyperplanes;
            # Therefore we can safely scale the linear feedback gain by min(1., 1./(2*minDist))
            thisProb['probDict']["scaleFacK"] = max(1., 1./(2.*minDist))
            # Remember
            thisProb['probDict']['u'] = thisCtrlType.copy()
            thisProb['probDict']['minDist'] = minDist
            thisProb['probDict']['center'] = thisY.copy()
            
            # Rescale the control dict
            ctrlDict = rescaleLinCtrlDict(ctrlDictIn, thisProb['probDict']["scaleFacK"], True) #Rescale a deepcopy
            
            # Now we have the necessary information and we can construct the actual problem
            thisCoeffs = ctrlDict[-1][0].copy()  # Objective resulting from system dynamics
            if __debug__:
                thisProb[f"obj_-1,0"] = ctrlDict[-1][0].copy()
            for i, type in enumerate(thisCtrlType.reshape((-1,))):
                thisCoeffs += ctrlDict[i][type] # Part resulting from input dynamics and applying i-th control type
                if __debug__:
                    thisProb[f"obj_{i},{type}"] = ctrlDict[i][type].copy()
                    
            thisProb['obj'] = -thisCoeffs  # Inverse sign to maximize divergence <-> minimize convergence
            
            # get the sphere
            thisPoly.setEllipsoidalConstraint(thisY, minDist)
        
            # Confine the current problem to the sphere
            thisProb['probDict']['nCstrNDegType'].append((2, 's'))
            thisProb['cstr'].append(thisPoly.coeffs.copy())
            
            
            # Exclude the sphere from linear control prob
            thisProbLin['probDict']['nCstrNDegType'].append((2, 's'))
            thisProbLin['cstr'].append(-thisPoly.coeffs.copy())
            
            # Add the new problem
            if 0:
                # TODO this needs to move elsewhere.
                # It is useful but seriously messes up the way the problems refer to each other
                # Also attention that the new worst solution is respecting the new constraints
                newSol = dp(thisSol)
                newSol['ySol'] = newSol['ySol'][:,[k]]
                newSol['xSol'] = newSol['xSol'][:,[k]]
                # TODO check if sth is missing
                
                if self.checkProbFeasibility(newSol, thisProb, opts):
                    # The old point is now stabilizable -> Use "global" search
                    probList.append(thisProb)
                else:
                    # This point does not converge with respect to the Lyapunov function under the new control law
                    # -> Directly resplit the problem
                    thisSubProbList = self.analyzeSolSphereDiscreteCtrl(newSol, ctrlDictIn, opt) # Pass the unscaled version of the ctrlDict
                    if not thisSubProbList:
                        # Even when splitted this point remains infeasible -> proof of non-convergence -> return empty list
                        if __debug__:
                            print(f"The solution {newSol} cannot be stabilized -> early exit")
                    probList.extend(thisSubProbList)
            else:
                # Append the new problem
                probList.append(thisProb)
        
            # Done for one point
        # All sub-problems created
        return probList

    def analyzeSolSphereDiscreteCtrl(self, thisSol, ctrlDictIn, opts):
        
        # Deep copy control dict as the scaling of the linear feedback control law depends
        # on the size of the restricting hypersphere
        
        nq_ = self.nq
        nu_ = self.nu
    
        thisPoly = polynomial.polynomial(self.repr)  # Helper
    
        probDict_ = thisSol['origProb']['probDict']
    
        probList = []
        
        # Rescale without modifying the original
        ctrlDict = rescaleLinCtrlDict(ctrlDictIn, probDict_["scaleFacK"], True)
    
        #thisProbLin = thisSol['origProb'] # Replaced by all (suitable) combinations
        #probList.append(thisProbLin)
    
        # Create the subset to exclude
        # There might be multiple critical points, in this case they have the same value for the primal objective
        allY = thisSol['ySol']
    
        # Test if the solution can be improved
        if nall(np.logical_or(probDict_['u'].flatten() == -1, probDict_['u'].flatten() == 1)):
            # Optimal (discrete) control input was used for all inputs -> infeasible
            return []
    
        # Construct new problems
        # Heuristic: Find an input combination that ensures convergence but minimizes the number of separations
        # ensure correct input form (1d)
        PG0n = ctrlDict['PG0']/(norm(ctrlDict['PG0'], axis=0, keepdims=True)+coreOptions.floatEps)
        probDict_['u'] = probDict_['u'].reshape((-1,))
        uLinCurrent = np.flatnonzero(probDict_['u'] == 2) # The input indexes which can be "improved"
    
        # Distance to separating hyperplanes
        allYPlaneDist = [ndot(allY[:, [k]].T, PG0n).reshape((nu_,)) for k in range(allY.shape[1])]
        allYPlaneSign = [np.sign(a).astype(nint) for a in allYPlaneDist]

        # Distance to separating hypersurface
        allYSurfaceDist = [nzeros((nu_,), dtype=nfloat) for _ in range(allY.shape[1])]
        for k in range(allY.shape[1]):
            for i in range(nu_):
                thisPoly.coeffs = ctrlDict[i][1]  # Only the sign counts,
                allYSurfaceDist[k][i] = thisPoly.eval2(allY[:, [k]])
        allYSurfaceSign = [np.sign(a) for a in allYSurfaceDist]
        for a, b in zip(allYPlaneSign, allYSurfaceSign):
            # No zeros allowed
            a[a == 0] = 1
            b[b == 0] = 1
    
        allChangeSign = [a != b for a, b in zip(allYPlaneSign, allYSurfaceSign)]
    
        # check if convergence can be assured relying on the optimal input
        # Compute the convergence portion for system dynamics and each control input
        convDict = {}
        for aKey, aVal in ctrlDict.items():
            convDict[aKey] = {}
            try:
                for bKey, bVal in aVal.items():
                    thisPoly.coeffs = bVal
                    val = [float(thisPoly.eval2(allY[:, [k]])) for k in range(allY.shape[1])]
                    convDict[aKey].update({bKey:val})
            except:
                if __debug__:
                    print(f"Unable to eval {aKey} with {aVal}")
        # Compute convergence (Optimal control input with respect to separating hypersurface)
        allRes = nzeros((allY.shape[1],), dtype=nfloat)
        for k in range(allY.shape[1]):
            thisRes = convDict[-1][0][k]
            for i in range(nu_):
                thisRes += convDict[i][-allYSurfaceSign[k][i]][k]
            allRes[k] = thisRes
    
        # If any of allRes are positive -> proof for divergence (Attention sign!), return
        if np.any(allRes > -opts['numericEpsPos']):  # numericEps is negative; here the objective was not inversed; sign ok
            return []
    
        # Now check which input control can be kept linear feedback
        # Ensure that all points stay convergent, maximize number of linear controls
        linInputRanking = -np.Inf*nones(uLinCurrent.shape, dtype=nfloat)
    
        for i, ui in enumerate(uLinCurrent):
            for k in range(allY.shape[1]):
                thisDelta = convDict[ui][2][k]-convDict[ui][-allYSurfaceSign[k][ui]][k]  # This term is always positive
                if allRes[k]+thisDelta > -opts['numericEpsPos']:  # numericEps is negative; here the objective was not inverse; sign ok
                    linInputRanking[i] = -np.Inf
                # TODO check if this is really the best sol
                linInputRanking[i] = min(linInputRanking[i], -abs(thisDelta/allRes[k]))
    
        linInputRankingIdx = np.argsort(linInputRanking)
    
        keepOptCtrl = np.ones(uLinCurrent.shape, dtype=np.bool_)
    
        # Now take away as many separations as possible (greedy approach)
        # for i,ui in enumerate(uCurrent):
        for i, idxi in enumerate(linInputRankingIdx):
            ui = uLinCurrent[idxi]
            # Update the convergence
            for k in range(allY.shape[1]):
                allRes[k] += (convDict[ui][2][k]-convDict[ui][-allYSurfaceSign[k][ui]][k])
            # Check if still ok
            if np.any(allRes > -opts['numericEpsPos']):  # numericEps is negative; here the objective was not inverse; sign ok
                # Nope
                break
            else:
                keepOptCtrl[idxi] = False
    
        # Now we can assemble the new problems
        thisProbBase = dp(thisSol['origProb'])
        thisProbBase['probDict']['isTerminal'] = 0
    
        # Assemble all possible input
        inputProduct = [[aU] for aU in thisProbBase['probDict']['u']]
        inputSepDeg = nzeros((nu_,), dtype=nint)
    
        for aUi, aKeep in zip(uLinCurrent, keepOptCtrl):
            if aKeep:
                # Split the regions based on this input
                inputSepDeg[aUi] = self.dynSys.maxTaylorDeg if any(aC[aUi] for aC in allChangeSign) else 1  # If the sign does not
                # change ->
                # use hyperplane else use hypersphere
                inputProduct[aUi] = [-1, 1]
            else:
                # Nothing to do here
                pass
    
        inputs = itertools.product(*inputProduct)
    
        maxDegCstr = nmax(inputSepDeg)
        allYZ = self.repr.evalAllMonoms(allY, maxDegCstr) # Take only the necessary monomials
        thisPoly = polynomial.polynomial(self.repr) # Helper
    
        for i, input in enumerate(inputs):
            thisProb = dp(thisProbBase)
            probList.append(thisProb)
            #thisProb['probDict']['critPointOrigin'] = {'y':allY[:,[i]].copy(), 'strictSep':0, 'currU':thisSol['probDict']['u'] }
            # TODO check
            thisProb['probDict']['critPointOrigin'] = {'y':allY.copy(), 'strictSep':0, 'currU':thisSol['probDict']['u'] }
            
            # Which original crit point is contained
            isContained = nones((allY.shape[1],), dtype=np.bool_)
            
            thisCoeffs = ctrlDict[-1][0].copy()
            
            if __debug__:
                thisProb[f'obj_-1,0'] = ctrlDict[-1][0].copy()
        
            for k in range(nu_):
                thisCoeffs += ctrlDict[k][input[k]]
                
                if __debug__:
                    thisProb[f'obj_{k},{input[k]}'] = ctrlDict[k][input[k]].copy()
                    
            
                try:
                    idx = np.flatnonzero(uLinCurrent == k)[0]
                    if keepOptCtrl[idx]:
                        # Split
                        thisCoeffsSep = np.zeros_like(thisCoeffs)
                        thisCoeffsSep[self.repr.varNumsUpToDeg[inputSepDeg[k]]] = -ctrlDict[k][input[k]][self.repr.varNumsUpToDeg[inputSepDeg[k]]]  # Will be automatically rescaled
                        # Now we have split up the hypersphere into smaller parts
                        thisProb['probDict']['nCstrNDegType'].append((inputSepDeg[k], 's'))  # TODO adjust degree of relaxation if increasing degree of separating surface
                        thisProb['cstr'].append(thisCoeffsSep)
                        
                        # Store critpoints
                        # Check which (if any) of the critical points are contained in this regions
                        # -> that is if all constraints are positive
                        thisPoly.coeffs = thisCoeffsSep
                        isContained = np.logical_and(isContained, thisPoly.eval2(allYZ, maxDegCstr).reshape((allY.shape[1],))>=-coreOptions.absTolCstr)
                        
                except IndexError:
                    # Was already linear
                    pass
            thisProb['obj'] = -thisCoeffs
            thisProb['probDict']['u'] = narray(input, dtype=nint).reshape((-1,1))
            # Store origin points
            thisProb['probDict']['critPointOrigin'] = {'y':allY[:, isContained].copy(), 'strictSep':0, 'currU':thisSol['probDict']['u']}
            # Done
        return probList

    def analyzeSol(self, thisSol, ctrlDict, opts):
    
        if thisSol['origProb']['probDict']['isTerminal'] == -1:
            # Using linear control everywhere
            # We can simply put a new sphere around the minimal point
        
            if opts['sphereBoundCritPoint']:
                # assert thisSol['probDict']['nPt'] == -1
                if __debug__:
                    if ((thisSol['probDict']['resPlacementParent'] is None) and (nany(thisSol['probDict']['u'] != 2))):
                        raise RuntimeError
                    if opts['projection'] != 'sphere':
                        raise RuntimeError
                probList = self.analyzeSolSphereLinCtrl(thisSol, ctrlDict, opts)
            else:
                raise NotImplementedError
    
        elif thisSol['origProb']['probDict']['isTerminal'] == 0:
            # Here divergence was found within one of the hyperspheres
            # First we can check if the critical point converges for the optimal input
            #   -> If not : Terminal divergence is proven
            #   -> If so  : Set up new problems by splitting the zone
        
            if opts['sphereBoundCritPoint']:
                assert opts['projection'] == 'sphere'
                probList = self.analyzeSolSphereDiscreteCtrl(thisSol, ctrlDict, opts)
            else:
                raise NotImplementedError
    
        return probList
    
    def checkProbFeasibility(self, oldSol:dict, newProb:dict, opts:dict)->Union[bool, List[bool]]:
        """
        TODO maybe deprecated
        (Among others) Check if the minimizers of the old solution converge now in the new problem(s)
        :param newProb:
        :return:
        """
        
        thisPoly = polynomial(self.repr) #helper
        
        isFeas = [] #Return
        
        if isinstance(newProb, dict):
            newProb = [newProb]
        
        yStarOld = oldSol['ySol']
        zStarOld = self.repr.evalAllMonoms(yStarOld)
        idxOk = nones((yStarOld.shape[1],)).astype(np.bool_)
        for aProb in newProb:
            # We have to check first if some/all of the critical points are excluded
            for aCstr in aProb['cstr'][1:]:
                thisPoly.coeffs = aCstr
                idxOk = np.logical_and( idxOk, (thisPoly.eval2(zStarOld)>=0.).reshape((-1,)) )
            # Get the value
            thisPoly.coeffs = aProb['obj']
            isFeas.append( bool(nall( thisPoly.eval2(zStarOld[:,idxOk])  > -opts['numericEpsPos']) ) ) # numericEps is negative; here the objective was not inversed; sign ok
        
        if len(isFeas)==1:
            return isFeas[0]
        else:
            return isFeas
    
    def checkProbFeasibility1(self, oldSol:dict, newProb:dict, opts:dict)->bool:
        """
        Asserts if the old solution satisfies the new constraints and whether or not it is stabilizable under the
        new control input. This avoids unnecessary global polynomial optimizations.
        :param oldSol:
        :param newProb:
        :param opts:
        :return: bool
        """
        # ATTENTION: Currently 'ySol' is also the SCALED solution, that is, it is the point lying within the
        # (unit) hypersphere
        
        thisPoly = polynomial(self.repr)
        
        isFeas = True
        
        assert oldSol['ySol'].shape[1] == 1, "Only one point is expected here"
        yStarOld = oldSol['ySol']
        zStarOld = self.repr.evalAllMonoms(yStarOld)
        
        if __debug__:
            # Check if the point is inside all constraints
            for aCstr in newProb['cstr'][1:]: # First constraint is lasserre constraint
                thisPoly.coeffs = aCstr
                if thisPoly.eval2(zStarOld) <= -coreOptions.absTolCstr:
                    # Constraint violated -> this should not happen
                    raise RuntimeError("Violated constrainted that is suppossed to be ok")
        # Check if it is stabilizable according to the new objective
        # TODO is it a good idea to do a local search with respect to the new objective?
        thisPoly.coeffs = newProb['obj']
        return bool(thisPoly.eval2(zStarOld) >= -coreOptions.absTolCstr)

    def createBaseProb_(self, opts):
        return {'probDict':{'solver':opts['solver'], 'minDist':1., 'scaleFacK':1., 'dimsNDeg':(self.dynSys.nq, self.repr.maxDeg), 'nCstrNDegType':[]}, 'cstr':[]}

    def getLinProb_(self, ctrlDict, opts):
        # Only the linear control prob has to be created
        thisProbLin = self.createBaseProb_(opts)
        thisProbLin['probDict']['isTerminal']=-1 # Base case, non-convergence is "treatable" via critPoints

        # 1.1.1 Construct the objective
        thisPoly = polynomial.polynomial(self.repr) #Helper
        thisCoeffs = ctrlDict[-1][0].copy() # Objective resulting from system dynamics
        for k in range(self.dynSys.nu):
            thisCoeffs += ctrlDict[k][2] # Objective from input and linear control
        thisProbLin['obj'] = -thisCoeffs # Inverse sign to maximize divergence <-> minimize convergence
        # 1.2 Construct the constraints
        # 1.2.1 Confine to hypersphere
        thisPoly.setEllipsoidalConstraint(nzeros((2,1), dtype=nfloat), 1.)
        thisProbLin['probDict']['nCstrNDegType'].append( (2,'s') )
        thisProbLin['cstr'].append( thisPoly.coeffs )

        #Set information
        thisProbLin['probDict']['u'] = 2*nones((self.dynSys.nu,), dtype=nint)

        return thisProbLin

    def Proofs2Prob1(self, at: float, aZone: List, resultsLin: List, aSubProof: List[List[dict]], aCtrlDict: dict, opts: dict = {}):
        """
        # Get the corresponding problems; aSubProof containts only critical proofs
        :param at:
        :param aZone:
        :param resultsLin:
        :param aSubProof:
        :param aCtrlDict:
        :param opts:
        :return:
        """
        # Return simply the linear problem -> Do not take into account any info
        thisLinProb = self.getLinProb_(aCtrlDict, opts)
        probList = [thisLinProb]
        return [probList]

    
    def Proofs2Prob3(self, at:float, aZone:List, resultsLin:List, aSubProof:List[List[dict]], aCtrlDict:dict, opts:dict={}):
        """
        # Get the corresponding problems; aSubProof containts only critical proofs
        :param at:
        :param aZone:
        :param resultsLin:
        :param aSubProof:
        :param aCtrlDict:
        :param opts:
        :return:
        """
        
        thisLinProb = self.getLinProb_(aCtrlDict, opts)
        probList = [thisLinProb]
        
        # Now add all critical points
        for aSubProofList in aSubProof:
            for aProof in aSubProofList:
                # Map to sphere
                aProof['ySol'] = self.ellip2Sphere(at, aProof['critPoints']['yCrit']) # TODO this is really fucked up with the inconsistency between x and y
                aProof['origProb'] = thisLinProb
                newProbList = self.analyzeSolSphereLinCtrl(aProof, aCtrlDict, opts)
                # Add all except first which is the linear prob
                probList.extend(newProbList[1:])
        
        return [probList]
    
    def Proofs2Prob(self, at:float, aZone:List, resultsLin:List, aSubProof:List[List[dict]], aCtrlDict:dict, opts:dict={}):
        """
        # \brief Converts a subproof to a set of "suitable" problems, which are likely to reduce overall computation time
        """
        
        if self.opts_['zoneCompLvl'] == 1:
            return self.Proofs2Prob1(at, aZone, resultsLin, aSubProof, aCtrlDict, opts)
        elif self.opts_['zoneCompLvl'] == 2:
            raise NotImplementedError
        elif self.opts_['zoneCompLvl'] == 3:
            return self.Proofs2Prob3(at, aZone, resultsLin, aSubProof, aCtrlDict, opts)
        else:
            raise RuntimeError

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

lyapunovFunctions = tuple(list(lyapunovFunctions)+[quadraticLyapunovFunction, quadraticLyapunovFunctionTimed])


