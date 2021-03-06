from coreUtils import *
from dynamicalSystems.inputs_numba import *
from trajectories import *

from polynomial import polynomialRepr

class inputConstraints:
    def __init__(self, repr:polynomialRepr, refTraj:referenceTrajectory, nu:int):
        self.repr = repr
        self.refTraj = refTraj
        self.nu = nu
    
    def getMinU(self, *args, **kwargs):
        raise NotImplementedError
    def getMaxU(self, *args, **kwargs):
        raise NotImplementedError
    
    def getU(self, idx:np.ndarray, *args, **kwargs):
        raise NotImplementedError

    def getU2(self, idx:np.ndarray, t, zone, gTaylor, x0, monomOut=False, uRef=None, *args, **kwargs):
        raise NotImplementedError
    
    def getBounds(self, *args, **kwargs):
        return self.getMinU(*args, **kwargs), self.getMaxU(*args, **kwargs)

    def computeCtrl(self, t:float, XX:np.ndarray, mode:int, isOffset:bool):
        raise NotImplementedError

    def __call__(self,*args,**kwargs):
        raise NotImplementedError
    
class boxInputCstr(inputConstraints):
    def __init__(self, repr:polynomialRepr, refTraj:referenceTrajectory, nu:int,limL=None,limU=None):
        """
        Class implementing input constraints
        Box constraints limit the inputs to a hyperrectangle
        :param nu:
        :param limL:
        :param limU:
        """
        super(boxInputCstr,self).__init__(repr,refTraj,nu)
        
        self.limLCall = hasattr(limL,"__call__")
        self.limUCall = hasattr(limU,"__call__")
        
        # Save values if limL is not callable -> not a function of time
        if not self.limLCall:
            self.limL = -np.ones((self.nu,1),dtype=nfloat) if limL is None else limL
            self.limL = np.array(limL).reshape((nu,1))
            self.thisLimL = self.limL
        # Save values if limL is not callable -> not a function of time
        if not self.limUCall:
            self.limU = np.ones((self.nu,1),dtype=nfloat) if limU is None else limU
            self.limU = np.array(limU).reshape((nu,1))
            self.thisLimU = self.limU

    #######################
    def getMinU(self,t:float=0.)->np.ndarray:
        # Get current lower lim
        if self.limLCall:
            self.thisLimL = self.limL(t)
        return self.thisLimL
    
    #######################
    def getMaxU(self,t:float=0.)->np.ndarray:
        # get current upper lim
        if self.limUCall:
            self.thisLimU = self.limU(t)
        return self.thisLimU
    
    #######################
    def getU(self,idx:np.ndarray,t=0.,uRef=None,uOut=None, monomOut=False, *args, **kwargs):
        # Get optimal input encoded by index vector idx
        # 1 is maximum input
        # 0 is reference input (must be given if occuring!
        # -1 is minimal input
        try:
            idx = idx.copy().reshape((-1,))
        except (AttributeError, TypeError):
            idx = narray(idx, ndmin=1, dtype=nint)

        if __debug__:
            assert idx.dtype in (nintu,nint)
            assert len(idx.shape) == 1
            assert (uRef is not None) and (not np.any(idx==0))
            assert (uOut is None) or (uOut.shape == (self.nu,1))
        
        uOut = np.zeros((self.nu,1)) if uOut is None else uOut
        
        if (uRef is None) and (nany(idx==0)):
            uRef = self.refTraj.getU(t)
        
        #set up new limits
        if self.limLCall:
            self.thisLimL = self.limL(t)
        if self.limUCall:
            self.thisLimU = self.limU(t)
            
        setIdxNumba(idx,uOut,self.thisLimL,self.thisLimU,uRef)
        if not monomOut:
            return uOut
        else:
            return uOut,self.repr.varNumsPerDeg[0]

    def getUVal(self, X:np.ndarray, idx:np.ndarray, t=0., zone=None, gTaylor=None, uRef=None, x0:np.ndarray=None, *args, **kwargs):
        """
        Compute the actual values
        :param X:
        :param idx:
        :param t:
        :param uRef:
        :param x0:
        :param args:
        :param kwargs:
        :return:
        """
        # If x0 is given, then X is in absolute coords

        assert zone is not None
        assert gTaylor is not None

        if x0 is None:
            DX = X
        else:
            DX = X-x0

        Z = self.repr.evalAllMonoms(DX)

        ctrlCoeffs_, ctrlMonoms_ = self.getU2(idx, t, zone, gTaylor=gTaylor, x0=self.refTraj.getX(t),monomOut=True, uRef=uRef)

        ctrlCoeffs = nzeros((self.nu, self.repr.nMonoms), dtype=nfloat)
        ctrlCoeffs[:,ctrlMonoms_] = ctrlCoeffs_

        u = ndot(ctrlCoeffs, Z)

        u=self(u)

        return u
    
    #######################
    def __call__(self,inputNonCstr,t:float=0.):
        # Limit given input array
        t = np.array(t)
        if t.size == 1:
            if self.limLCall:
                self.thisLimL = self.limL(t)
            if self.limUCall:
                self.thisLimU = self.limU(t)
            # inputNonCstr.resize((inputNonCstr.size,))
            inputNonCstr = np.maximum(self.thisLimL,np.minimum(self.thisLimU,inputNonCstr))
        elif t.size == inputNonCstr.shape[1]:
            if (self.limLCall or self.limUCall):
                for k,aT in enumerate(t):
                    if self.limLCall:
                        self.thisLimL = self.limL(aT)
                    if self.limUCall:
                        self.thisLimU = self.limU(aT)
                    inputNonCstr[:,k] = np.maximum(self.thisLimL,np.minimum(self.thisLimU,inputNonCstr[:,k]))
                    #inputNonCstr[:,k] = self(inputNonCstr[:,k], t=aT)#Significantly slower
            else:
                inputNonCstr = np.maximum(self.thisLimL,np.minimum(self.thisLimU,inputNonCstr))
        
        return inputNonCstr
    


class boxInputCstrNoised(boxInputCstr):
    def __init__(self, nuCtrl, nuNoise, limL=None, limU=None):
        """
        Implemements boxed control and noise
        noise is treated like an inversed control -> always seeks to make the system deviate
        Attention: this is not treated explicitely but must be taken into account directly by limL and limU, that is LimL[nuCtrl:,0] >= LimU[nuCtrl:,0]
        :param nuCtrl:
        :param nuNoise:
        :param limL:
        :param limU:
        """
        
        if __debug__:
            assert nuCtrl >=0
            assert nuNoise >= 0
        
        limLUisNone = (limL is None, limU is None)
        
        super(boxInputCstrNoised, self).__init__(nuCtrl+nuNoise, limL, limU)
        
        if limLUisNone[0]:
            self.limL[nuCtrl:,0] *= -1.
            self.thislimL = self.limL
        if limLUisNone[1]:
            self.limU[nuCtrl:,0] *= -1.
            self.thislimU = self.limU

class boxInputCstrLFBG(boxInputCstr):
    """
    Class implementing box constraint input with partially linear feedback control
    """
    
    def __init__(self, repr:polynomialRepr, refTraj:referenceTrajectory, nu:int, limL=None, limU=None):
        super(type(self), self).__init__(repr, refTraj, nu,limL,limU)
    
    def getU(self,idx:np.ndarray,t=0.,uRef:np.ndarray=None,uOut:np.ndarray=None, P:np.ndarray=None, PG0:np.ndarray=None, alpha:float=None,
             monomOut=False, scale:float=1., *args, **kwargs):
        """
        Computes the optimal bang-bang control or the optimal and scaled linear feedback law
        We define:
        # Get optimal input encoded by index vector idx
        # 2 is linear feedback
        # 1 is maximum input
        # 0 is reference input (must be given if occuring!
        # -1 is minimal input
        :param idx:
        :param t:
        :param uRef:
        :param uOut:
        :param PG0:
        :param scale:
        :return:
        """

        try:
            idx = idx.copy().reshape((-1,))
        except (AttributeError, TypeError):
            idx = narray(idx, ndmin=1, dtype=nint)
        
        if __debug__:
            assert (P is None) or (P.shape[0]==P.shape[1])
            assert ((P is None) and (PG0 is None)) or ((P is not None) and (PG0 is not None))
            assert np.all(idx<2) or (PG0 is not None)
            assert (alpha is None) or (alpha > 0.)
        
        if alpha is not None:
            P /= alpha
        
        Ci = inv(cholesky(P))
        
        nq = 0 if PG0 is None else P.shape[0]
        
        #First column of uout is
        uOut = np.zeros((self.nu,1+nq)) if uOut is None else uOut
        
        if uRef is None:
            uRef = self.refTraj.getU(t)
        
        if __debug__:
            assert uOut.shape[1] == 1+nq

        # First get the bang bang optimal control
        # This also computes the current lower and upper bound
        uOut[:,[0]] = super(type(self),self).getU(idx,t,uRef)
        
        #Check if linear feedback control is needed
        # Omega is defined as x'.P.x <= alpha
        # The separating hyperplanes are stored in PG0 (the ith column vector corresponds to the ith hyperplane
        for k,aIdx in enumerate(idx):
            if aIdx==2:
                # Linear feedback is demanded for this input
                # smallest allowable abs input
                if __debug__:
                    assert self.thisLimL[k,0] <= uRef[k,0] <= self.thisLimU[k,0]
                uMaxAbsK = min(uRef[k,0]-self.thisLimL[k,0],self.thisLimU[k,0]-uRef[k,0])
                #ki = -PG0[:,[k]].T/norm(PG0[:,k],ord=2,axis=0)#normalise
                #scale such that uOut[k] never exceeds limits
                #ki *= uMaxAbsK*(mndot([ki,P,ki.T])/alpha)**.5
                # The version above is wrong
                # In fact we have to solve the following problem:
                # max ki.x
                # s.t x.T.P.x <= 1
                # Pose y = C.x with P = C.T.C; Ci = inv(C)
                # max ki.Ci.y
                # s.t. y.T.y <= 1
                # -> 2-Norm y is 1.
                # -> maximum attained for y and (ki.Ci).T parallel
                ki = -PG0[:,[k]].T/norm(PG0[:,k],ord=2,axis=0)#normalise
                kTildeNorm = norm(ndot(ki, Ci), axis = 1)
                ki *= uMaxAbsK/kTildeNorm
                
                uOut[k,0]=uRef[k,0]
                uOut[[k],1:]=ki
        
        if nq != 0:
            uOut[:,1:]*=scale

        if not monomOut:
            return uOut
        else:
            return uOut, self.repr.varNumsUpToDeg[0 if nq==0 else 1]

    #######################

    def getU2(self, idx:np.ndarray,t,zone, gTaylor, x0, monomOut=False, uRef=None):
        """

        :param idx:
        :param t:
        :param zone:
        :param gTaylor:
        :param x0:
        :param monomOut:
        :return:
        """

        #Test if quadratic zone
        try:
            if isinstance(zone, (list, tuple)):
                P,alpha,Pd = zone
                assert (P.shape == (x0.size, x0.size)) and (Pd.shape == (x0.size, x0.size))
                P=P/alpha
                alpha=1.
                zoneType = 'q'
            else:
                P = zone
                assert (P.shape == (x0.size, x0.size))
        except:
            print("Could not determine zone")
            raise NotImplementedError

        # Defaults
        uRef = self.refTraj.getU(t) if uRef is None else uRef


        if zoneType == 'q':
            ctrlCoeffs_, ctrlMonoms_ = self.getU(idx, t, uRef, P=P, PG0=ndot(P, gTaylor[0,:,:]), alpha=alpha, monomOut=True)
        else:
            raise NotImplementedError

        if monomOut:
            return ctrlCoeffs_, ctrlMonoms_
        else:
            ctrlCoeffs = nzeros((self.nu, self.repr.nMonoms), dtype=nfloat)
            ctrlCoeffs[:,ctrlMonoms_] = ctrlCoeffs_
            return ctrlCoeffs

    #######################
    def getPolyCtrl(self, aDeg, t, x0, gTaylor, zone, alwaysfull=True, scale=1., limitEqual=False):

        """

        :param aDeg:
        :param t:
        :param x0:
        :param gTaylor:
        :param zone:
        :param alwaysfull:
        :param scale:
        :param limitEqual:
        :return:
        """
        
        #Test if quadratic zone
        try:
            if isinstance(zone, (list, tuple)):
                P,alpha,Pd = zone
                assert (P.shape == (x0.size, x0.size)) and (Pd.shape == (x0.size, x0.size))
                P=P/alpha
                zoneType = 'q'
            else:
                P = zone
                assert (P.shape == (x0.size, x0.size))
        except:
            print("Could not determine zone")
            raise NotImplementedError
        
        # Get the input control limits
        uRef = self.refTraj.getU(t)
        deltaUMaxAbs = nmin(uRef-self.getMinU(t), self.getMaxU(t)-uRef)
        
        
        if zoneType == 'q':
            if aDeg == 1: # Linear control based on linear seperation
                K = -ndot(P,gTaylor[0,:,:]).T.copy()
                #Normalize
                K /= norm(K, ord=2, axis=1, keepdims=True)
                # Limit
                # scale such that uOut[k] never exceeds limits
                minScale = np.Inf
                for i in range(self.nu):
                    scaleI = deltaUMaxAbs[i]*float(mndot([K[[i],:],P,K[[i],:].T])**.5)
                    if limitEqual:
                        minScale = min(minScale, scaleI)
                    else:
                        K[i,:] *= scaleI
                if limitEqual:
                    K *= minScale
                # Global additional scaling for safety
                K *= scale

                
                if alwaysfull:
                    ctrlCoeffs = nzeros((self.nu, self.repr.nMonoms), dtype=nfloat)
                    ctrlCoeffs[:, self.repr.varNumsPerDeg[1]] = K
                else:
                    ctrlCoeffs = K
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            
        return ctrlCoeffs
            
        
            
        
    
    
            
        
        
        