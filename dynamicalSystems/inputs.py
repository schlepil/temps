from coreUtils import *
from dynamicalSystems.inputs_numba import *


class inputConstraints:
    def __init__(self):
        pass
    
    def getMinU(self, *args, **kwargs):
        raise NotImplementedError
    def getMaxU(self, *args, **kwargs):
        raise NotImplementedError
    
    def getU(self, idx:np.ndarray, *args, **kwargs):
        raise NotImplementedError
    
    def getBounds(self, *args, **kwargs):
        return self.getMinU(*args, **kwargs), self.getMaxU(*args, **kwargs)

    def __call__(self,*args,**kwargs):
        raise NotImplementedError
    
class boxInputCstr(inputConstraints):
    def __init__(self,nu,limL=None,limU=None):
        """
        Class implementing input constraints
        Box constraints limit the inputs to a hyperrectangle
        :param nu:
        :param limL:
        :param limU:
        """
        super(boxInputCstr,self).__init__()
        
        self.nu = nu
        
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
    
    def getU(self,idx:np.ndarray,t=0.,uRef=None, idxU=None):
        # Get optimal input encoded by index vector thisInd
        # 1 is maximum input
        # 0 is reference input (must be given if occuring!
        # -1 is minimal input
        
        if __debug__:
            assert idx.dtype in (nintu,nint)
            assert len(idx.shape) == 1
            assert (uRef is not None) and (not np.any(idx==0))
            assert (idxU is None) or (idxU.shape == (self.nu,1))
        
        idxU = np.zeros((self.nu,1)) if idxU is None else idxU
        
        #set up new limits
        if self.limLCall:
            self.thisLimL = self.limL(t)
        if self.limUCall:
            self.thisLimU = self.limU(t)
            
        setIdxNumba(idx, idxU, self.thisLimL, self.thisLimU, uRef)
        
        return idxU
    
    #######################
    def __call__(self,inputNonCstr,t=0):
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
            
        
        
        