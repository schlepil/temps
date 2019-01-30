from coreUtils import *

#setIdxNumba(idx, idxU, self.thisLimL, self.thisLimU, uRef)
@njit
def setIdxNumba(idx, u, limL, limU, uRef):
    
    for k, aIdx in enumerate(idx):
        if aIdx == -1:
            u[k,0] = limL[k,0]
        elif aIdx == 0:
            u[k,0] = uRef[k,0]
        else:
            u[k,0] = limU[k,0]
    
    return None