from coreUtils import *

class referenceTrajectory:
    def __int__(self, nx:int, nu:int, tMin:float, tMax:float):
        self.nx = nx
        self.nu = nu
        self.tLims = [tMin, tMax]
    
    def __call__(self, t:float):
        return self.getX(t),self.getXD(t),self.getU(t)
    def getU(self,t:float):
        raise NotImplementedError
    def getX(self,t:float):
        raise NotImplementedError
    def getDX(self,t:float):
        raise NotImplementedError