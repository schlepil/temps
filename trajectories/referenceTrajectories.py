from coreUtils import *

class referenceTrajectory:
    def __int__(self, nx:int, nu:int, tMin:float, tMax:float):
        
        assert tMax>tMin
        
        self.nx = nx
        self.nu = nu
        self.tLims = [tMin, tMax]
    
    def __call__(self, t:float, doRestrict:bool=True):
        t = self.checkTime(t,doRestrict)
        return self.getX(t, False),self.getDX(t, False),self.getU(t, False)
    def checkTime(self,t,restrict=True):
        if __debug__:
            if nany(t>self.tLims[1]):
                print("time larger then final time")
            if nany(t<self.tLims[0]):
                print("time smaller then start time")
        if restrict:
            t = nminimum(self.tLims[1], nmaximum(self.tLims[0],t))
        return t
    def getU(self,t:float):
        raise NotImplementedError
    def getX(self,t:float):
        raise NotImplementedError
    def getDX(self,t:float):
        raise NotImplementedError


class analyticTrajectory(referenceTrajectory):
    
    def __init__(self, fX:Callable, fU:Callable, nx:int, nu:int, fXd:Callable=None, tMin:float=0., tMax:float=1. ):
        
        super(analyticTrajectory, self).__int__(nx, nu, tMin, tMax)
        
        self._dt = 1e-6*(self.tLims[1]-self.tLims[0])
        
        self.fX = fX
        self.fU = fU
        self.fXd = fXd
    
    def getU(self,t:float, doRestrict:bool=True):
        t=self.checkTime(t, doRestrict)
        return self.fU(t)
    def getX(self,t:float, doRestrict:bool=True):
        t = self.checkTime(t, doRestrict)
        return self.fX(t)
    def getDX(self,t:float, doRestrict:bool=True):
        t = self.checkTime(t, doRestrict)
        if self.fXd is None:
            #Use simple forward finite differences if possible
            if t+self._dt <= self.tLims[1]:
                return (self.fX(t+self._dt)-self.fX(t))/self._dt
            else:
                return (self.fX(t)-self.fX(t-self._dt))/self._dt
        else:
            return self.fXd(t)
        
class omplTrajectory(referenceTrajectory):

    def __init__(self, dynF:Callable, fileName:str, nx:int, nu:int, tMin:float=0., tMax:float=1.):

        super(omplTrajectory, self).__init__(nx, nu, tMin, tMax)

        self.X = np.load( f"{fileName}_X.npy" )
        self.U = np.load( f"{fileName}_U.npy" )
        self.t = np.load( f"{fileName}_T.npy" )

        assert(self.tLims[0]>=self.t[0])
        assert (self.tLims[1] <= self.t[1])

        self.dynF = dynF # It is better to compute the derivative using the interpolated position and control input

        self.xrefI = sp.interpolate.PchipInterpolator(self.t, self.X, axis=1)

        # OMPL uses piecewise constant inputs, if trajectory was preprocessed other interpolators can be used
        # self.urefI = sp.interpolate.PchipInterpolator(self.t, self.U, axis=1)

    def getU(self,t:float, doRestrict:bool=True):
        t=self.checkTime(t, doRestrict)
        return self.urefI(t).reshape((self.nu,-1))
    def getX(self,t:float, doRestrict:bool=True):
        t = self.checkTime(t, doRestrict)
        return self.fX(t).reshape((self.nx,-1))
    def getDX(self,t:float, doRestrict:bool=True):
        t = self.checkTime(t, doRestrict)
        return self.dynF(self.getX(t,False), self.getU(t,False), t).reshape((self.nx,-1))


