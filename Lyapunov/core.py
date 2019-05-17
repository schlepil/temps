from polynomial import *
from dynamicalSystems import dynamicalSystem

from itertools import product

# Dummy for typing
class zone:
    pass


class LyapunovFunction():
    
    def __init__(self, degLyap:int, dynSys:dynamicalSystem):
        
        self.dynSys = dynSys
        
        self.nq = self.dynSys.nq
        self.nu = self.dynSys.nu
        
        self.repr = self.dynSys.repr
        self.listOfMonomialsAsInt = self.dynSys.repr.listOfMonomialsAsInt
        self.nMonoms = self.dynSys.repr.nMonoms
        self.idxMat = self.dynSys.repr.idxMat
        self.num2monom = self.dynSys.repr.num2monom
        self.monom2num = self.dynSys.repr.monom2num
        self.refTraj = self.dynSys.ctrlInput.refTraj
        self.degLyap = degLyap

    def getPnPdot(self, t: np.ndarray, returnPd = True):
        raise NotImplementedError
    
    def getObjectivePoly(self,x0:np.ndarray=None,dx0:np.ndarray=None,fTaylor:np.ndarray=None, gTaylor:np.ndarray=None, uOpt:np.ndarray=None, idxCtrl:np.ndarray=None, t:float=0., taylorDeg:int=3):
        raise NotImplementedError
    
    def getCstrCoeffs(self,x0:np.ndarray,PG0:np.ndarray=None, cstrDegList:np.ndarray=None):
        raise NotImplementedError

    def getObjectiveAsArray(self, fTaylor: np.ndarray = None, gTaylor: np.ndarray = None, taylorDeg: int = 3, u: np.ndarray = None, uMonom: np.ndarray = None, x0: np.ndarray = None, dx0: np.ndarray = None, t: float = 0., P=None, Pdot=None):
        raise NotImplementedError
    
    def getLyap(self, t):
        raise NotImplementedError
    
    def getZone(self,t):
        raise NotImplementedError

    def plot(self, ax: "plot.plt.axes", t: float = 0.0, opts = {}):
        raise NotImplementedError
    
    def getOptIdx(self, TorZone:Union[float, zone], gTaylor:np.ndarray, X:np.ndarray, deg:int):
        raise NotImplementedError
    
    def getCtrl(self, t, mode, dX:np.ndarray, x0:None, zone=None):
        raise NotImplementedError
    
    def getCtrl(self, t, mode, dX:np.ndarray, x0:None, zone=None):
        raise NotImplementedError

    def evalV(self, x: np.ndarray, t: np.ndarray, kd: bool = True):
        raise NotImplementedError
    
    def evalVd(self, x: np.ndarray, xd: np.ndarray, t: np.ndarray, kd: bool = True):
        raise NotImplementedError

    def convAng(self, x: np.ndarray, t: np.ndarray, kd: bool = True):
        raise NotImplementedError

lyapunovFunctions = ()