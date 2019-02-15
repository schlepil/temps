from coreUtils import *
from polynomial import *
from dynamicalSystems import dynamicalSystem


class LyapunovFunction():
    
    def __init__(self, dynSys:dynamicalSystem):
        
        self.dynSys = dynSys
        
        self.nq = self.dynSys.nq
        self.nu = self.dynSys.nu
        
        self.repr = self.dynSys.repr
        self.listOfMonomialsAsInt = self.dynSys.repr.listOfMonomialsAsInt
        self.nMonoms = self.dynSys.repr.nMonoms
        self.idxMat = self.dynSys.repr.idxMat
        self.self.dynSys.repr = self.dynSys.repr.num2monom
        self.monom2num = self.dynSys.repr.monom2num
        self.refTraj = self.dynSys.ctrlInput.refTraj
    
    def getObjectivePoly(self,x0:np.ndarray=None,dx0:np.ndarray=None,fTaylor:np.ndarray=None, gTaylor:np.ndarray=None, uOpt:np.ndarray=None, idxCtrl:np.ndarray=None, t:float=0., taylorDeg:int=3):
        raise NotImplementedError
    
    def getCstrCoeffs(self,x0:np.ndarray,PG0:np.ndarray=None, cstrDegList:np.ndarray=None):
        raise NotImplementedError
    
    
    
        