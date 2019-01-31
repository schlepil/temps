from coreUtils import *
from polynomial import polynomialRepr


class dynamicalSystem:
    
    def __init__(self, repr:polynomialRepr, maxTaylorDeg:int, ):
        assert maxTaylorDeg<=polynomialRepr.maxDeg
        
        self.repr = repr
        self.maxTaylorDeg = maxTaylorDeg
        
    
    def getTaylorApprox(self, x:np.ndarray, maxDeg:int)->Tuple:
        raise NotImplementedError
    
    def precompute(self):
        raise NotImplementedError
    
    def __call__(self, x:np.ndarray, u:np.ndarray, mode:str='OO', x0:np.ndarray=None):
        raise NotImplementedError
        