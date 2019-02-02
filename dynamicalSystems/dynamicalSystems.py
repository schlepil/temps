from coreUtils import *
from polynomial import polynomialRepr

class dynamicalSystem:
    
    def __init__(self, repr:polynomialRepr, q, u, maxTaylorDegree):
        
        if __debug__:
            assert q.shape[1] == 1
            assert u.shape[1] == 1
            assert maxTaylorDegree <= repr.maxDeg
        
        self.repr = repr
        
        self.q = q
        self.u = u
        
        self.nq = int(q.shape[0])
        self.nu = int(u.shape[0])
        
        self.maxTaylorDeg=maxTaylorDegree
        
    
    def getTaylorApprox(self, x:np.ndarray, maxDeg:int)->Tuple:
        raise NotImplementedError
    
    def precompute(self):
        raise NotImplementedError
    
    def __call__(self, x:np.ndarray, u:np.ndarray, mode:str='OO', x0:np.ndarray=None):
        raise NotImplementedError

class secondOrderSys(dynamicalSystem):
    
    def __init__(self, repr:polynomialRepr, massMat:sy.Matrix, f:sy.Matrix, g:sy.Matrix, q:"symbols", u:"symbols", maxTaylorDegree:int=3, file:str=None):
        """
        represents a system of the form
        q is composed of (position,velocity) -> (q_p,q_v)
        dq -> (dq_p,dq_v) = (q_v,q_a)
        q_v = dq_p
        massMat(q_p).q_a = f(q) + g(q).u
        :param repr:
        :param massMat:
        :param f:
        :param g:
        :param q:
        :param u:
        """
        
        super(secondOrderSys,self).__init__(repr,q,u,maxTaylorDegree)
        
        self.massMat = massMat
        self.f = f
        self.g = g
        
        if file == None:
            self.compTaylor()
        else:
            self.fromFile(file)
    
    def compTaylor(self):
        self.compTaylorF()
        self.compTaylorG()
        self.compTaylorM()
    
    def compTaylerF(self):
        
        nMonomPerDeg = [len(aMonomList) for aMonomList in self.repr.listOfMonomialsPerDeg]
        
        taylorFD ={ 'f0':sy.Matrix(nzeros((self.nq,1))) }
        
        taylorFD['f0'] += self.f
        
        cFac = 1.
        lastMat = self.f
        for k in range(1,self.maxTaylorDeg+1):
            thisMat = sy.Matrix(nzeros((self.nq,len(self.repr.listOfMonomialsPerDeg[k]))))

            #Get the derivative corresponding to each monomial
            for j,aMonom in enumerate(self.repr.listOfMonomialsPerDeg[k]):
                idxParent, idxDeriv = getParentAndDeriv(aMonom, self.repr.listOfMonomialsPerDeg[k-1])
                
                #Loop over each element
                for i in range(self.nq):
                    thisMat[i,j] = sy.diff(lastMat[i,idxParent], self.q[idxDeriv,0])/float(k)
            
            #save
            taylorFD["f{0:d}".format(k)] = thisMat
        
        self.taylorFD = variableStruct(**taylorFD)
                    
                
            
                    
                    
        
        
        
        
    