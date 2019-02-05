from coreUtils import *
from polynomial import polynomialRepr

from polynomial.utils_numba import getIdxAndParent

import string

from dynamicalSystems.derivUtils import getInverseTaylorStrings, compTaylorExp

class dynamicalSystem:
    
    def __init__(self, repr:polynomialRepr, q, u, maxTaylorDegree):
        
        if __debug__:
            assert q.shape[1] == 1
            assert u.shape[1] == 1
            assert maxTaylorDegree <= repr.maxDeg
            assert repr.nDims == q.shape[0]
        
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
        
        self.nqv = self.nq//2
        
        if file == None:
            self.compTaylor()
        else:
            self.fromFile(file)

        return None
    
    def compTaylor(self):
        self.compTaylorF()
        self.compTaylorG()
        self.compTaylorM()
        if self.maxTaylorDeg<=3:
            derivStrings = string.ascii_lowercase[23:23+self.maxTaylorDeg]
        else:
            derivStrings = string.ascii_lowercase[23:23+self.maxTaylorDeg]
        self.inversionTaylor = variableStruct(**getInverseTaylorStrings('M','Mi','W',derivStrings))

        return None
        
    def compTaylorF(self):
        
        taylorFD = compTaylorExp(self.f, 'f', self.q, False, self.maxTaylorDeg, self.repr)

        self.taylorF = variableStruct(**taylorFD)

        return None
    
    def compTaylorM(self):
        """
        Taylor expansion of the mass matrix
        
        :return:
        """

        taylorMD = compTaylorExp(self.massMat, 'M', self.q, True, self.maxTaylorDeg, self.repr)


        self.taylorM = variableStruct(**taylorMD)

    def compTaylorG(self):
        """
        Taylor expansion of the input matrix

        :return:
        """

        taylorGD = compTaylorExp(self.g, 'G', self.q, True, self.maxTaylorDeg, self.repr)
    
        self.taylorG = variableStruct(**taylorGD)

        return None
    
    def evalTaylor(self, x:np.ndarray, maxDeg:int=None):
        
        # TODO this is a naive implementation
        
        if __debug__:
            assert (maxDeg is None) or (maxDeg <= self.maxTaylorDeg)
        
        maxDeg = self.maxTaylorDeg if maxDeg is None else maxDeg
        
        # return Value
        MifTaylor = nzeros((self.nq, self.repr.nMonoms), dtype=nfloat)
        MigTaylor = nzeros((self.repr.nMonoms,self.nq,self.nu), dtype=nfloat)
        
        # The integration of the velocity
        MifTaylor[:self.nqv, 1:1+self.nqv] = nidentity(self.nqv, dtype=nfloat)
        
        # Setup the inputs
        xList = [float(ax) for ax in x]
        # First get all taylor series of non-inversed
        fTaylor = nmatrix(self.taylorF.fTaylor_eval(*xList)) # Pure np.matrix #TODO search for ways to vectorize
        MTaylor = [nmatrix(aFunc(*xList)) for aFunc in self.taylorM.MTaylor_eval] #List of matrices #TODO search for ways to vectorize
        GTaylor = [nmatrix(aFunc(*xList)) for aFunc in self.taylorG.GTaylor_eval] #List of matrices #TODO search for ways to vectorize
        
        #Setup the results
        MifTaylor = nzeros((self.nq, self.repr.nMonoms),dtype=nfloat)
        MifTaylor[0:self.nqv,1:1+self.nqv] = nidentity(self.nqv) #Second order nature of the system
        MiGTaylor = [None for _ in range(self.repr.nMonoms)]
        
        # Inverse the inertia matrix at the current point
        Mi = nmatrix(inv(MTaylor[0]))

        #Build up the dict
        evalDictF = {'M':MTaylor[0], 'Mi':Mi, 'W':None}
        #Add the derivative keys (values set later on
        for aDerivStr,_ in self.inversionTaylor.allDerivs.items():
            evalDictF[aDerivStr+'M'] = None
            evalDictF[aDerivStr+'W'] = None
        
        evalDictG = evalDictF.copy()
        
        evalDictF['W'] = nmatrix(fTaylor[:,[0]])
        evalDictF['W'] = nmatrix(GTaylor[0])

        # Now loop over all
        digits_ = self.repr.digits
        monom2num_ = self.repr.monom2num

        derivVarAsInt = nzeros((self.nq,),dtype=nintu)

        for k,aMonom in enumerate(self.repr.listOfMonomials):
            idxC = 0
            derivVarAsInt[:] = 0
            for i, aExp in enumerate(aMonom):
                for _ in range(aExp):
                    #Save as int
                    derivVarAsInt[idxC] = 10**(digits_*i) #The int of each deriv
                    idxC += 1

            idxDict = self.inversionTaylor.allDerivs.copy() # -> get the column of the corresponding column
            for aKey, aVal in idxDict.items():
                tmpVal = 0
                for aaVal in aVal:
                    tmpVal += derivVarAsInt[aaVal]
                #To idxColumn
                idxDict[aKey] = monom2num_[tmpVal]
            
            # Now we have the correct id associated to each derivative
            # TODO here we simply calculate all, even the ones that are not used, ie the degree of the current monome is smaller than the maximal degree

            # Set up the two dicts
            for idxKey,idxVal in idxDict.items():
                #Set deriv mass mat
                evalDictF[idxKey+'M'] = evalDictG[idxKey+'M'] = MTaylor[idxVal]
                #Set the "function"
                evalDictF[idxKey+'W'] = nmatrix(fTaylor[:,[idxVal]])
                evalDictF[idxKey+'W'] = nmatrix(GTaylor[idxVal])
            
            
    
    
    
                # Get
            #allDependentList =
        
        
        
        
        


                    
        
        
        
        
    
    
    
    
    
    
    
    