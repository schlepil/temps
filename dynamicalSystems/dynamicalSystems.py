from coreUtils import *
from polynomial import polynomialRepr
from dynamicalSystems.inputs import boxInputCstr

from polynomial.utils_numba import getIdxAndParent

import string

from dynamicalSystems.derivUtils import getInverseTaylorStrings, compParDerivs

class dynamicalSystem:
    
    def __init__(self, repr:polynomialRepr, q, u, maxTaylorDegree, ctrlInput:boxInputCstr):
        
        if __debug__:
            assert q.shape[1] == 1
            assert u.shape[1] == 1
            assert maxTaylorDegree <= repr.maxDeg
            assert repr.nDims == q.shape[0]
            assert u.shape[0] == ctrlInput.nu
        
        self.repr = repr
        
        self.q = q
        self.u = u
        
        self.nq = int(q.shape[0])
        self.nu = int(u.shape[0])
        
        self.ctrlInput = ctrlInput
        
        self.maxTaylorDeg=maxTaylorDegree
        
    
    def getTaylorApprox(self, x:np.ndarray, maxDeg:int)->Tuple:
        raise NotImplementedError
    
    def precompute(self):
        raise NotImplementedError
    
    def __call__(self, x:np.ndarray, u:np.ndarray, mode:str='OO', x0:np.ndarray=None):
        raise NotImplementedError

class secondOrderSys(dynamicalSystem):
    
    def __init__(self, repr:polynomialRepr, massMat:sy.Matrix, f:sy.Matrix, g:sy.Matrix, q:"symbols", u:"symbols", maxTaylorDegree:int=3, ctrlInput:boxInputCstr, file:str=None):
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
        
        super(secondOrderSys,self).__init__(repr,q,u,maxTaylorDegree,ctrlInput)
        
        self.massMat = massMat
        self.f = f
        self.g = g
        
        self.nqv = self.nq//2
        self.nqp = self.nqv
        
        if file == None:
            self.compPderivAndTaylor()
        else:
            self.fromFile(file)

        return None
    
    def compPderivAndTaylor(self):
        from math import factorial
        
        self.compPDerivF()
        self.compPDerivG()
        self.compPDerivM()
        if self.maxTaylorDeg<=3:
            derivStrings = string.ascii_lowercase[23:23+self.maxTaylorDeg]
        else:
            derivStrings = string.ascii_lowercase[:self.maxTaylorDeg]
        self.inversionTaylor = variableStruct(**getInverseTaylorStrings('M','Mi','W',derivStrings))
        #Add an array with the weighting coefficient of the Taylor expansion
        self.inversionTaylor.weightingMonoms = []
        for k in range(self.maxTaylorDeg+1):
            self.inversionTaylor.weightingMonoms.extend( len(self.repr.listOfMonomialsPerDeg[k])*[1./float(factorial(k))] )
        self.inversionTaylor.weightingMonoms = narray(self.inversionTaylor.weightingMonoms, dtype=nfloat)
        self.inversionTaylor.weightingMonoms3d = np.transpose(np.broadcast_to(self.inversionTaylor.weightingMonoms,(self.nu,self.nq,self.inversionTaylor.weightingMonoms.size)),(2,1,0))

        return None
        
    def compPDerivF(self):
        
        pDerivFD = compParDerivs(self.f, 'f', self.q, False, self.maxTaylorDeg, self.repr)

        self.pDerivF = variableStruct(**pDerivFD)

        return None
    
    def compPDerivM(self):
        """
        Taylor expansion of the mass matrix
        
        :return:
        """

        pDerivMD = compParDerivs(self.massMat, 'M', self.q, True, self.maxTaylorDeg, self.repr)


        self.pDerivM = variableStruct(**pDerivMD)

    def compPDerivG(self):
        """
        Taylor expansion of the input matrix

        :return:
        """

        pDerivGD = compParDerivs(self.g, 'G', self.q, True, self.maxTaylorDeg, self.repr)
    
        self.pDerivG = variableStruct(**pDerivGD)

        return None
    
    def getTaylorApprox(self,x:np.ndarray,maxDeg:int=None):
        
        # TODO this is a naive implementation
        
        if __debug__:
            assert (maxDeg is None) or (maxDeg <= self.maxTaylorDeg)
        
        maxDeg = self.maxTaylorDeg if maxDeg is None else maxDeg
        
        # Set up the monoms need for the Taylor exp
        listOfMonomsTaylor_ = []
        for k in range(maxDeg+1):
            listOfMonomsTaylor_.extend(self.repr.listOfMonomialsPerDeg[k])
        nMonomsTaylor = len(listOfMonomsTaylor_)
        
        # return Value
        MifTaylor = nzeros((self.nq, nMonomsTaylor), dtype=nfloat)
        #Integration of velocity
        MifTaylor[0:self.nqp,1+self.nqp:1+self.nq] = nidentity(self.nqv)  # Second order nature of the system
        MiGTaylor = nzeros((nMonomsTaylor,self.nq,self.nu), dtype=nfloat)
        
        # Setup the inputs
        xList = [float(ax) for ax in x]
        # First get all taylor series of non-inversed up to maxDeg
        # The functions [f,M,G]Taylor are  created such that the monomials up maxDeg are returned
        #fTaylor = nmatrix(self.taylorF.fTaylor_eval(*xList)) # Pure np.matrix #TODO search for ways to vectorize
        #MTaylor = [nmatrix(aFunc(*xList)) for aFunc in self.taylorM.MTaylor_eval] #List of matrices #TODO search for ways to vectorize
        #GTaylor = [nmatrix(aFunc(*xList)) for aFunc in self.taylorG.GTaylor_eval] #List of matrices #TODO search for ways to vectorize
        
        # Compute the taylor only up to the degree needed
        indexKey = "PDeriv_to_{0:d}_eval".format(maxDeg)
        fPDeriv = nmatrix(self.pDerivF.__dict__["f"+indexKey](*xList)) # Pure np.matrix #TODO search for ways to vectorize
        MPDeriv = [nmatrix(aFunc(*xList)) for aFunc in self.pDerivM.__dict__["M"+indexKey]] #List of matrices #TODO search for ways to vectorize
        GPDeriv = [nmatrix(aFunc(*xList)) for aFunc in self.pDerivG.__dict__["G"+indexKey]] #List of matrices #TODO search for ways to vectorize
        
        # Inverse the inertia matrix at the current point
        Mi = nmatrix(inv(MPDeriv[0]))

        #Build up the dict
        evalDictF = {'M':MPDeriv[0], 'Mi':Mi, 'W':None}
        #Add the derivative keys (values set later on
        for aDerivStr,_ in self.inversionTaylor.allDerivs.items():
            evalDictF[aDerivStr+'M'] = None
            evalDictF[aDerivStr+'W'] = None
        
        evalDictG = evalDictF.copy()
        
        evalDictF['W'] = nmatrix(fPDeriv[:,[0]])
        evalDictG['W'] = nmatrix(GPDeriv[0])

        # Now loop over all
        digits_ = self.repr.digits
        monom2num_ = self.repr.monom2num
        funcStrings_ = self.inversionTaylor.funcstr
        eDict_ = {}

        derivVarAsInt = nzeros((maxDeg,),dtype=nintu)

        for k,aMonom in enumerate(listOfMonomsTaylor_):
            idxC = 0
            derivVarAsInt[:] = 0
            multi0 = 1
            multi1 = 10**digits_
            for aExp in reversed(aMonom):
                for _ in range(aExp):
                    #Save as int
                    derivVarAsInt[idxC] = multi0 #The int of each deriv
                    idxC += 1
                multi0*=multi1#New exponent -> adjust

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
                evalDictF[idxKey+'M'] = evalDictG[idxKey+'M'] = MPDeriv[idxVal]
                #Set the "function"
                evalDictF[idxKey+'W'] = nmatrix(fPDeriv[:,[idxVal]])
                evalDictG[idxKey+'W'] = nmatrix(GPDeriv[idxVal])
            
            thisFuncString = funcStrings_[aMonom.sum()]
            #Evaluate
            MifTaylor[self.nqp:,[k]] = eval(thisFuncString,eDict_,evalDictF)
            MiGTaylor[k,self.nqp:,:] = eval(thisFuncString,eDict_,evalDictG)
            
        # Do the weighting to go from partial derivs to taylor
        nmultiply(MifTaylor, self.inversionTaylor.weightingMonoms, out=MifTaylor)
        nmultiply(MiGTaylor,self.inversionTaylor.weightingMonoms3d,out=MiGTaylor)
        #Done
        return MifTaylor, MiGTaylor
    
    def __call__(self, x:np.ndarray, u:np.ndarray):
        """Evaluate dynamics for current position and control input"""
        if __debug__:
            assert x.shape[1] == u.shape[1]
            assert x.shape[0] == self.nq
            assert u.shape[0] == self.nu
        
        if x.shape[1] == 1:
            #Integrative part
            xd = nzeros((self.nq,1), dtype=nfloat)
            xd[:self.nqp,0] = x[self.nqp:,0]
            
            #Evaluate
            x = x.squeeze()
            Mi = self.pDerivM.M0_eval(x)
            F = self.pDerivM.f0_eval(x)
            G = self.pDerivM.G0_eval(x)
            
            xd[self.nqp:,0] = ssolve(Mi, F+G, assume_a='pos')# Mass matrix is positive definite
        else:
            raise NotImplementedError

        return xd
        
        


                    
        
        
        
        
    
    
    
    
    
    
    
    