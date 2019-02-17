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
            if ctrlInput is not None:
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
    
    def getUopt(self,x:np.ndarray, dx:np.ndarray, respectCstr:bool=False, t:float=0.):
        """
        Computes the necessary control input to achieve the derivative given the position
        Seek uStar such that xd = f(x)+g(x).uStar
        :param x:
        :param dx:
        :param respectCstr:
        :param t:
        :return:
        """
        
        x = x.reshape((self.nq,-1))
        dx = dx.reshape((self.nq,-1))
        m = x.shape(1)
        
        if __debug__:
            assert x.shape==dx.shape
        
        uStar = np.zeros((self.nu, m), dtype=nfloat_)
        
        for k in range(m):
            fx, gx = self.getTaylorApprox(x[:,[k]],maxDeg=0)
            # We need to solve
            # g(x).uStar = xd - f(x)
            uStar[:,[k]], res, _, _ = lstsq(gx[0], dx[:,[k]]-fx)
            if __debug__:
                assert res < 1e-9, "Could not solve"
        
        if respectCstr:
            self.ctrlInput(uStar,t)
        
        return uStar
        
        
        

class secondOrderSys(dynamicalSystem):
    
    def __init__(self, repr:polynomialRepr, massMat:sy.Matrix, f:sy.Matrix, g:sy.Matrix, q:"symbols", u:"symbols", maxTaylorDegree:int=3, ctrlInput:boxInputCstr=None, file:str=None):
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
    
    def getTaylorApprox(self,x:np.ndarray,maxDeg:int=None,minDeg:int=0):
        
        # TODO this is a naive implementation
        
        if __debug__:
            assert (maxDeg is None) or (maxDeg <= self.maxTaylorDeg)
        
        maxDeg = self.maxTaylorDeg if maxDeg is None else maxDeg
        
        # Set up the monoms need for the Taylor exp
        listOfMonomsTaylor_ = []
        for k in range(minDeg,maxDeg+1):
            listOfMonomsTaylor_.extend(self.repr.listOfMonomialsPerDeg[k])
        nMonomsTaylor_ = len(listOfMonomsTaylor_)
        
        # return Value
        MifTaylor = nzeros((self.nq, nMonomsTaylor_), dtype=nfloat)
        #Integration of velocity (Only if the first order terms appear
        if (minDeg<=1) and (maxDeg>=1):
            MifTaylor[0:self.nqp,int(minDeg==0)+self.nqp:int(minDeg==0)+self.nq] = nidentity(self.nqv)  # Second order nature of the system
        MiGTaylor = nzeros((nMonomsTaylor_,self.nq,self.nu), dtype=nfloat)
        
        # Setup the inputs
        xList = x.squeeze()#[float(ax) for ax in x]
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
    
    def __call__(self, x:np.ndarray, u_:Union[np.ndarray,Callable], t:float=0., restrictInput:bool=True, mode:List[int]=[0,0], x0:np.ndarray=None):
        """
        Evaluate dynamics for current position and control input
        :param x:
        :param u_:
        :param t:
        :param restrictInput:
        :param mode: First letter -> sys dyn; second: sym dyn; Zero is nonlinear dyn, int means taylor approx
        :param x0:
        :return:
        """
        if __debug__:
            assert x.shape[0] == self.nq
            assert all([(aMode >= 0) and (aMode <=self.maxTaylorDeg) for aMode in mode ])
        
        # Check if u_ is Callable evaluate first
        if hasattr(u_, "__call__"):
            # General function used for optimized input later on
            u = u_(x,t)
        elif u_.shape == (self.nu, self.nq):
            #This is actually a feedback matrix
            u = ndot(u_,x)
        else:
            # Its an actual control input
            u=u_
        
        if restrictInput:
            u = self.ctrlInput(u,t)
        
        if __debug__:
            assert x.shape[1] == u.shape[1]
            assert u.shape[0] == self.nu
        
        if x.shape[1] == 1:
            #Always
            # Integrative part
            xd = nzeros((self.nq, 1), dtype=nfloat)
            xd[:self.nqp, 0] = x[self.nqp:, 0]

            xL = x.squeeze()
            if mode[0] == 0:
                # This is a bit inconvenient as we have to solve twice with different approx of the mass matrix
                # System dynamics
                Mi = self.pDerivM.M0_eval(*xL)
                F = self.pDerivM.f0_eval(*xL)
                
                xd[self.nqp:, 0] = ssolve(Mi, F, assume_a='pos')  # Mass matrix is positive definite
            else:
                # todo write a efficient function to evaluate all monomials
                raise NotImplementedError
            
            if mode[1] == 0:
                Mi = self.pDerivM.M0_eval(*xL)
                G = self.pDerivM.G0_eval(*xL)
                
                # Add input dependent part
                xd[self.nqp:,0] += ssolve(Mi, ndot(G,u), assume_a='pos')# Mass matrix is positive definite
            else:
                raise NotImplementedError
                
        else:
            raise NotImplementedError

        return xd

    def getUopt(self,x: np.ndarray,ddx: np.ndarray,respectCstr: bool = False,t: float = 0., fullDeriv:bool=False):
        """
        Computes the necessary control input to achieve the second derivative given the position and velocity
        Seek uStar such that M.ddx = f(x)+g(x).uStar
        :param x:
        :param dx:
        :param respectCstr:
        :param t:
        :return:
        """
    
        x = x.reshape((self.nq,-1))
        if fullDeriv:
            ddx = ddx.reshape((self.nq,-1))
            ddx = ddx[self.nqv:,:]
        else:
            ddx = ddx.reshape((self.nqv,-1))
        
        m = x.shape[1]
    
        if __debug__:
            assert x.shape[1] == ddx.shape[1]
    
        uStar = np.zeros((self.nu,m),dtype=nfloat)
    
        for k in range(m):
            # Compute current mass matrix, system dynamics and input dynamics
            Mx = self.pDerivM.M0_eval[0](*x[:,k])
            fx = self.pDerivF.f0_eval(*x[:,k])
            Gx = self.pDerivG.G0_eval[0](*x[:,k])
            # We need to solve
            # g(x).uStar = M.ddx - f(x)
            uStar[:,[k]],res,_,_ = lstsq(Gx,ndot(Mx,ddx[:,[k]])-fx)

        if respectCstr:
            self.ctrlInput(uStar,t)
    
        return uStar
        


                    
        
        
        
        
    
    
    
    
    
    
    
    