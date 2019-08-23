from coreUtils import *
from polynomial import polynomialRepr, polynomial, polyFunction
from dynamicalSystems.inputs import boxInputCstr

from polynomial.utils_numba import getIdxAndParent

import string

from dynamicalSystems.derivUtils import getInverseTaylorStrings, compParDerivs, getTaylorWeights

class dynamicalSystem:
    
    def __init__(self, repr:polynomialRepr, q, u, maxTaylorDegree, ctrlInput:boxInputCstr):
        
        if dbg__0:
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
        
        self.ctrlInput = ctrlInput #type: ctrlInput
        
        self.maxTaylorDeg=maxTaylorDegree
        
    
    def getTaylorApprox(self, x:np.ndarray, maxDeg:int)->Tuple:
        raise NotImplementedError
    
    def precompute(self):
        raise NotImplementedError
    
    def gEval(self, X:np.ndarray):
        raise NotImplementedError
    
    def fEval (self, X:np.ndarray):
        raise NotImplementedError
    
    def __call__(self, x:np.ndarray, u:np.ndarray, restrictInput:bool=True, mode:str=[0,0], x0:np.ndarray=None, dx0:np.ndarray=None):
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
        
        if dbg__0:
            assert x.shape==dx.shape
        
        uStar = np.zeros((self.nu, m), dtype=nfloat_)
        
        for k in range(m):
            fx, gx = self.getTaylorApprox(x[:,[k]],maxDeg=0)
            # We need to solve
            # g(x).uStar = xd - f(x)
            uStar[:,[k]], res, _, _ = lstsq(gx[0], dx[:,[k]]-fx)
            if dbg__0:
                assert res < 1e-9, "Could not solve"
        
        if respectCstr:
            self.ctrlInput(uStar,t)
        
        return uStar
        
        
        

class secondOrderSys(dynamicalSystem):
   # dynSys = secondOrderSys(repr, M, -F, gInput, qM, uM)
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

        # Determines how the Taylor approx is evaluated
        # strictEval = True -> Get taylor approx and compute
        # strictEval = False -> Compute the taylor approx of M (Mt), f (ft) and G (Gt), then compute xdd = Mt^-1.(ft + Gt.u) #Except if the maximal degrees for system and input dynamics do not coincide
        self.strictEval = True

        return None

    @classmethod
    def fromFileOrDict(cls, repr:polynomialRepr, fileOrDict:Union[str,dict]):
        raise NotImplementedError
    
    def compPderivAndTaylor(self):
        
        self.compPDerivF()
        self.compPDerivG()
        self.compPDerivM()
        if self.maxTaylorDeg<=3:
            derivStrings = string.ascii_lowercase[23:23+self.maxTaylorDeg]
        else:
            derivStrings = string.ascii_lowercase[:self.maxTaylorDeg]
        self.inversionTaylor = variableStruct(**getInverseTaylorStrings('M','Mi','W',derivStrings))
        #Add an array with the weighting coefficient of the Taylor expansion
        #self.inversionTaylor.weightingMonoms = []
        #for k in range(self.maxTaylorDeg+1):
        #    self.inversionTaylor.weightingMonoms.extend( len(self.repr.listOfMonomialsPerDeg[k])*[1./float(factorial(k))] )
        #self.inversionTaylor.weightingMonoms = narray(self.inversionTaylor.weightingMonoms, dtype=nfloat)
        # Weighting coeffs wrong
        self.inversionTaylor.weightingMonoms = getTaylorWeights(self.repr.listOfMonomials[self.repr.varSlicesUpToDeg[self.maxTaylorDeg]]) #getTaylorWeights(self.repr.listOfMonomials)
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
        
        if dbg__0:
            assert (maxDeg is None) or (maxDeg <= self.maxTaylorDeg)
        
        maxDeg = self.maxTaylorDeg if maxDeg is None else maxDeg
        
        # Set up the monoms need for the Taylor exp
        listOfMonomsTaylor_ = []
        for k in range(minDeg,maxDeg+1):
            listOfMonomsTaylor_.extend(self.repr.listOfMonomialsPerDeg[k])
        nMonomsOffset_ = sum([len(a) for a in self.repr.listOfMonomialsPerDeg[:minDeg]])
        nMonomsTaylor_ = len(listOfMonomsTaylor_)
        
        # return Value
        MifTaylor = nzeros((self.nq, nMonomsTaylor_), dtype=nfloat)
        #Integration of velocity (Only if the first order terms appear
        if (minDeg<=1) and (maxDeg>=1):
            MifTaylor[0:self.nqp,int(minDeg==0)+self.nqp:int(minDeg==0)+self.nq] = nidentity(self.nqv)  # Second order nature of the system
        # BUG FIX
        # Add the current reference velocity (if necessary)
        if (minDeg == 0):
            MifTaylor[0:self.nqp,0] = x[self.nqp:,0]
        
        MiGTaylor = nzeros((nMonomsTaylor_,self.nq,self.nu), dtype=nfloat)
        
        # Setup the inputs
        xList = x.squeeze()#[float(ax) for ax in x]
        # First get all taylor series of non-inversed up to maxDeg
        # The functions [f,M,G]Taylor are  created such that the monomials up maxDeg are returned
        #fTaylor = nmatrix(self.taylorF.fTaylor_eval(*xList)) # Pure np.matrix #TODO search for ways to vectorize
        #MTaylor = [nmatrix(aFunc(*xList)) for aFunc in self.taylorM.MTaylor_eval] #List of matrices #TODO search for ways to vectorize
        #GTaylor = [nmatrix(aFunc(*xList)) for aFunc in self.taylorG.GTaylor_eval] #List of matrices #TODO search for ways to vectorize

        # Compute the taylor only up to the degree needed
        # Note that, in order to compute the needed orders (i.e. minDeg>0) we have to compute all
        # monoms of the original (non-inversed) dynamics
        indexKey = f"PDeriv_to_{maxDeg:d}_eval"
        fPDeriv = nmatrix(self.pDerivF.__dict__["f"+indexKey](*xList),dtype=nfloat) # Pure np.matrix #TODO search for ways to vectorize
        MPDeriv = [nmatrix(aFunc(*xList),dtype=nfloat) for aFunc in self.pDerivM.__dict__["M"+indexKey]] #List of matrices #TODO search for ways to vectorize
        GPDeriv = [nmatrix(aFunc(*xList),dtype=nfloat) for aFunc in self.pDerivG.__dict__["G"+indexKey]] #List of matrices #TODO search for ways to vectorize

        
        # Inverse the inertia matrix at the current point
        Mi = nmatrix(inv(MPDeriv[0]))

        ##Build up the dict
        #evalDictF = {'M':MPDeriv[0], 'Mi':Mi, 'W':None}
        ##Add the derivative keys (values set later on)
        #for aDerivStr,_ in self.inversionTaylor.allDerivs.items():
        #    evalDictF[aDerivStr+'M'] = None
        #    evalDictF[aDerivStr+'W'] = None

        evalDictF = {}.fromkeys( ichain.from_iterable( [[aDerivStr+'M', aDerivStr+'W'] for aDerivStr in self.inversionTaylor.allDerivs.keys()] ) )
        evalDictF['M'] = MPDeriv[0]
        evalDictF['Mi'] = Mi
        evalDictF['W'] = None


        evalDictG = evalDictF.copy()
        
        evalDictF['W'] = nmatrix(fPDeriv[:,[0]],dtype=nfloat)
        evalDictG['W'] = nmatrix(GPDeriv[0],dtype=nfloat)

        # Now loop over all
        digits_ = self.repr.digits
        monom2num_ = self.repr.monom2num
        funcStrings_ = self.inversionTaylor.funcstr
        eDict_ = {}

        derivVarAsInt = nzeros((self.maxTaylorDeg,),dtype=nintu) #Initialize all unused derivatives to zero

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

            # Set up the two dicts
            for idxKey,idxVal in idxDict.items():
                #Set deriv mass mat
                evalDictF[idxKey+'M'] = evalDictG[idxKey+'M'] = MPDeriv[idxVal]
                #Set the "function"
                evalDictF[idxKey+'W'] = nmatrix(fPDeriv[:,[idxVal]],dtype=nfloat)
                evalDictG[idxKey+'W'] = nmatrix(GPDeriv[idxVal],dtype=nfloat)
            
            thisFuncString = funcStrings_[aMonom.sum()]
            #Evaluate
            MifTaylor[self.nqp:,[k]] = eval(thisFuncString,eDict_,evalDictF)
            MiGTaylor[k,self.nqp:,:] = eval(thisFuncString,eDict_,evalDictG)
            
        # Do the weighting to go from partial derivs to taylor
        nmultiply(MifTaylor, self.inversionTaylor.weightingMonoms[nMonomsOffset_:nMonomsOffset_+nMonomsTaylor_], out=MifTaylor)
        nmultiply(MiGTaylor,self.inversionTaylor.weightingMonoms3d[nMonomsOffset_:nMonomsOffset_+nMonomsTaylor_, :, :],out=MiGTaylor)
        #Done
        return MifTaylor, MiGTaylor

    def ensureShape(self, MG:"M or G", thisShape:"shape"):
        # Take care of certain particularities of lambdify
        try:
            # If the mass matrix is input dependent or x.shape[1]==1 it can be correctly reshaped
            MG.resize(thisShape)  # [nqv,nqv,nPt]
        except:
            # If the mass matrix is not input dependent and there are several points -> tile
            assert MG.size == thisShape[0]*thisShape[1]
            MG.resize((thisShape[0], thisShape[1], 1))
            MG = np.tile(MG, (1, 1, thisShape[2]))
        return MG

    def computeQddInv(self, M:"Mass matrices", fg:"RHS dynamics"):
        # Compute - system dynamics
        M = self.ensureShape(M, (self.nqv, self.nqv, fg.shape[1]))
        if self.nqv == 1:
            # special case: only one degree of freedom
            xdd = np.divide(fg.squeeze(), M.squeeze()).reshape((1, fg.shape[1]))
        else:
            xdd = nempty((self.nqp, fg.shape[1]), dtype=nfloat)
            for i in range(fg.shape[1]):
                xdd[self.nq:, [i]] = ssolve(M[:, :, i], fg[:, [i]], assume_a='pos')
        return xdd
    
    def compDynSysOrig(self, x:np.ndarray):
        """
        Compute the nonlinear system dynamics
        :param x:
        :param u:
        :param t:
        :param restrictInput:
        :param mode:
        :param x0:
        :param dx0:
        :return:
        """

        xLi = [x[i, :] for i in range(self.nq)]
        M = self.pDerivM.M0_eval[0](*xLi)  # [nqv,nqv,nPt] or [nqv,nqv]
        f = self.pDerivF.f0_eval(*xLi)  # [nqv,nPt] or [nqv,1]
        f.resize((self.nqv, x.shape[1]))
        return self.computeQddInv(M, f)
    
    def compDynSysPolyNonStrict(self, x:"Points", x0:"origin Point", deg:int):
        """
        "Nonstrict computations" results might be slightly different from what is used within the proofs
        Here we compute
        1) Compute Taylor approx Mt.xdd = ft
        2) Inverse -> xdd = Mt^-1.ft
        :param x:
        :param x0:
        :param deg:
        :return:
        """
        # Get partial derivs
        fPDeriv = self.pDerivF.__dict__[f"fPDeriv_to_{deg:d}_eval"](*x0.squeeze())  # [nq,nMonoms]
        # MPDeriv = self.pDerivM.__dict__[f"MPDeriv_to_{mode[0]:d}_MAT_eval"](*x0.squeeze())  # [nqv,nqv,nMonoms]
        # Lambdifying tensors does not work as expected -> TODO
        # Is returned as List of lists -> explicit conversion to np.array
        MPDeriv = narray(self.pDerivM.__dict__[f"MPDeriv_to_{deg:d}_MAT_eval"](*x0.squeeze()), dtype=nfloat)  # [nqv,nqv,nMonoms]
        MPDeriv.resize((self.nqv, self.nqv, len(self.repr.varNumsUpToDeg[deg])))  # Ensure dimension
    
        # Partial derivs to evaluated Taylor
        z = self.repr.evalAllMonoms(x if x0 is None else x-x0, deg)  # TODO check correctness, monoms have to be evaluated at the offset
        # multiply with weights
        z *= self.inversionTaylor.weightingMonoms[:z.shape[0]].reshape((z.shape[0], 1))
        # (broadcast) multiply and sum up and contract
        f = ndot(fPDeriv, z)
        M = neinsum("ijk,kn->ijn", MPDeriv, z)  # [nqv,nqv,nMonoms] . [nMonoms,nPt] -> [nqv,nqv,nPt] Mass matrices stacked along third axis as above
        return self.computeQddInv(M, f)
    
    def compDynSysPolyStrict(self, deltax:"Points", x0:"origin Point", deg:int, taylorExp=[None, None]):
        """
        Compute system dynamics based on taylor approx. Gives strictly the same results as the ones used in the proofs
        xdd = M^-1.f
        compute directly the taylor expansion of this so {M^-1.f}t
        :param x:
        :param x0:
        :param deg:
        :return:
        """
        
        # Compute Taylor expansion
        fTaylor, gTaylor =  self.getTaylorApprox(x0, maxDeg=deg)
        taylorExp[0] = fTaylor
        taylorExp[1] = gTaylor
        
        # deltax -> derivation variables
        # Check if deltax is already the vector of monomials
        if deltax.shape[0] == self.nq:
            deltaz = self.repr.evalAllMonoms(deltax, deg)
        else:
            nMonoms = len(self.repr.varNumsUpToDeg[deg])
            assert deltax.shape[0]>=nMonoms
            deltaz = deltax
        
        # Compute
        return ndot(fTaylor, deltaz[:nMonoms, :])
    
    def compInputSysOrig(self, x:"Points", u:"Input"):
        """
        Same as compDynSysOrig but for input
        :param x:
        :param u:
        :return:
        """
        xLi = [x[i, :] for i in range(self.nq)]
        M = self.pDerivM.M0_eval[0](*xLi)  # [nqv,nqv,nPt] or [nqv,nqv]
        G = self.pDerivG.G0_eval[0](*xLi)  # [nqv,nu,nPt] or [nqv,nu]
        # Ensure dimensions
        # Mass matrix dim ensured by computeQddInv
        # Same issue for G
        G = self.ensureShape(G, (self.nqv, self.nu, x.shape[1]))
        # Compute
        g = neinsum("ijk,jk->ik", G, u)
        
        return self.computeQddInv(M, g)
        
    def compInputSysPolyNonStrict(self, x:"Points", x0:"Origin", u:"Input", deg:int):
        
        # Get partial derivs
        # Lambdifying tensors does not work as expected -> TODO
        # Is returned as List of lists -> explicit conversion to np.array
        GPDeriv = narray(self.pDerivG.__dict__[f"GPDeriv_to_{deg}_MAT_eval"](*x0.squeeze()), dtype=nfloat)  # [nqv,nu,nMonoms]
        MPDeriv = narray(self.pDerivM.__dict__[f"MPDeriv_to_{deg}_MAT_eval"](*x0.squeeze()), dtype=nfloat)  # [nqv,nqv,nMonoms]
        # Ensure dimensions
        GPDeriv.resize((self.nqv, self.nu, len(self.repr.varNumsUpToDeg[deg])))
        MPDeriv.resize((self.nqv, self.nqv, len(self.repr.varNumsUpToDeg[deg])))
    
        # Partial derivs to evaluated Taylor
        z = self.repr.evalAllMonoms(x if x0 is None else x-x0, deg)  # TODO check correctness, monoms have to be evaluated at the offset
        # multiply with weights
        z *= self.inversionTaylor.weightingMonoms[:z.shape[0]].reshape((z.shape[0], 1))
        # (broadcast) multiply and sum up and contract
        g = neinsum("ijk,kn,jn->in", GPDeriv, z, u)  # ([nq,nu,nMonoms] . [nMonoms,nPt]) . (nu,nPt) -> [nq,nPt] Compute input
        M = neinsum("ijk,kn->ijn", MPDeriv, z)  # [nq,nq,nMonoms] . [nMonoms,nPt] -> [nq,nq,nPt] Mass matrices stacked along third axis as above
       
        return self.computeQddInv(M, g)
    
    def compInputSysPolyStrict(self, deltax:"Points", x0:"origin Point", u:"Input", deg:int, taylorExp=[None, None]):
        
        # Get the taylor approx
        gTaylor = self.getTaylorApprox(x0, maxDeg=deg)[1] if taylorExp[1] is None else taylorExp[1]
        
        # deltax -> derivation variables
        # Check if deltax is already the vector of monomials
        if deltax.shape[0] == self.nq:
            deltaz = self.repr.evalAllMonoms(deltax, deg)
        else:
            nMonoms = len(self.repr.varNumsUpToDeg[deg])
            assert deltax.shape[0] >= nMonoms
            deltaz = deltax
        
        # Compute
        # [nMonoms,nq,nu].[nMonoms, nPt].[nu, nPt] -> [nq, nPt]
        return neinsum("kij,kn,jn->in", gTaylor, deltaz[:nMonoms, :], u)


    def __call__(self, x:np.ndarray, u:Union[np.ndarray,Callable], t:float=0., restrictInput:bool=True, mode:List[int]=[0,0], x0:np.ndarray=None, dx0:np.ndarray=None):
        """
        Evaluate dynamics for current position and control input
        :param x:
        :param u_:
        :param t:
        :param restrictInput:
        :param mode: First letter -> sys dyn; second: sym dyn; Zero is nonlinear dyn, int means taylor approx
        :param x0:
        :param dx0: reference velocity. If given, only the velocity difference will be returned
        :return:
        """

        if dbg__0:
            assert x.shape[0] == self.nq
            assert all([(aMode >= 0) and (aMode <=self.maxTaylorDeg) for aMode in mode ])
        
        # Check if u_ is Callable evaluate first
        if hasattr(u, "__call__"):
            # General function used for optimized input later on
            u = u(x,t)
        #elif u_.shape == (self.nu, self.nq):
        #    #This is actually a feedback matrix
        #    u = ndot(u_,x-x0)
        # Can no longer be supported to avoid ambiguity
        else:
            # Its an actual control input
            u=u
        
        if restrictInput:
            u = self.ctrlInput(u,t)
        
        if dbg__0:
            assert x.shape[1] == u.shape[1]
            assert u.shape[0] == self.nu
        
        xd = np.zeros_like(x)
        xd[:self.nqp, :] = x[self.nqp:, :]

        #Speedup stuff
        taylorExp = [None,None]
        deltaz = None
        
        # system dynamics
        if mode[0] == 0:
            xd[self.nqp:,:] += self.compDynSysOrig(x)
        elif mode[0] <= self.maxTaylorDeg:
            if self.strictEval:
                deltaz = self.repr.evalAllMonoms(x-x0, max(mode)) # monomial vector in derivation variables
                xd = self.compDynSysPolyStrict(deltaz, x0, mode[0], taylorExp=taylorExp) #Override
            else:
                xd[self.nqp:, :] += self.compDynSysPolyNonStrict(x, x0, mode[0]) #Keep base velocity

        # input dynamics
        if mode[1] == 0:
            xd[self.nqp:,:] += self.compInputSysOrig(x, u)
        else:
            if self.strictEval:
                if mode[0] != mode[1]:
                    taylorExp[1] = None
                deltaz = self.repr.evalAllMonoms(x-x0, mode[1]) if deltaz is None else deltaz # monomial vector in derivation variables
                xd += self.compInputSysPolyStrict(deltaz, x0, u, mode[1],taylorExp=taylorExp)
            else:
                xd[self.nqp:, :] += self.compInputSysPolyNonStrict(x, x0, u, mode[1])
            
        if dx0 is not None:
            # Adjust for reference
            xd -= dx0
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
        # print('this is nq',self.nq)
        # print('this is nqv', self.nqv)
        # print("this is x", x)
        if fullDeriv:
            ddx = ddx.reshape((self.nq,-1))
            ddx = ddx[self.nqv:,:]
        else:
            ddx = ddx.reshape((self.nqv,-1))
        # print("this is ddx", ddx)
        m = x.shape[1]
    
        if dbg__0:
            assert x.shape[1] == ddx.shape[1]
    
        uStar = np.zeros((self.nu,m),dtype=nfloat)

        for k in range(m):
            # Compute current mass matrix, sthis is uStarystem dynamics and input dynamics
            Mx = self.pDerivM.M0_eval[0](*x[:,k])
            fx = self.pDerivF.f0_eval(*x[:,k])
            Gx = self.pDerivG.G0_eval[0](*x[:,k])

            # We need to solve
            # g(x).uStar = M.ddx - f(x)
            uStar[:,[k]],res,_,_ = lstsq(Gx,ndot(Mx,ddx[:,[k]])-fx)

        if respectCstr:
            self.ctrlInput(uStar,t)

        return uStar


class polynomialSys(dynamicalSystem):
    
    def __init__(self, repr: polynomialRepr, fCoeffs:np.ndarray, gCoeffs:np.ndarray, q: "symbols", u: "symbols", maxTaylorDegree: int = 3, ctrlInput: boxInputCstr = None, file: str = None):
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
        
        super(type(self), self).__init__(repr, q, u, maxTaylorDegree, ctrlInput)
                    
        self.fCoeffs_ = fCoeffs.copy()
        self.gCoeffs_ = gCoeffs.copy()
        
        self.fTaylorCoeffs_ = None
        self.gTaylorCoeffs_ = None
        
        self.f = polyFunction(self.repr, (self.nq,1))
        self.g = polyFunction(self.repr, (self.nq,self.nu))
        
        for i in range(self.nq):
            self.f[i,0] = polynomial(self.repr, self.fCoeffs_[i,:])
        
        for i in range(self.nq):
            for j in range(self.nu):
                self.g[i, j] = polynomial(self.repr, self.gCoeffs_[:, i, j])
        
        self.fSym = sy.Matrix(nzeros((self.nq,1)))
        self.gSym = sy.Matrix(nzeros((self.nq,self.nu)))
        
        self.precompute()
    
    def fEval(self, x:np.ndarray):
        return self.f.eval(x)
    
    def gEval(self, x:np.ndarray):
        return self.g.eval(x)
    
    def precompute(self):
        """
        The Taylor expansion of a polynomial is the polynomial itself as long as the degree of the taylor expansion is at least the degree of the polynomial
        However this is not guaranteed therefore we use the "standard" computation
        :return:
        """
        
        zList = [ sy.prod( [xx**ee for xx,ee in zip(self.q, eList) ] ) for eList in self.repr.listOfMonomials ]
        
        # Compute for f
        for i in range(self.nq):
            self.fSym[i,0] = sum( [ az*aCoef for az,aCoef in zip(zList, self.fCoeffs_[i,:]) ] )
        
        for i in range(self.nq):
            for j in range(self.nu):
                self.gSym[i, j] = sum([az*aCoef for az, aCoef in zip(zList, self.gCoeffs_[:, i, j])])
        
        self.compPderivAndTaylor()
                
                
    def compPderivAndTaylor(self):
        from math import factorial

        self.compPDerivFG()
        
        self.taylorExp = variableStruct()
        # Add an array with the weighting coefficient of the Taylor expansion
        
        #weighting coeffs wrong!!! -> multinomial coefficient!!!
        #self.taylorExp.weightingMonoms = []
        #for k in range(self.maxTaylorDeg+1):
        #    self.taylorExp.weightingMonoms.extend(len(self.repr.listOfMonomialsPerDeg[k])*[1./float(factorial(k))])

        self.taylorExp.weightingMonoms = getTaylorWeights(self.repr.listOfMonomials[self.repr.varSlicesUpToDeg[self.maxTaylorDeg]])
        self.taylorExp.weightingMonoms3d = np.transpose(np.broadcast_to(self.taylorExp.weightingMonoms, (self.nu, self.nq, self.taylorExp.weightingMonoms.size)), (2, 1, 0))

        return None

    def compPDerivFG(self):

        pDerivFD = compParDerivs(self.fSym, 'f', self.q, False, self.maxTaylorDeg, self.repr)
        self.pDerivF = variableStruct(**pDerivFD)

        pDerivGD = compParDerivs(self.gSym, 'g', self.q, True, self.maxTaylorDeg, self.repr)
        self.pDerivG = variableStruct(**pDerivGD)

        return None

    def getTaylorApprox(self, x: np.ndarray, maxDeg: int = None, minDeg: int = 0):
        # TODO this is a naive implementation

        if dbg__0:
            assert (maxDeg is None) or (maxDeg <= self.maxTaylorDeg)
    
        maxDeg = self.maxTaylorDeg if maxDeg is None else maxDeg
        
        idxDemand = np.hstack( self.repr.varNumsPerDeg[minDeg:maxDeg+1] )
        
        # Setup the inputs
        xList = x.squeeze()  # [float(ax) for ax in x]
        
        indexKey = f"PDeriv_to_{maxDeg:d}_eval"
        fPDeriv = narray(self.pDerivF.__dict__["f"+indexKey](*xList))  # Pure np.matrix #TODO search for ways to vectorize
        gPDeriv = np.stack([narray(aFunc(*xList)) for aFunc in self.pDerivG.__dict__["g"+indexKey]])  # List of matrices #TODO search for ways to vectorize
    
        # Do the weighting to go from partial derivs to taylor
        nmultiply(fPDeriv, self.taylorExp.weightingMonoms[self.repr.varNumsUpToDeg[maxDeg]], out=fPDeriv)
        nmultiply(gPDeriv, self.taylorExp.weightingMonoms3d[self.repr.varNumsUpToDeg[maxDeg],:,:], out=gPDeriv)
        # Done
        return nrequire(fPDeriv[:, idxDemand], dtype=nfloat, requirements='OA'), nrequire(gPDeriv[idxDemand,:,:], dtype=nfloat, requirements='OA')

    def getUopt(self, x: np.ndarray, dx: np.ndarray, respectCstr: bool = False, t: float = 0.):
        """
        Computes the necessary control input to achieve the desired velocity
        Seek uStar such that dx = f(x)+g(x).uStar
        :param x:
        :param dx:
        :param respectCstr:
        :param t:
        :return:
        """
    
        x = x.reshape((self.nq, -1))

        dx = dx.reshape((self.nq, -1))

        m = x.shape[1]
    
        if dbg__0:
            assert x.shape[1] == dx.shape[1]
    
        uStar = np.zeros((self.nu, m), dtype=nfloat)
        
        F = self.fEval(x)
        G = self.gEval(x)
        if m==1:
            uStar, res, _, _ = lstsq(G, dx-F)
        else:
            for k in range(m):
                # We need to solve
                # g(x).uStar = ddx - f(x)
                uStar[:, [k]], res, _, _ = lstsq(G[k,:,:], dx[:, [k]]-F[:,[k]])
    
        if respectCstr:
            self.ctrlInput(uStar, t)
    
        return uStar



    def __call__(self, x: np.ndarray, u: Union[np.ndarray, Callable], t: float = 0., restrictInput: bool = True, mode: List[int] = [0, 0],
                 x0: np.ndarray = None, dx0: np.ndarray = None):
        """
        Evaluate dynamics for current position and control input
        :param x:
        :param u_:
        :param t:
        :param restrictInput:
        :param mode: First letter -> sys dyn; second: sym dyn; Zero is nonlinear dyn, int means taylor approx
        :param x0:
        :param dx0: reference velocity. If given, only the velocity difference will be returned
        :return:
        """
        if dbg__0:
            assert x.shape[0] == self.nq
            assert all([(aMode >= 0) and (aMode <= self.maxTaylorDeg) for aMode in mode])

        # Check if u_ is Callable evaluate first
        if hasattr(u, "__call__"):
            # General function used for optimized input later on
            u = u(x, t)
        # elif u_.shape == (self.nu, self.nq):
        #    #This is actually a feedback matrix
        #    u = ndot(u_,x-x0)
        # Can no longer be supported to avoid ambiguity
        else:
            # Its an actual control input
            u = u

        if restrictInput:
            u = self.ctrlInput(u, t)

        if dbg__0:
            assert x.shape[1] == u.shape[1]
            assert u.shape[0] == self.nu

        # system dynamics
        if mode[0] == 0:
            f = self.f.eval(x)  # [nPt, nq, 1] or [nq,1]
            f.resize((x.shape[1],self.nq, 1))
            f = nrequire(np.transpose(f, (2,1,0))[0,:,:], dtype=nfloat, requirements='OA')
        elif mode[0] <= self.maxTaylorDeg:
            # Get partial derivs
            fPDeriv = self.pDerivF.__dict__[f"fPDeriv_to_{mode[0]:d}_eval"](*x0.squeeze())  # [nq,nMonoms]

            # Partial derivs to evaluated Taylor
            z = self.repr.evalAllMonoms(x, mode[0])
            # multiply with weights
            z *= self.taylorExp.weightingMonoms[:fPDeriv.shape[1]].reshape((-1,1))
            # (broadcast) multiply and sum up and contract
            f = ndot(fPDeriv, z)

        # Compute - system dynamics
        xd = f

        # input dynamics
        if mode[1] == 0:
            G = self.g.eval(x) # [nPt,nq,nu] or [1,nq,nu]
            # ensure dim
            G.resize((x.shape[1],self.nq,self.nu))
            #Compute
            g = neinsum( 'nij,jn->in', G, u ) #[nPt,nq,nu] . [nu,nPt] -> [nq,nPt]
        else:
            # Get partial derivs
            xList = x0.squeeze()
            indexKey = f"PDeriv_to_{mode[1]:d}_eval"
            GPDeriv = np.stack([narray(aFunc(*xList)) for aFunc in self.pDerivG.__dict__["g"+indexKey]])  # [nMonoms,nq,nu]

            # Partial derivs to evaluated Taylor
            z = self.repr.evalAllMonoms(x, mode[1])
            # multiply with weights
            z *= self.taylorExp.weightingMonoms[:GPDeriv.shape[0]].reshape((-1,1))
            # (broadcast) multiply and sum up and contract
            g = neinsum("kij,kn,jn->in", GPDeriv, z, u)  # ([nMonoms,nq,nu] . [nMonoms,nPt]) . (nu,nPt) -> [nq,nPt] Compute input

        # Compute - input dynamics
        xd += g

        if dx0 is not None:
            # Adjust for reference
            xd -= dx0

        return xd

    
    