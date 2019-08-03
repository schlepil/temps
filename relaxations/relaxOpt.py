from coreUtils import *
from polynomial import *
from relaxations.lasserre import *
from relaxations.constraints import *
from relaxations.optUtils import *
from relaxations.rref import robustRREF
from scipy.optimize import minimize as sp_minimize
from scipy import optimize

class convexProg():

    def __init__(self, repr:polynomialRepr, solver:str='cvxopt', objective:polynomial=None, firstVarIsOne:bool=True):

        assert solver in ['cvxopt'], 'Solver not supported'
        assert (objective is None) or (repr is objective.repr)

        self.solver = solver
        self.repr = repr
        self.constraints=variableStruct(l=variableStruct(nCstr=0, cstrList=[]),q=variableStruct(nCstr=0, cstrList=[]),s=variableStruct(nCstr=0, cstrList=[]),eq=variableStruct(nCstr=0, cstrList=[]))
        self.__objective = polynomial(repr) if objective is None else objective
        
        self.firstVarIsOne = firstVarIsOne

        self.isUpdate = False


    @property
    def objective(self):
        return self.__objective
    @objective.setter
    def objective(self,new):
        if isinstance(new, polynomial):
            self.__objective = new
        elif isinstance(new, np.ndarray):
            self.__objective.coeffs = new

        self.isUpdate = False

    def addCstr(self, newConstraint:Union[linearConstraint,socpConstraint,lasserreRelax,lasserreConstraint]):
        if isinstance(newConstraint, linearConstraint):
            self.constraints.l.nCstr += 1
            self.constraints.l.cstrList.append(newConstraint)
        elif isinstance(newConstraint, socpConstraint):
            self.constraints.q.nCstr += 1
            self.constraints.q.cstrList.append(newConstraint)
        elif isinstance(newConstraint, (lasserreRelax,lasserreConstraint)):
            self.constraints.s.nCstr += 1
            self.constraints.s.cstrList.append(newConstraint)
        else:
            raise NotImplementedError
        
        self.isUpdate = False
        
        return None
    
    def removeCstr(self, type, idx):
        if type == 'l':
            self.constraints.l.nCstr -= 1
            self.constraints.l.cstrList.pop(idx)
        elif type == 'q':
            self.constraints.q.nCstr -= 1
            self.constraints.q.cstrList.pop(idx)
        elif type == 's':
            self.constraints.s.nCstr -= 1
            self.constraints.s.cstrList.pop(idx)
        elif type == 'eq':
            self.constraints.eq.nCstr -= 1
            self.constraints.eq.cstrList.pop(idx)
        else:
            raise TypeError


    def precomp(self):
        if self.solver == 'cvxopt':
            self.precomp_cvxopt()
        else:
            raise NotImplementedError
        
        self.isUpdate = True

        return None

    def precomp_cvxopt(self):
        return None #Nothing to do for the moment

    def extractOptSol(self, sol:Union[np.ndarray, dict], **kwargs):
        """
        Implements ftp://nozdr.ru/biblio/kolxoz/M/MOac/Henrion%20D.,%20Garulli%20A.%20(eds.)%20Positive%20polynomials%20in%20control%20(Springer,%202005)(ISBN%203540239480)(O)(310s)_MOc_.pdf#page=289
        but does a quick-check first
        :param sol:
        :param kwargs:
        :return:
        """

        opts_ = {'relTol': 1e-2, 'reOptimize':False}
        opts_.update(kwargs)

        if isinstance(sol, dict):
            x_ = sol["x_np"].copy().squeeze()
        else:
            x_ = narray(sol, dtype=nfloat).copy().squeeze()

        # First check if there is only one solution stored

        nMonomsH_ = self.repr.varNumsUpToDeg[self.repr.maxDeg//2].size

        varMat = x_[self.repr.idxMat[:nMonomsH_, :nMonomsH_].flatten()].reshape((nMonomsH_, nMonomsH_)) # -> Matrix corresponding to the solution

        v,e = eigh(varMat, check_finite=False)
        vMax = v[-1]
        vRel = v*(1./vMax)

        if __debug__:
            assert vRel[0] > -opts_['relTol']

        if (nall(vRel[:-1]<opts_['relTol'])):
            # There is only one optimal solution
            xSol = varMat[1:self.repr.varNumsUpToDeg[1].size,[0]]
            optimalCstr = nzeros((0,self.repr.nMonoms))
            if opts_['reOptimize']:
                # Optimize the solution locally to reduce rounding errors
                raise NotImplementedError
            #The simple solution is exact enough
            
            #Check all constraints
            isValid = True
            zSol = self.repr.evalAllMonoms(xSol)

            atol = -absTolCstr #make a global variable
            for aCstr in self.constraints.l.cstrList+self.constraints.q.cstrList+self.constraints.s.cstrList:
                    isValid = isValid and bool(aCstr.isValid(zSol, atol=atol))

            if not isValid:
                       isValid=True
                       xSolnew=self.localSolve(xSol)
                       zSolnew=self.repr.evalAllMonoms(xSolnew)
                       for aCstr in self.constraints.l.cstrList + self.constraints.q.cstrList + self.constraints.s.cstrList:
                           isValid = isValid and bool(aCstr.isValid(zSolnew, atol=atol))
                           if not isValid:
                               raise RuntimeError
                           else:
                               xSol=xSolnew
            assert abs(atol) < 1e-1
            if __debug__:
                print(f"Found valid solution for atol : {atol}")
            
            return xSol, None, (None, None, None)
        else:
            # Here it is more tricky
            # We will return optimal solutions, however they are not unique
            # It more that all optimal solutions lie
            print('wtfffffffffffffff')
            #First get the cholesky decomp if all positive eigenvalues
            ind = np.where(vRel>opts_['relTol'])[0]
            thisRank = ind.size
            v2 = v[ind]**.5
            e2 = e[:, ind]
            V = nmultiply(e2, v2)

            # Get column echelon form
            # Scipy only provides row echelon form
            #U, monomBase = sy.Matrix(V.T).rref()# <- This is ; U, monomBase = sy.Matrix(V.T).rref(simplify=True, iszerofunc=lambda x:x**2<1e-20) #is correct but slow
            #U = narray(U, dtype=nfloat).T
            cond_ = (v2[0]/v2[-1])
            
            try:
                Ue, varMonomBase = robustRREF(V.T, cond_/10., tol0_=1e-8, fullOut=False) #"Exact" rref
            except RuntimeError:
                try:
                    Ue, varMonomBase = robustRREF(V.T, cond_/100., tol0_=1e-8, fullOut=False)  # "Exact" rref
                except RuntimeError:
                       # Ue, varMonomBase = robustRREF(V.T, cond_/500., tol0_=1e-7, fullOut=False)  # "Exact" rref
                       try:
                           Ue, varMonomBase = robustRREF(V.T, cond_ / 500., tol0_=1e-7, fullOut=False)
                       except RuntimeError:
                           Ue, varMonomBase = robustRREF(V.T, cond_ / 1., tol0_=1e-5, fullOut=False)
                           print('cest bizzard')
            Ue = Ue.T.copy()

            varMonomBase = narray(varMonomBase, dtype=nintu).reshape((-1,))
            monomIsBase = nones((Ue.shape[0],), dtype=np.bool_)
            monomIsBase[varMonomBase] = False

            allMonomsHalf = self.repr.listOfMonomials[:nMonomsH_]

            firstDegMonoms = self.repr.varNumsPerDeg[1]
            
            relTol = 1e-6 #Set relative zero, increment if infeasible up to max
            relTolMax = 1e-1
            isNOk_ = 1
            while ((relTol <= relTolMax) and (isNOk_!=0)):
                relTol *= 10.
                U = Ue.copy()
                #Set to relative zero
                U[nabs(U) <= nmax(nabs(U), axis=1, keepdims=True)*relTol] = 0.
            
                monomNotBaseNCoeffs = []
                for k,aIsNotBase in enumerate(monomIsBase):
                    if not aIsNotBase:
                        continue
                    newList = [allMonomsHalf[k], [[],[]], nzeros((self.repr.nMonoms,1))]
                    newList[2][varMonomBase,0] = U[k,:]
                    for i, aIdx in enumerate(U[k,:]):
                        if abs(aIdx) == 0.:
                            continue
                        newList[1][0].append( allMonomsHalf[varMonomBase[i]] )
                        newList[1][1].append( aIdx )
                        
                    monomNotBaseNCoeffs.append(newList)
            
                #Get the multiplier matrices
                NList = nempty((self.repr.nDims, thisRank, thisRank), nfloat)
                ###NList = nempty((thisRank, thisRank, thisRank), nfloat)

                isNOk_ = 1
                for i, iMonom in enumerate(firstDegMonoms): #The first one is allows the constant value
                    if isNOk_>=2:
                        break
                ###for i, iMonom in enumerate(monomBase):  # The first one is allows the constant value
                    for j, jMonom in enumerate(varMonomBase):
                        if isNOk_ >= 2:
                            break
                        idxU = self.repr.idxMat[iMonom, jMonom]
                        if idxU<nMonomsH_:
                            NList[i,j,:] = U[idxU,:]
                            isNOk_ = 0
                        else:
                            # Recursively replace until all monomials are in the base
                            xiwjMonomNCoeff = [[idxU],[self.repr.listOfMonomials[idxU]], [1.0]]
                            while ((not all([aIdx<nMonomsH_ for aIdx in xiwjMonomNCoeff[0]])) and (isNOk_<2)):
                                for k, aIdx in enumerate(xiwjMonomNCoeff[0]):
                                    # Seek the monomial to simplify
                                    isNOk_ = 1
                                    if aIdx>=nMonomsH_:
                                        toReplace = [aaList.pop(k) for aaList in  xiwjMonomNCoeff]
                                    
                                    #Try relacing once and append
                                    replacement = [[],[],[]]
                                    for aMonomNBase in monomNotBaseNCoeffs:
                                        nRMonom = toReplace[1]-aMonomNBase[0]
                                        if all(nRMonom>=0):
                                            for (rMonomAdd, rCoeff) in zip(*aMonomNBase[1]):
                                                replacement[1].append(nRMonom+rMonomAdd)
                                                replacement[0].append( self.repr.monom2num[list2int(replacement[1][-1], self.repr.digits)] )
                                                replacement[2].append(rCoeff*toReplace[2])
                                            for jj in range(3):
                                                xiwjMonomNCoeff[jj].extend(replacement[jj])
                                            isNOk_ = 0
                                            break
                                    if isNOk_==0:
                                        break
                                    else:
                                        # The solution/tolerance combo is not sufficient to generate the minimizer with this approach
                                        isNOk_=2
                                        #relTol *= 10.
                                        break
                                        
                            #Apply
                            if isNOk_ == 0:
                                NList[i, j, :] = 0.
                                for thisIdxU,_,thisCoeff in zip(*xiwjMonomNCoeff):
                                    NList[i, j, :] += U[thisIdxU, :]*thisCoeff
            if isNOk_ != 0:
                #No solution can be extracted -> return exect constraints
                return None, None, (varMonomBase, Ue, relTol)
                    
                        

            # Here comes the tricky part:
            # we need to get the common eigenvalues of all Ni -> get a "random" linear combination
            # Therefore all optimal solutions respect some constraints, but they are not unique...
            N = nsum(NList,axis=0)/NList.shape[0]

            T,Q = schur(N)
            
            #Now compute the actual solutions or better the representations of it
            xSol = nempty((self.repr.nDims, thisRank), dtype=nfloat)
            #Sum up
            for i in range(NList.shape[0]):
                for j in range(Q.shape[1]):
                ###for j in range(Q.shape[1]):
                    xSol[i,j] = neinsum('m,mn,n', Q[:,j], NList[i,:,:], Q[:,j])
                    ###xSol[j, i] = neinsum('m,mn,n', Q[:, j], NList[i, :, :], Q[:, j])
            
            # Check if all constraints are respected (Here it can happen that they are not)
            isValid = nones((xSol.shape[1],), dtype=np.bool_)
            print('xSol',xSol)
            zSol = self.repr.evalAllMonoms(xSol)
            #Check all constraints
            # atol = -1e-6
            # for aCstr in self.constraints.l.cstrList + self.constraints.q.cstrList + self.constraints.s.cstrList:
            #     isValid = isValid and bool(aCstr.isValid(zSol, atol=atol))
            #
            # if not isValid:
            #     isValid = True
            #     xSolnew = self.localSolve(xSol)
            #     zSolnew = self.repr.evalAllMonoms(xSolnew)
            #     for aCstr in self.constraints.l.cstrList + self.constraints.q.cstrList + self.constraints.s.cstrList:
            #         isValid = isValid and bool(aCstr.isValid(zSolnew, atol=atol))
            #         if not isValid:
            #             raise RuntimeError
            #         else:
            #             xSol = xSolnew
            # atol = -1.e-6
            # while True:
            #     #assert abs(atol)<1e-1
            #
            #     for aCstr in self.constraints.l.cstrList+self.constraints.q.cstrList+self.constraints.s.cstrList:
            #         isValid = np.logical_and(isValid, aCstr.isValid(zSol, atol=atol))
            #
            #     # TODO check if this is a proper solution
            #     if nany(isValid) :
            #         break
            #     else:
            #         print('olaolaola')
            #         for i in range(xSol.shape[0]):
            #             xSol[i,:]=self.localSolve(xSol[i,:])
            #             #xSol_i.append(self.localSolve(xSol[i,:]))
            #         zSol=self.repr.evalAllMonoms(xSol)
            #         isValid[:] = True
            #         print('ok',atol)
            atol=-1e-6

            for aCstr in self.constraints.l.cstrList + self.constraints.q.cstrList + self.constraints.s.cstrList:
                isValid = np.logical_and(isValid, aCstr.isValid(zSol, atol=atol))

            if not nany(isValid):
                        isValid=nones((xSol.shape[1],), dtype=np.bool_)
                        xSolnew=nzeros(xSol.shape)
                        for i in range(xSol.shape[1]):
                            xSolnew[:,i]=self.localSolve(xSol[:, i])
                            #xSol_i.append(self.localSolve(xSol[i,:]))
                        zSolnew=self.repr.evalAllMonoms(xSolnew)
                        for aCstr in self.constraints.l.cstrList + self.constraints.q.cstrList + self.constraints.s.cstrList:
                            isValid = np.logical_and(isValid, aCstr.isValid(zSolnew, atol=atol))
                        print('hello',isValid)
                        if not nany(isValid):
                           raise RuntimeError
                        else:
                           xSol=xSolnew
            if __debug__:
                print(f"Found valid solution for atol : {atol}")
            
            #Only return valid ones
            xSol = xSol[:, isValid]
            
            #Finally construct the constraint polynomials
            if U.shape[1] == 1:
                # No need to compute them as the result is optimal -> no further search necessary
                optimalCstr = nzeros((0, self.repr.nMonoms))
            else:
                optimalCstr = nzeros((U.shape[0]-thisRank, self.repr.nMonoms))
                # Fill up
                optimalCstr[:, varMonomBase]=U[monomIsBase,:]
                idxC = 0
                for k, nBase in enumerate(monomIsBase):
                    if nBase:
                        optimalCstr[idxC,k] = -1. #Set the equality
                        idxC += 1
            
        return xSol, optimalCstr, (varMonomBase, U, relTol)


    def localSolve(self, xsol):
        # TODO @xiao take into account possible linear / socp constraints
        # TODO @xiao reduce the computational complexity by only taking non-zero coefficients (monomials up to the highest polynomial in the constraints)
        # TODO @xiao make this more generic, so that solvers can be easily exchanged, maybe by creating a local-solver object "sugar-coating" existing solvers
        
        Amat = nzeros((len(self.constraints.s.cstrList[1:]), self.__objective.coeffs.size), dtype=nfloat)
        for i, acstr in enumerate(self.constraints.s.cstrList[1:]):
           # print("acstr.poly.coeffs",acstr.poly.coeffs)
            #Amat[i, :] = acstr.poly.coeffs.copy() #Unnecessary copy, will be copied anyways
            Amat[i, :] = acstr.poly.coeffs
        this_cstr = {'type': 'ineq', 'fun': lambda x: ndot(Amat, self.repr.evalAllMonoms(x.reshape((-1, 1)))).squeeze()}
        gx = lambda x: ndot(self.objective.coeffs, self.repr.evalAllMonoms(x))
      #  res = sp_minimize(gx, xsol, method='COBYLA',constraints=this_cstr,options={'rhobeg': 1.0, 'maxiter': 1000, 'disp': False, 'catol': 1e-7})
        #res = sp_minimize(gx, xsol, method='COBYLA', constraints=this_cstr, options={'rhobeg': 1.0, 'maxiter': 1000, 'disp': False, 'tol': 1e-7})
        res = sp_minimize(gx, xsol, method='COBYLA', tol=0.9*absTolCstr, constraints=this_cstr) #Leave some margin
        #res = sp_minimize(gx, xsol, method='COBYLA', constraints=this_cstr)
        assert res.success
        #cstrverif[1.979833e-01 - 1.993135e-06, 1.664354e+00 - 4.200682e-06]
        if __debug__:
            print('cstrverif',this_cstr['fun'](res.x))
            if not nall(this_cstr['fun'](res.x)>-absTolCstr):
                print('shit')
        return res.x
    
    def localSolve(self, *args, **kwargs):
        return self.recalcul(*args, **kwargs)


    def checkSol(self, sol:Union[np.ndarray, dict], **kwargs):
        
        if isinstance(sol, dict):
            x_ = sol["x_np"].copy().squeeze()
        else:
            x_ = narray(sol, dtype=nfloat).copy().squeeze()

        print(np.inner(self.objective.coeffs, x_))
        print(self.objective.eval2(x_[1:3].reshape((-1, 1))))

        nMonomsH_ = self.repr.varNumsUpToDeg[self.repr.maxDeg//2].size

        varMat = x_[self.repr.idxMat[:nMonomsH_, :nMonomsH_].flatten()].reshape((nMonomsH_, nMonomsH_)) # -> Matrix corresponding to the solution

        xStar = x_[self.repr.varNumsUpToDeg[0].size:self.repr.varNumsUpToDeg[1].size].reshape((-1,1))
        zHalfStar = evalMonomsNumba(xStar, self.repr.varNum2varNumParents[:nMonomsH_, :])
        zMatStar = np.outer(zHalfStar.squeeze(), zHalfStar.squeeze())
        
        print(f"Difference in moment matrix is \n {varMat-zMatStar}")
        uz,sz,vz = svd(varMat-zMatStar)
        print(f"With eigen values and vectors \n{sz}\n{sz}")
        print(f"The resulting difference in the objective funciton is \n {np.inner(self.objective.coeffs.squeeze(), nsum(uz,sz, axis=1, keepdims=False))}")
        
        return xStar



    def solve(self, isSparse=False, opts={}):
        if not self.isUpdate:
            self.precomp()

        if self.solver == 'cvxopt':
            return self.solve_cvxopt(isSparse, opts=opts)
        else:
            return None

    def solve_cvxopt(self,isSparse=False,primalstart=None, dualstart=None, opts={}):
        # Currently focalised on sdp with linear constraints
        _opts = {}
        _opts.update(opts)
        
        solvers.options.update(_opts) #Set them for cvx
        
        if isSparse == True:
            raise NotImplementedError

        # Assemble
        obj = matrix(self.objective.coeffs)
        constantValue = 0.
        if self.firstVarIsOne:
            #The first variable corresponds to zero order polynomial -> is always zero and can be added to the constant terms
            constantValue = float(obj[0])
            obj = obj[1:]
        
        dims = {'l':0, 'q':[], 's':[]}
        
        G = []
        h = []
        
        if self.constraints.l.nCstr:
            for aCstr in self.constraints.l.cstrList:
                thisGlhl = aCstr.getCstr(isSparse)
                dims['l'] += thisGlhl[1].size
                G.append(thisGlhl[0])
                h.append(thisGlhl[1])
        
        if self.constraints.q.nCstr:
            raise NotImplementedError
            
        
        if self.constraints.s.nCstr:
            for aCstr in self.constraints.s.cstrList:
                thisGshs = aCstr.getCstr(isSparse)
                dims['s'].append(int((thisGshs[1].size)**.5))
                G.append(thisGshs[0])
                h.append(thisGshs[1])
        
        G = np.vstack(G)
        h = np.vstack(h)
        
        if self.firstVarIsOne:
            # The first variable corresponds to zero order polynomial -> is always zero and can be added to the constant terms
            h -= G[:,[0]]
            G = G[:,1:]
        
        G = matrix(G)
        h = matrix(h)
        
        sol = solvers.conelp(obj, G,h,dims,primalstart=primalstart,dualstart=dualstart)
        
        #Add
        if sol['status'] == 'optimal':
            sol['primal objective'] += constantValue
            sol['dual objective'] += constantValue
        
        sol['x_np'] = narray(sol['x']).astype(nfloat).reshape((1,-1))
        if self.firstVarIsOne:
            sol['x_np'] = np.hstack(([[1]],sol['x_np']))
        
        return sol
        
        


