from coreUtils import *
from polynomial import *
from relaxations.lasserre import *
from relaxations.constraints import *
from relaxations.optUtils import *

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

        opts_ = {'relTol': 1e-6}
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

        if nall(v[:-1]<opts_['relTol']):
            # There is only one optimal solution
            xSol = varMat[1:self.repr.varNumsUpToDeg[1].size,[0]]
            optimalCstr = nzeros((0,self.repr.nMonoms))
            if opts_['reOptimize']:
                # Optimize the solution locally to reduce rounding errors
                raise NotImplementedError
        else:
            # Here it is more tricky
            # We will return optimal solutions, however they are not unique
            # It more that all optimal solutions lie

            #First get the cholesky decomp if all positive eigenvalues
            ind = np.where(vRel>opts_['relTol'])
            thisRank = np.sum(ind)
            v2 = v[ind]**.5
            e2 = e[:, ind]
            V = nmultiply(e2, v2)

            # Get column echelon form
            # Scipy only provides row echelon form
            U, monomBase = sy.Matrix(V.T).rref()
            U = narray(monomBase, dtype=nfloat).T
            monomBase = narray(monomBase, dtype=nintu).squeeze()

            #Get the multiplier matrices
            NList = nempty((thisRank, thisRank, thisRank), nfloat)

            for i, iMonom in enumerate(monomBase):
                for j, jMonom in enumerate(monomBase):
                    Ni[i,j,:] = U[self.repr.idxMat[iMonom, jMonom],:]

            # Here comes the tricky part:
            # we need to get the common eigenvalues of all Ni -> get a "random" linear combination
            # Therefore all optimal solutions respect some constraints, but they are not unique...
            N = nsum(NList,axis=0)

            T,Q = schur(N)









        return xSol, optimalCstr








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
        
        


