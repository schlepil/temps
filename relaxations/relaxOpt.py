from coreUtils import *
from polynomial import *
from relaxations.lasserre import *
from relaxations.constraints import *
from relaxations.optUtils import *


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
        
        


