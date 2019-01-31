from coreUtils import *
from polynomial import *
from relaxations.lasserre import *
from relaxations.constraints import *



class convexProg():

    def __init__(self, repr:polynomialRepr, solver:str='cvxopt', objective:polynomials=None):

        assert solver in ['cvxopt'], 'Solver not supported'
        assert (objective is None) or (repr is objective.repr)

        self.solver = solver
        self.repr = repr
        self.constraints=variableStruct(l=variableStruct(nCstr=0, cstrList=[]),q=variableStruct(nCstr=0, cstrList=[]),s=variableStruct(nCstr=0, cstrList=[]))
        self.__objective = polynomials(repr) if objective is None else objective

        self.isUpdate = False


    @property
    def objective(self):
        return self.__objective
    @objective.setter
    def objective(self,new):
        if isinstance(new, polynomials):
            self.__objective = new
        elif isinstance(new, np.ndarray):
            self.__objective.coeffs = new

        self.isUpdate = False

    def addCstr(self, newConstraint:Union[linearConstraint,socpConstraint,lasserreConstraint]):
        if isinstance(newConstraint, linearConstraint):
            self.constraints.l.nCstr += 1
            self.constraints.l.cstrList.append(newConstraint)
        elif isinstance(newConstraint, socpConstraint):
            self.constraints.q.nCstr += 1
            self.constraints.q.cstrList.append(newConstraint)
        elif isinstance(newConstraint, lasserreConstraint):
            self.constraints.s.nCstr += 1
            self.constraints.s.cstrList.append(newConstraint)
        else:
            raise NotImplementedError
        
        self.isUpdate = False
        
        return None

    def precomp(self):
        if self.solver == 'cvxopt':
            self.precomp_cvxopt()
        else:
            raise NotImplementedError
        
        self.isUpdate = True

        return None

    def precomp_cvxopt(self):
        return None #Nothing to do for the moment

    def solve(self, opts={}):
        if not self.isUpdate:
            self.precomp()

        if self.solver == 'cvxopt':
            return self.solve_cvxopt(opts)
        else:
            return None

    def solve_cvxopt(self,opts={}):
        # Currently focalised on sdp with linear constraints
        _opts = {}
        _opts.update(opts)
        
        assert self.constraints.q.nCstr == 0
        
        #Assemble
        


