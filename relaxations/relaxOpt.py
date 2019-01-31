from coreUtils import *
from polynomial import *
from relaxations.lasserre import *
from relaxations.constraints import *



class polynomialProg():

    def __init__(self, repr:polynomialRepr, solver:str='cvxopt', objective:polynomials=None):

        assert solver in ['cvxopt'], 'Solver not supported'
        assert (objective is None) or (repr is objective.repr)

        self.solver = solver
        self.repr = repr
        self.constraints=variableStruct('l':variableStruct(nCstr=0),'q':variableStruct(nCstr=0),'s':variableStruct(nCstr=0))
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

    def addCstr(self, newConstraint:Union):

    def precomp(self):
        if self.solver == 'cvxopt':
            self.precomp_cvxopt()
        else:
            raise NotImplementedError

        return None

    def precomp_cvxopt(self):
        return None #Nothing to do for the moment

    def solve(self):
        if not self.isUpdate:
            self.precomp()

        if self.solver == 'cvxopt':
            return self.solve_cvxopt()
        else:
            return None

    def solve_cvxopt


