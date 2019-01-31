from coreUtils import *

class constraint:
    def __init__(self):
        pass

    def getCstr(self):
        raise NotImplementedError

class linearConstraint(constraint):
    def __init__(self, A, b):

        self.A = A
        self.b = b

        if isinstance(A, np.ndarray):
            self.shape = self.A.shape
            self.isSparse = False
        elif isinstance(A, (sparse.coo_matrix, sparse.csc_matrix,sparse.csr_matrix)):
            self.shape = self.A.shape
            self.isSparse = True
        elif isinstance(A, cvxopt.M)

