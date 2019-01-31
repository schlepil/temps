from coreUtils import *

class constraint:
    def __init__(self):
        pass

    def getCstr(self):
        raise NotImplementedError

class linearConstraint(constraint):
    def __init__(self, A, b):

        self.A = A
        self.b = np.array(b,dtype=nfloat).reshape((-1,1))

        if isinstance(A, np.ndarray):
            self.shape = self.A.shape
            self.isSparse = False
        elif isinstance(A, (sparse.coo_matrix, sparse.csc_matrix,sparse.csr_matrix)):
            self.A = sparse.coo_matrix(A)
            self.shape = self.A.shape
            self.isSparse = True
        else:
            raise TypeError
    
    def getCstr(self, isSparse=False):
        
        if not isSparse:
            if not self.isSparse:
                A = self.A.todense()
            else:
                A = self.A
        else:
            if self.isSparse:
                if isSparse == 'coo':
                    A = self.A
                elif isSparse == 'csr':
                    A = self.A.tocsr()
                else:
                    A = self.A.tocsc()
            else:
                if isSparse == 'coo':
                    A = sparse.coo_matrix(self.A)
                elif isSparse == 'csr':
                    A = sparse.csr_matrix(self.A)
                else:
                    A = sparse.csr_matrix(self.A)
        return A, self.b
            
        
class socpConstraint(constraint):
    def __init__(self):
        raise NotImplementedError
