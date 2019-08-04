from coreUtils import *

class constraint:
    def __init__(self):
        pass

    def getCstr(self):
        raise NotImplementedError

    def evalCstr(self, x:np.array):
        # Evaluates the error at given points
        raise NotImplementedError

    def isValid(self, z: np.ndarray, atol=-coreOptions.absTolCstr):
        # Checks if constraint is verified for a certain points
        raise NotImplementedError
    
    def coeffs(self, *args, **kwargs):
        raise NotImplementedError

class linearConstraint(constraint):
    def __init__(self,Gl,hl):

        self.Gl = Gl
        self.hl = np.array(hl,dtype=nfloat).reshape((-1,1))

        if isinstance(Gl,np.ndarray):
            self.shape = self.Gl.shape
            self.isSparse = False
        elif isinstance(Gl,(sparse.coo_matrix,sparse.csc_matrix,sparse.csr_matrix)):
            self.Gl = sparse.coo_matrix(Gl)
            self.shape = self.Gl.shape
            self.isSparse = True
        else:
            raise TypeError
    
    def getCstr(self, isSparse=False):
        
        if not isSparse:
            if not self.isSparse:
                Gl = self.Gl.todense()
            else:
                Gl = self.Gl
        else:
            if self.isSparse:
                if isSparse == 'coo':
                    Gl = self.Gl
                elif isSparse == 'csr':
                    Gl = self.Gl.tocsr()
                else:
                    Gl = self.Gl.tocsc()
            else:
                if isSparse == 'coo':
                    Gl = sparse.coo_matrix(self.Gl)
                elif isSparse == 'csr':
                    Gl = sparse.csr_matrix(self.Gl)
                else:
                    Gl = sparse.csr_matrix(self.Gl)
        return Gl, self.hl
            
        
class socpConstraint(constraint):
    def __init__(self):
        raise NotImplementedError
