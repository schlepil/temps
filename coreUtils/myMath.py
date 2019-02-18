from coreUtils.coreUtilsImport import *

def normalizeEllip(P:np.ndarray):
    """
    Return a scaled version of P with determinant 1.
    :param P:
    :return:
    """
    assert len(P.shape)==2
    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T)
    
    return P*(1./det(P))**(1./P.shape[0])
    
    
    