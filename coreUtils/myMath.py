from coreUtils.coreUtilsImport import *

def normalizeEllip(P:np.ndarray):
    """
    Return a scaled version of P sucht that the hypervolume of {x|x'.P.x <= 1} = 1
    :param P:
    :return:
    """
    assert len(P.shape)==2
    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T)
    
    volP = 1./det(P)**.5
    
    return P/(volP**P.shape[0])
    
    
    