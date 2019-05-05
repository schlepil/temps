from coreUtils import *
from dynamicalSystems import polynomialSys
from dynamicalSystems.inputs import boxInputCstrLFBG
from polynomial import polynomialRepr, polynomial, polyFunction
from polynomial import list2int

limL_ = -1.
limU_ =  1.
nu_ = None

def getUlims():
    return nu_*[limL_],nu_*[limU_]

def getSys(repr: polynomialRepr, P=None, G=None):
    
    global nu_
    
    nDim = repr.nDims
    
    fCoeffs = nzeros((nDim, repr.nMonoms), dtype=nfloat)
    if G is None:
        gCoeffs = nzeros((repr.nMonoms, nDim, 1), dtype=nfloat)
    else:
        gCoeffs = nzeros((repr.nMonoms, nDim, G.shape[1]), dtype=nfloat)
    
    #Populate
    expAsList = nzeros((nDim,), nintu)
    for i in range(nDim):
        expAsList[:] = 0
        expAsList[i] = 1
        idx = repr.monom2num[list2int(expAsList)]
        fCoeffs[i,idx] = -1.
        expAsList[i] = 3
        idx = repr.monom2num[list2int(expAsList)]
        fCoeffs[i, idx] = 1.
    
    if G is None:
        gCoeffs[0,nDim-1,0] = 1.
    else:
        gCoeffs[0,:,:] = G
    
    if P is not None:
        Ci = inv(cholesky(P))
        
        for i in range(nDim):
            fCoeffs[i,:] = repr.doLinCoordChange(fCoeffs[i,:].copy(), Ci)

        for i in range(nDim):
            for j in range(gCoeffs.shape[2]):
                gCoeffs[:,i,j] = repr.doLinCoordChange(gCoeffs[:,i,j].copy(), Ci)
    
    qS = sy.symbols(f"q:{nDim}")
    qM =sy.Matrix(nzeros((nDim,1)))
    for i in range(nDim):
        qM[i,0] = qS[i]
    uS = sy.symbols(f"u:{gCoeffs.shape[2]}")
    uM = sy.Matrix(nzeros((1, gCoeffs.shape[2])))
    for i in range(gCoeffs.shape[2]):
        uM[i,0] = uS[i]
    nu_ = gCoeffs.shape[2]
    
    pSys = polynomialSys(repr, fCoeffs, gCoeffs, qM, uM, 3)
    
    return pSys


def getSysStablePos(nDims:int, deg:int, P=None, G=None):
    import polynomial as poly
    import trajectories as traj

    repr = poly.polynomialRepr(nDims, deg)

    # Get the dynamical system
    pSys = getSys(repr, P, G)

    # Get the trajectory
    xTraj = lambda t: nzeros((nDims,1), dtype=nfloat)
    dxTraj = xTraj

    # Compute necessary input (here 0.)
    uRefTmp = pSys.getUopt(xTraj(0.), dxTraj(0.), respectCstr=False)
    uTraj = lambda t: uRefTmp.copy()

    # def __int__(self, fX: Callable, fU: Callable, nx: int, nu: int, fXd: Callable = None, tMin: float = 0., tMax: float = 1.):
    refTraj = traj.analyticTrajectory(xTraj, uTraj, nDims, 1, dxTraj)
    print(refTraj(0.))

    # Get the input constraints along the refTraj
    pSys.ctrlInput = boxInputCstrLFBG(repr, refTraj, 1, *getUlims())

    return pSys