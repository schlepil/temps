from coreUtils import *
from dynamicalSystems import polynomialSys
from dynamicalSystems.inputs import boxInputCstrLFBG
from polynomial import polynomialRepr, polynomial, polyFunction
from polynomial import list2int

def getUlims():
    return -1e-6,1e-6

def getSys(repr: polynomialRepr):
    
    nDim = repr.nDims
    
    fCoeffs = nzeros((nDim, repr.nMonoms), dtype=nfloat)
    gCoeffs = nzeros((repr.nMonoms, nDim, 1), dtype=nfloat)
    
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
    
    gCoeffs[0,nDim-1,0] = 1.
    
    qS = sy.symbols(f"q:{nDim}")
    qM =sy.Matrix(nzeros((nDim,1)))
    for i in range(nDim):
        qM[i,0] = qS[i]
    uS = sy.symbols(f"u:1")
    uM = sy.Matrix(nzeros((1, 1)))
    uM[0,0] = uS[0]
    
    pSys = polynomialSys(repr, fCoeffs, gCoeffs, qM, uM, 3)
    
    return pSys


def getSysStablePos(nDims:int, deg:int):
    import polynomial as poly
    import trajectories as traj

    repr = poly.polynomialRepr(nDims, deg)

    # Get the dynamical system
    pSys = getSys(repr)

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