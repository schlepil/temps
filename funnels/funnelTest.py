from coreUtils import *
from systems.pendulum import getSys, getUlims
import polynomial as poly
import dynamicalSystems as dynSys
import trajectories as traj
import Lyapunov as lyap
import relaxations as relax

import plotting as plot
from plotting import plt

from scipy.integrate import solve_ivp

from funnels import *

#from parallelChecker import probSetter, solGetter, probQueues, solQueues


if __name__ == "__main__":
    # Get the polynomial representation which also decides on the maximal relaxation
    # Let use full here
    thisRepr = poly.polynomialRepr(2, 6, digits=1)  # Todo debug digits. there is an error somewhere

    # Get the dynamical system
    pendSys = getSys(thisRepr, fileName=None)  # "~/tmp/pendulumDict.pickle")

    # Get the trajectory
    xTraj = lambda t:narray([[np.pi], [0.]], dtype=nfloat)
    dxTraj = lambda t:narray([[0.], [0.]], dtype=nfloat)

    # Compute necessary input (here 0.)
    uRefTmp = pendSys.getUopt(xTraj(0, ), dxTraj(0.), respectCstr=False, fullDeriv=True)
    uTraj = lambda t:uRefTmp.copy()

    # def __int__(self, fX: Callable, fU: Callable, nx: int, nu: int, fXd: Callable = None, tMin: float = 0., tMax: float = 1.):
    refTraj = traj.analyticTrajectory(xTraj, uTraj, 2, 1, dxTraj)
    
    # Get the input constraints along the refTraj
    pendSys.ctrlInput = dynSys.constraints.boxInputCstrLFBG(thisRepr, refTraj, 1, *getUlims())
    
    lyapF = lyap.quadraticLyapunovFunctionTimed(pendSys)
    
    # Set the interpolator
    lyapF.interpolate = lyap.standardInterpol
    
    # Get some initial guess
    P, K = lyapF.lqrP(np.identity(2), np.identity(1), refTraj.getX(0.))
    P = myMath.normalizeEllip(P)
    
    # evolving the Lyapunov function along the trajectory
    thisLyapEvol = lyap.quadLyapTimeVaryingLQR(pendSys, refTraj, nidentity(pendSys.nq, dtype=nfloat), 0.1*nidentity(pendSys.nu, dtype=nfloat))
    
    myFunnel = distributedFunnel(pendSys, lyapF, refTraj, thisLyapEvol)
    
    myFunnel.compute(0.0, 1.0, (P, 1.))
    

