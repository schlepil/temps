from coreUtils import *
from systems.polySysTest import getSysStablePos, getUlims
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
    pSys = getSysStablePos(2,6)
    
    thisRepr = pSys.repr  # Todo debug digits. there is an error somewhere

    lyapF = lyap.quadraticLyapunovFunctionTimed(pSys)
    
    # Set the interpolator
    lyapF.interpolate = lyap.standardInterpolNoDeriv
    
    # evolving the Lyapunov function along the trajectory
    thisLyapEvol = lyap.noChangeLyap()
    
    myFunnel = distributedFunnel(pSys, lyapF, pSys.ctrlInput.refTraj, thisLyapEvol)
    
    myFunnel.compute(0.0, 1.0, (nidentity(2), 0.8))
    

