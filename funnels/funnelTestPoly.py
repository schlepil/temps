import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
    # Complicate the problem
    cplx = 2
    shapeP = 1

    if cplx == 0:
        R = P = nidentity(2, dtype=nfloat)
        G = narray([[0.],[1.]], dtype=nfloat)
    elif cplx == 1:
        R = plot.Rot(45./180*np.pi).astype(dtype=nfloat)
        P = mndot([R.T, narray([[2,0],[0,1]], dtype=nfloat), R])
        G = narray([[0.],[1.]], dtype=nfloat)
    elif cplx == 2:
        R = plot.Rot(45. / 180 * np.pi)
        P = mndot([R.T, narray([[2, 0], [0, 1]], dtype=nfloat), R])
        G = nzeros((2,2), dtype=nfloat)
        G[0,1] = 1.
        G[1,0] = 0.6
        G[1,1] = 0.9
    else:
        raise NotImplementedError
    
    if shapeP == 0:
        Ps = nidentity(2)
        alpha = 0.8
    elif shapeP == 1:
        Ps = np.array([[1.1,0.05],[0.05, 0.95]], dtype=nfloat)
        alpha = 0.8
        


    
    # Get the polynomial representation which also decides on the maximal relaxation
    # Let use full here
    pSys = getSysStablePos(2,6,P=P,G=G)
    
    thisRepr = pSys.repr  # Todo debug digits. there is an error somewhere

    lyapF = lyap.quadraticLyapunovFunctionTimed(pSys)
    
    # Set the interpolator
    #lyapF.interpolate = lyap.standardInterpolNoDeriv
    lyapF.interpolate = lyap.standardInterpol
    
    # evolving the Lyapunov function along the trajectory
    thisLyapEvol = lyap.noChangeLyap()
    
    myFunnel = distributedFunnel(pSys, lyapF, pSys.ctrlInput.refTraj, thisLyapEvol)
    
    myFunnel.compute(0.0, 0.3, (Ps, alpha))
    
    plot.plot2dConv(myFunnel, 0.0)
    
    plot.plot2dProof(myFunnel, 0.0)
    
    distributor.terminate()
    
    
    print(f"final funnel is \n P: \n {myFunnel.lyapFunc.getPnPdot(0.,True)[0]} \n P: \n {myFunnel.lyapFunc.getPnPdot(0.,True)[1]}")
    plot.plt.show()
    

