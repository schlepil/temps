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

if __name__ == "__main__":
    
    import plotting as plot
    from plotting import plt
    
    # Get the polynomial representation which also decides on the maximal relaxation
    thisRepr = poly.polynomialRepr(2, 4)
    
    #Get the dynamical system
    pendSys = getSys(thisRepr, fileName=None)#"~/tmp/pendulumDict.pickle")
    
    #Get the trajectory
    xTraj = lambda t:narray([[np.pi], [0.]], dtype=nfloat)
    dxTraj = lambda t:narray([[0.], [0.]], dtype=nfloat)
    
    #Compute necessary input (here 0.)
    uRefTmp = pendSys.getUopt(xTraj(0,), dxTraj(0.),respectCstr=False, fullDeriv=True)
    uTraj = lambda t: uRefTmp.copy()

    #def __int__(self, fX: Callable, fU: Callable, nx: int, nu: int, fXd: Callable = None, tMin: float = 0., tMax: float = 1.):
    refTraj = traj.analyticTrajectory(xTraj, uTraj, 2, 1, dxTraj)
    print(refTraj(0.))
    
    # Get the input constraints along the refTraj
    pendSys.ctrlInput = dynSys.constraints.boxInputCstrLFBG(thisRepr, refTraj, 1, *getUlims())
    
    # Get a quadratic Lyapunov function, initialize it and perform some simulations
    lyapF = lyap.quadraticLyapunovFunction(pendSys)
    lyapF.P, K = lyapF.lqrP(np.identity(2), np.identity(1), refTraj.getX(0.))
    lyapF.alpha = 1.

    lyapF.P = myMath.normalizeEllip(lyapF.P)*5.
    
    xInit = plot.getV(lyapF.P, n=30, alpha=lyapF.alpha, endPoint=False)+refTraj.getX(0.)#regular points on surface
    
    ff,aa = plt.subplots(1,2, figsize=[1+2*4,4])
    
    plot.plotEllipse(aa[0], refTraj.getX(0.), lyapF.P, 1., faceAlpha=0.)
    
    #Get the integration function
    #def __call__(self, x:np.ndarray, u_:Union[np.ndarray,Callable], t:float=0., restrictInput:bool=True, mode:List[int]=[0,0], x0:np.ndarray=False):
    fInt = lambda t, x: pendSys(x, u_=-K, t=t, x0=refTraj.getX(0.))
    maxStep = 0.1
    
    # Define an end
    fTerminalConv = lambda t, x: float(lyapF.evalV(x.reshape((lyapF.nq,-1))-refTraj.getX(t),kd=False))-1e-5
    fTerminalConv.terminal = True
    fTerminalDiverge = lambda t, x:float(lyapF.evalV(x.reshape((lyapF.nq,-1))-refTraj.getX(t), kd=False))-20.
    fTerminalDiverge.terminal = True
    
    for k in range(xInit.shape[1]):
    
        sol = solve_ivp(fInt, [0.,1.], xInit[:,k], events=[fTerminalConv, fTerminalDiverge], vectorized=True, max_step=maxStep)

        aa[0].plot(sol.y[0,:], sol.y[1,:], 'k')
        aa[1].semilogy(sol.t, lyapF.evalV(sol.y-refTraj.getX(sol.t), kd=False), 'k')
    
    
    #Plot the constraint
    ff, aa = plt.subplots(1, 2, figsize=[1 + 2 * 4, 4])
    plot.plotEllipse(aa[0], refTraj.getX(0.), lyapF.P, lyapF.alpha, faceAlpha=0.)
    # First get the taylor
    fTaylorX0, gTaylorX0 = pendSys.getTaylorApprox(refTraj.getX(0.), 3)

    #Constraint
    polyCstr = poly.polynomial(thisRepr)
    polyCstr.coeffs = lyapF.getCstrWithDeg(gTaylorX0, 3, 3)

    #Objective
    uCtrlStar = np.array([[1.]]) #Normalized input
    uMonomStar = thisRepr.varNumsUpToDeg[0]
    objectArrayStr = lyapF.getObjectiveAsArray(fTaylorX0, gTaylorX0, taylorDeg=3, u=uCtrlStar, uMonom=uMonomStar )

    uCtrlLin, uMonomLin = pendSys.ctrlInput.getU(narray([2]), 0., P = lyapF.P, PG0 = ndot(lyapF.P_, gTaylorX0[0,:,:]), alpha=lyapF.alpha,
                                                 monomOut=True)
    objectArrayLin = lyapF.getObjectiveAsArray(fTaylorX0, gTaylorX0, taylorDeg=3, u=uCtrlLin, uMonoms=uMonomLin)


    Ngrid = 100
    plot.plotEllipse(aa[0], refTraj.getX(0.), lyapF.P, 1., faceAlpha=0.)
    aa[0].autoscale()
    xx,yy = plot.ax2Grid(aa[0], Ngrid)
    XX = np.vstack((xx.flatten(), yy.flatten()))
    valCstr = polyCstr.eval2(XX-refTraj.getX(0.))
    aa[0].contour(xx,yy, valCstr.reshape((Ngrid,Ngrid)), [0.])

    # Build up the optimization problem
    baseRelax = relax.lasserreRelax(thisRepr)
    baseRelax = relax.lasserreConstraint()
    probCVX = relax.convexProg(thisRepr)
    plot.plt.show()




    
    
    
    
    
    
    
    
    
    
    
    
    
    
