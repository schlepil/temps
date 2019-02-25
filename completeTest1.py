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
    # Let use full here
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
    cstrPolySep = poly.polynomial(thisRepr)
    cstrPolySep.coeffs = lyapF.getCstrWithDeg(gTaylorX0, 3, 3)

    #Objective
    uCtrlStar = np.array([[1.]]) #Normalized input
    uMonomStar = thisRepr.varNumsUpToDeg[0]
    objectArrayStar = lyapF.getObjectiveAsArray(fTaylorX0, gTaylorX0, taylorDeg=3, u=uCtrlStar, uMonom=uMonomStar)

    uCtrlLin, uMonomLin = pendSys.ctrlInput.getU(narray([2]), 0., P = lyapF.P, PG0 = ndot(lyapF.P_, gTaylorX0[0,:,:]), alpha=lyapF.alpha,
                                                 monomOut=True)
    objectArrayLin = lyapF.getObjectiveAsArray(fTaylorX0, gTaylorX0, taylorDeg=3, u=uCtrlLin, uMonom=uMonomLin)
    
    Ngrid = 200
    plot.plotEllipse(aa[0], refTraj.getX(0.), lyapF.P, 1., faceAlpha=0.)
    aa[0].autoscale()
    xx,yy = plot.ax2Grid(aa[0], Ngrid)
    XX = np.vstack((xx.flatten(), yy.flatten()))
    valCstr = cstrPolySep.eval2(XX-refTraj.getX(0.))
    aa[0].contour(xx,yy, valCstr.reshape((Ngrid,Ngrid)), [0.])

    # Evaluate the norm of the control input
    uNorm = ndot(uCtrlLin, np.vstack((np.zeros((1,XX.shape[1])), XX-refTraj.getX(0.)))).flatten()
    plot.plotEllipse(aa[1], refTraj.getX(0.), lyapF.P, 1., faceAlpha=0.)
    aa[1].contourf(xx,yy,uNorm.reshape((Ngrid,Ngrid)))
    aa[1].contour(xx, yy, uNorm.reshape((Ngrid, Ngrid)), [-10,-5,0,5,10])
    

    # Build up the optimization problem
    baseRelax = relax.lasserreRelax(thisRepr)
    cstrRelaxSep = relax.lasserreConstraint(baseRelax, cstrPolySep) # Here the 'plus' space is singled out -> negative input
    cstrPolyEllip = poly.polynomial(thisRepr)
    cstrPolyEllip.setQuadraticForm(-lyapF.P, thisRepr.varNumsPerDeg[1], narray([lyapF.alpha], dtype=nfloat), thisRepr.varNumsPerDeg[0]) #  x'.P.x<=alpha --> x'.(-P).x + alpha >= 0
    cstrRelaxEllip = relax.lasserreConstraint(baseRelax, cstrPolyEllip)
    # Exclude inner as zero always yields zero
    excludeInner = 0.1
    cstrPolyEllipInner = poly.polynomial(thisRepr)
    cstrPolyEllipInner.setQuadraticForm(lyapF.P, thisRepr.varNumsPerDeg[1], narray([-excludeInner*lyapF.alpha], dtype=nfloat),
                                   thisRepr.varNumsPerDeg[0])  # x'.P.x>=epsilon*alpha --> x'.P.x - epsilon*alpha >= 0
    cstrRelaxEllipInner = relax.lasserreConstraint(baseRelax, cstrPolyEllipInner)
    
    probCVX = relax.convexProg(thisRepr)
    probCVX.addCstr(baseRelax)
    probCVX.addCstr(cstrRelaxSep)
    probCVX.addCstr(cstrRelaxEllip)
    probCVX.addCstr(cstrRelaxEllipInner)
    
    #Get the objective
    #Solve for optimal input in the plus zone (negative input)
    probCVX.objective = -(objectArrayStar[0,:]+(-10.)*objectArrayStar[1,:]) #Polynomial approximation of the convergence. The more negative the higher
    # the convergence, if positive divergence -> inverse signs to get minimial convergence
    solPlusStar = probCVX.solve()
    xStarOpt = probCVX.checkSol(solPlusStar)
    
    # Plot
    ff,aa = plt.subplots(1,2, figsize=(1+2*4,4))
    plot.plotEllipse(aa[0], refTraj.getX(0.), lyapF.P, 1., faceAlpha=0.)
    plot.plotEllipse(aa[0], refTraj.getX(0.), lyapF.P, excludeInner*1., faceAlpha=0.)
    aa[0].autoscale()
    xx,yy = plot.ax2Grid(aa[0], Ngrid)
    XX = np.vstack((xx.flatten(), yy.flatten()))
    DXX = XX-refTraj.getX(0.)
    
    ## Optimal control
    # Get the convergence values
    dVx = probCVX.objective.eval2(DXX).squeeze()
    #Get the index for the constraints
    idxFeasible = nones((XX.shape[1],), dtype=np.bool_)
    idxFeasible = np.logical_and(idxFeasible, cstrPolySep.eval2(DXX).squeeze()>=0.)
    idxFeasible = np.logical_and(idxFeasible, cstrPolyEllip.eval2(DXX).squeeze() >= 0.)
    idxFeasible = np.logical_and(idxFeasible, cstrPolyEllipInner.eval2(DXX).squeeze() >= 0.)

    dVx[~idxFeasible] = nmin(dVx[idxFeasible])

    aa[0].contour(xx,yy,dVx.reshape((Ngrid,Ngrid)))
    aa[0].plot(xStarOpt[0,:], xStarOpt[1,:], '*r')
    
    ##
    #For the linear control input
    probCVX.objective = -(objectArrayLin[0,:]+objectArrayLin[1,:])
    solPlusLin = probCVX.solve()
    xLinOpt = probCVX.checkSol(solPlusLin)
    
    # Get the convergence values
    dVx = probCVX.objective.eval2(DXX).squeeze()
    #Get the index for the constraints
    idxFeasible = nones((XX.shape[1],), dtype=np.bool_)
    idxFeasible = np.logical_and(idxFeasible, cstrPolySep.eval2(DXX).squeeze()>=0.)
    idxFeasible = np.logical_and(idxFeasible, cstrPolyEllip.eval2(DXX).squeeze() >= 0.)
    idxFeasible = np.logical_and(idxFeasible, cstrPolyEllipInner.eval2(DXX).squeeze() >= 0.)
    dVx[~idxFeasible] = nmin(dVx[idxFeasible])

    aa[1].contour(xx,yy,dVx.reshape((Ngrid,Ngrid)))
    aa[1].plot(xLinOpt[0,:], xLinOpt[1,:], '*r')

    plot.plotEllipse(aa[1], refTraj.getX(0.), lyapF.P, 1., faceAlpha=0.)
    plot.plotEllipse(aa[1], refTraj.getX(0.), lyapF.P, excludeInner*1., faceAlpha=0.)
    aa[0].autoscale()

    plot.plt.show()




    
    
    
    
    
    
    
    
    
    
    
    
    
    
