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
    thisRepr = poly.polynomialRepr(2, 6, digits=1)#Todo debug digits. there is an error somewhere
    
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
    lyapF.alpha = 20.

    lyapF.P = myMath.normalizeEllip(lyapF.P)*5.
    
    xInit = plot.getV(lyapF.P, n=30, alpha=lyapF.alpha, endPoint=False)+refTraj.getX(0.)#regular points on surface
    
    ff,aa = plt.subplots(1,2, figsize=[1+2*4,4])
    
    plot.plotEllipse(aa[0], refTraj.getX(0.), lyapF.P, 1., faceAlpha=0.)
    
    #Get the integration function
    #def __call__(self, x:np.ndarray, u_:Union[np.ndarray,Callable], t:float=0., restrictInput:bool=True, mode:List[int]=[0,0], x0:np.ndarray=False):
    fInt = lambda t, x: pendSys(x, u=lambda x,t: ndot(-K, x-refTraj.getX(t)), t=t, x0=refTraj.getX(t))
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
    #probCVX.addCstr(cstrRelaxEllipInner)
    
    #Get the objective
    #Solve for optimal input in the plus zone (negative input)
    probCVX.objective = -(objectArrayStar[0,:]+(getUlims()[0])*objectArrayStar[1,:]) #Polynomial approximation of the convergence. The more negative
    # the higher
    # the convergence, if positive divergence -> inverse signs to get minimial convergence
    solPlusStar = probCVX.solve()
    xStarPlusOpt,_,_ = probCVX.extractOptSol(solPlusStar)
    
    # Plot
    ff,aa = plt.subplots(1,1, figsize=(1+1*4,4))
    plot.plotEllipse(aa, refTraj.getX(0.), lyapF.P, lyapF.alpha, faceAlpha=0.)
    #plot.plotEllipse(aa, refTraj.getX(0.), lyapF.P, excludeInner*lyapF.alpha, faceAlpha=0.)
    aa.autoscale()
    xx,yy = plot.ax2Grid(aa, Ngrid)
    XX = np.vstack((xx.flatten(), yy.flatten()))
    X0 = refTraj.getX(0.)
    DXX = XX-X0
    #Compute separation
    cstrVal = cstrRelaxSep.poly.eval2(DXX)
    aa.contour(xx, yy, cstrVal.reshape((Ngrid, Ngrid)), [0.])

    # Compute convergence for upper half
    VDXX = lyapF.evalV(DXX)
    dVDXX = probCVX.objective.eval2(DXX)
    aa.plot(xStarPlusOpt[0,:]+X0[0,0], xStarPlusOpt[1,:]+X0[1,0], 'sb')

    #Do the same for the other half
    cstrRelaxSep.poly*=-1. #Update seperating constraint
    probCVX.objective = -(objectArrayStar[0, :] + (getUlims()[1]) * objectArrayStar[1, :])  # Update convergence / objective
    solMinusStar = probCVX.solve()
    xStarMinusOpt, _, _ = probCVX.extractOptSol(solMinusStar)

    dVDXXl = probCVX.objective.eval2(DXX)
    idxL = cstrRelaxSep.poly.eval2(DXX)>0.
    dVDXX[idxL] = dVDXXl[idxL]
    #dVDXX /= (VDXX+0.05)
    dVDXX = np.sign(dVDXX)*np.abs(dVDXX)**0.5#np.log(dVDXX - np.min(dVDXX)+0.01)
    dVDXX[np.isnan(dVDXX)] = 0.

    aa.plot(xStarMinusOpt[0, :]+X0[0,0], xStarMinusOpt[1, :]+X0[1,0], 'sr')
    CS = aa.contour(xx,yy,dVDXX.reshape((Ngrid,Ngrid)), cmap='jet')
    plt.colorbar(CS)

    
    ##
    #For the linear control input
    ff, aa = plt.subplots(1, 1, figsize=(1 + 1 * 4, 4))
    #Apply everywhere
    probCVXLin = relax.convexProg(thisRepr)
    probCVXLin.addCstr(baseRelax)
    probCVXLin.addCstr(cstrRelaxEllip)

    probCVXLin.objective = -(objectArrayLin[0,:]+objectArrayLin[1,:])
    solLin = probCVXLin.solve()
    xLinOpt, optimalCstrLinOpt, (varMonomBaseLinOpt, ULinOpt, relTolLinOpt) = probCVXLin.extractOptSol(solLin)
    
    # Get the convergence values
    dVDXX = probCVXLin.objective.eval2(DXX).squeeze()
    dVDXX = np.sign(dVDXX) * np.abs(dVDXX) ** 0.5

    plot.plotEllipse(aa, X0, lyapF.P, lyapF.alpha, faceAlpha=0.)
    CS = aa.contour(xx,yy,dVDXX.reshape((Ngrid,Ngrid)), cmap='jet')
    aa.plot(xLinOpt[0,:]+X0[0,0], xLinOpt[1,:]+X0[1,0], '*r')
    plt.colorbar(CS)

    ## Now project all onto unit circle
    # Do only for linear
    Pg = lyapF.P/lyapF.alpha
    Cg = cholesky(Pg, lower=False, check_finite=False)
    Cgi = inv(Cg, check_finite=False)

    #One compare
    xEllip = plot.getV(Pg, endPoint=False)
    yCirc = ndot(Cg, xEllip)

    xVals = probCVXLin.objective.eval2(xEllip)

    probCVXLin.objective.coeffs = probCVXLin.repr.doLinCoordChange(probCVXLin.objective.coeffs, Cgi)
    yVals = probCVXLin.objective.eval2(yCirc)

    # Change coords for rest
    for aCstr in probCVXLin.constraints.s.cstrList:
        if isinstance(aCstr, relax.lasserreConstraint):
            aCstr.poly.coeffs = aCstr.repr.doLinCoordChange(aCstr.poly.coeffs, Cgi)
            aCstr.poly *= (1./nmax(nabs(aCstr.poly.coeffs)))

    solLinY = probCVXLin.solve()
    xLinOptY, optimalCstrLinOptY, (varMonomBaseLinOptY, ULinOptY, relTolLinOptY) = probCVXLin.extractOptSol(solLinY)

    ff,aa = plt.subplots(1,1, figsize=(1+1*4,4))
    plot.plotEllipse(aa, refTraj.getX(0.), np.identity(2), 1, faceAlpha=0.)
    #plot.plotEllipse(aa, refTraj.getX(0.), lyapF.P, excludeInner*lyapF.alpha, faceAlpha=0.)
    aa.autoscale()
    xx,yy = plot.ax2Grid(aa, Ngrid)
    XX = np.vstack((xx.flatten(), yy.flatten()))
    X0 = refTraj.getX(0.)
    DXX = XX-X0

    dVDXX = probCVXLin.objective.eval2(DXX).squeeze()
    dVDXX = np.sign(dVDXX) * np.abs(dVDXX) ** 0.5

    CS = aa.contour(xx, yy, dVDXX.reshape((Ngrid, Ngrid)), cmap='jet')
    aa.plot(xLinOptY[0, :] + X0[0, 0], xLinOptY[1, :] + X0[1, 0], '*r')
    plt.colorbar(CS)

    #Isolate one of them by hand
    cstrPolyEllipExcl = poly.polynomial(thisRepr)
    Qexcl = np.identity(2)
    xExcl = xLinOptY[:,[0]]
    cstrPolyEllipExcl.setQuadraticForm(Qexcl, thisRepr.varNumsPerDeg[1],
                                       np.hstack([-.01 + float(mndot([xExcl.T, Qexcl, xExcl])), -2.*ndot(xExcl.T, Qexcl).squeeze()]).astype(nfloat),
                                       thisRepr.varNumsUpToDeg[1])

    cstrPolyEllipExcl *= (1./nmax(nabs(cstrPolyEllipExcl.coeffs)))

    exclCstrVal = cstrPolyEllipExcl.eval2(DXX)
    aa.contour(xx,yy,exclCstrVal.reshape((Ngrid,Ngrid)), [0.], cmap='jet')

    cstrRelaxEllipExcl = relax.lasserreConstraint(baseRelax, cstrPolyEllipExcl)  # Here the 'plus' space is singled out -> negative input
    probCVXLin.addCstr(cstrRelaxEllipExcl)
    solLinYExcl = probCVXLin.solve()
    xLinOptYExcl, optimalCstrLinOptYExcl, (varMonomBaseLinOptYExcl, ULinOptYExcl, relTolLinOptYExcl) = probCVXLin.extractOptSol(solLinYExcl)
    aa.plot(xLinOptYExcl[0, :] + X0[0, 0], xLinOptYExcl[1, :] + X0[1, 0], 'og')

    plot.plt.show()




    
    
    
    
    
    
    
    
    
    
    
    
    
    
