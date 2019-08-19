if __name__ == "__main__":
  
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
  from systems.pendulum import getSys, getUlims
  
  if coreOptions.doPlot:
    import plotting as plot
    from plotting import plt
  
  from scipy.integrate import solve_ivp
  
  from funnels import *


  #from parallelChecker import probSetter, solGetter, probQueues, solQueues
  # Complicate the problem
  thisRepr = poly.polynomialRepr(2, 4)

  # Get the dynamical system
  pendSys = getSys(thisRepr, fileName=None)  # "~/tmp/pendulumDict.pickle")

  # Get the trajectory
  # Static
  #xTraj = lambda t: narray([[np.pi], [0.]], dtype=nfloat)
  #dxTraj = lambda t: narray([[0.], [0.]], dtype=nfloat)
  # Moving
  xTraj = lambda t: narray([[np.pi*t], [np.pi*1]], dtype=nfloat)
  dxTraj = lambda t: narray([[np.pi], [0.]], dtype=nfloat)
  # Compute necessary input (here 0.)
  #uRefTmp = pendSys.getUopt(xTraj(0), dxTraj(0), respectCstr=False, fullDeriv=True)
  uRefTmp = lambda t: pendSys.getUopt(xTraj(t), dxTraj(t), respectCstr=False, fullDeriv=True)
  #uTraj = lambda t: uRefTmp.copy()
  #uTraj=uRefTmp
  # def __int__(self, fX: Callable, fU: Callable, nx: int, nu: int, fXd: Callable = None, tMin: float = 0., tMax: float = 1.):
  refTraj = traj.analyticTrajectory(xTraj,uRefTmp, 2, 1, dxTraj)

  # Get the input constraints along the refTraj
  pendSys.ctrlInput = dynSys.constraints.boxInputCstrLFBG(thisRepr, refTraj, 1, *getUlims())

  lyapF = lyap.quadraticLyapunovFunctionTimed(pendSys)

  # Set the interpolator
  # lyapF.interpolate = lyap.standardInterpolNoDeriv
  lyapF.interpolate = lyap.standardInterpol

  # evolving the Lyapunov function along the trajectory
  thisLyapEvol = lyap.noChangeLyap()
  
  # Get the propagator of critical solutions
  thisPropagator = relax.propagators.localFixedPropagator()

  myFunnel = distributedFunnel(dynSys=pendSys, lyapFunc=lyapF, traj=refTraj, evolveLyap=thisLyapEvol, propagator=thisPropagator, opts={'minConvRate':-0.})

  Pinit = lyapF.lqrP(np.identity(2), np.identity(1), refTraj.getX(0.))[0]
  #Scale
  Pinit *= 3./min(eigh(Pinit)[0])**0.5

  #P=np.array([[1.,0.],[0.,1.]])
  
  myFunnel.compute(0.0, 0.001, (Pinit, 1.))

  if coreOptions.doPlot:
    opts_ = {'pltStyle': 'proj', 'linewidth': 1., 'color': [0.0, 0.0, 1.0, 1.0],
             'faceAlpha': 0.0, 'linestyle': '-',
             'plotAx': np.array([0, 1]),
             'cmap': 'viridis', 'colorStreams': 'ang', 'nGrid': 200, 'cbar': True,
             'modeCtrl': (2, nones((myFunnel.dynSys.nu,), dtype=nint)),
             'modeDyn': [3, 3]}
    plot.plot2dConv(myFunnel, 0.0)
    plot.plot2dConv(myFunnel, 0.0,opts=opts_)
    # plot.plot2dProof(myFunnel, 0.0)
    #
    # plot.plot2dConv(myFunnel, 0.05)
    # plot.plot2dProof(myFunnel, 0.05)

    myFunnel.distributor.terminate()
  
    print(f"final funnel is \n P: \n {myFunnel.lyapFunc.getPnPdot(0., True)[0]} \n P: \n {myFunnel.lyapFunc.getPnPdot(0., True)[1]}")
    plot.plt.show()