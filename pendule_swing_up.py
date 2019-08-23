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

    import trajectories as traj

    # Costs
    Q = np.identity(2)
    R = np.identity(1)

    thisRepr = poly.polynomialRepr(2, 4)

    # Get the dynamical system
    pendSys = getSys(thisRepr, fileName=None)  # "~/tmp/pendulumDict.pickle")

    refTraj = traj.omplTrajectory(pendSys, traj.decomposeOMPLFile("./data/ompl_traj/invPendSwingUp", pendSys.nq, pendSys.nu), pendSys.nq,
                                   pendSys.nu)

    # Get the input constraints along the refTraj
    pendSys.ctrlInput = dynSys.constraints.boxInputCstrLFBG(thisRepr, refTraj, 1, -2.5, 2.5)

    lyapF = lyap.quadraticLyapunovFunctionTimed(pendSys)

    # Set the interpolator
    lyapF.interpolate = lyap.standardInterpol

    # Set how to propagate critical points
    lyapF.opts_['zoneCompLvl'] = 1

    # evolving the Lyapunov function along the trajectory
    thisLyapEvol = lyap.quadLyapTimeVaryingLQR(pendSys,refTraj,Q=Q,R=R)
    # Get the propagator of critical solutions
    thisPropagator = relax.propagators.localFixedPropagator()

    myFunnel = distributedFunnel(dynSys=pendSys, lyapFunc=lyapF, traj=refTraj, evolveLyap=thisLyapEvol, propagator=thisPropagator,
                                 opts={'minConvRate': -0., 'optsEvol': {
                                     'tDeltaMax': 0.05}, 'interSteps': 2})

    if lyapF.opts_['zoneCompLvl'] == 1:
        myFunnel.opts['useAllAlphas'] = False  # Cannot use this option without propagation

    Pinit = lyapF.lqrP(Q, R, refTraj.getX(0.))[0]
    # Scale
    Pinit *= .1 / min(eigh(Pinit)[0]) ** 0.5

    # P=np.array([[1.,0.],[0.,1.]])

    assert lyapF.opts_['zoneCompLvl'] == 1, "TBD"
    myFunnel.compute(refTraj.t[0], refTraj.t[-1], (Pinit, 1.))

    print("Done")
