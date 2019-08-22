from coreUtils import *

from systems import pendulum
import trajectories as traj
import polynomial as poly

from scipy.integrate import solve_ivp

if __name__ == "__main__":
    repr = poly.polynomialRepr(2,4)
    pendSys = pendulum.getSys(repr)

    trajOMPL = traj.omplTrajectory(pendSys, traj.decomposeOMPLFile("./data/ompl_traj/invPendSwingUp", pendSys.nq, pendSys.nu), pendSys.nq,
                                   pendSys.nu)

    # Integrate and see
    fI = lambda t,x: pendSys(x.reshape((pendSys.nq,1)), trajOMPL.getU(t), mode=[0,0], restrictInput=False)
    solve_ivp(fI, [trajOMPL.t[0],trajOMPL.t[-1]], nzeros((2,)), vectorized=True, t_eval=trajOMPL.t)

