from coreUtils import *

from systems import pendulum
import trajectories as traj
import polynomial as poly

from scipy.integrate import solve_ivp, quad

import plotting as plot

if __name__ == "__main__":
    repr = poly.polynomialRepr(2,4)
    pendSys = pendulum.getSys(repr)

    trajOMPL = traj.omplTrajectory(pendSys, traj.decomposeOMPLFile("./data/ompl_traj/invPendSwingUp", pendSys.nq, pendSys.nu), pendSys.nq,
                                   pendSys.nu)

    # Integrate and see
    method='LSODA' #RK45 got some issues here, i don't know why
    tEval = np.linspace(trajOMPL.t[0], trajOMPL.t[-1], 20*trajOMPL.t.size)
    fI = lambda t,x: pendSys(x.reshape((pendSys.nq,1)), trajOMPL.getU(t), mode=[0,0], restrictInput=False)
    sol_ode = solve_ivp(fI, [trajOMPL.t[0],trajOMPL.t[-1]], nzeros((2,)), vectorized=True, t_eval=tEval, method=method)

    # Some "small" feedback
    K = np.array([[.025, 0.]])
    getU = lambda t,x : trajOMPL.getU(t)-ndot(K,x-trajOMPL.getX(t))
    fI = lambda t,x: pendSys(x.reshape((pendSys.nq,1)), getU(t,x), mode=[0,0], restrictInput=False)
    sol_ode_feedback = solve_ivp(fI, [trajOMPL.t[0], trajOMPL.t[-1]], nzeros((2,)), vectorized=True, t_eval=tEval, method=method)

    Ufeed = getU(sol_ode_feedback.t, sol_ode_feedback.y)

    XdSys = pendSys(trajOMPL.getX(tEval), trajOMPL.getU(tEval), restrictInput=False, mode=[0,0])

    ff,aa = plot.plt.subplots(3,1)
    tI = np.linspace(trajOMPL.t[0], trajOMPL.t[-1], 10*trajOMPL.t.size)

    for i in range(2):
        aa[i].plot(sol_ode.t, sol_ode.y[i,:], 'k')
        aa[i].plot(sol_ode.t, sol_ode_feedback.y[i,:], 'b')
        aa[i].plot(tEval, trajOMPL.getX(tEval)[i, :], '--g')
        aa[i].plot(trajOMPL.t, trajOMPL.X[i, :], '.g')

    aa[1].plot(tEval, XdSys[0,:], 'r')


    aa[2].plot(tI, trajOMPL.getU(tI)[0,:], 'k')
    aa[2].plot(sol_ode_feedback.t, Ufeed[0,:], 'b')
    aa[2].plot(tEval, trajOMPL.getU(tEval, doRestrict=False)[0,:], '--g')
    aa[2].plot(trajOMPL.t, trajOMPL.U[0,:], '.g')

    plot.plt.show()