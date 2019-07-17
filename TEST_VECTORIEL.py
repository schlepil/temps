import numpy as np
from scipy.sparse import coo_matrix
import polynomial as poly
import plotting as plot
from scipy.integrate import solve_ivp
from plotting import plt
import relaxations as relax
from relaxations.lasserre import *
from relaxations.constraints import *
from relaxations.optUtils import *
from relaxations.rref import robustRREF
import dynamicalSystems as dynSys
import trajectories as traj
import Lyapunov as lyap
from systems.pendulum import getSys, getUlims
myrepr = poly.polynomialRepr(2,6)
pendSys = getSys(myrepr, fileName=None)  # "~/tmp/pendulumDict.pickle")

# Get the trajectory
xTraj = lambda t: narray([[np.pi], [0.]], dtype=nfloat)

dxTraj = lambda t: narray([[0.], [0.]], dtype=nfloat)

# Compute necessary input (here 0.)
uRefTmp = pendSys.getUopt(xTraj(0,), ddx=dxTraj(0.), respectCstr=False, fullDeriv=True)
#print(uRefTmp)
uTraj = lambda t: uRefTmp.copy()
# Creation d'un polynome a partir de la represetation
#mypoly = poly.polynomial(myrepr,coeff)
refTraj = traj.analyticTrajectory(fX=xTraj, fU=uTraj, nx=2, nu=1, fXd=dxTraj)
print("refTrajjjj",refTraj(0.))
pendSys.ctrlInput = dynSys.constraints.boxInputCstrLFBG(myrepr, refTraj, 1, *getUlims()) #Ulime = [-2,2]
lyapF = lyap.quadraticLyapunovFunction(pendSys)
lyapF.P, K = lyapF.lqrP(Q=np.identity(2), R=np.identity(1), x=refTraj.getX(0.))
print('getX(0.)',refTraj.getX(0.))
print("this is P",lyapF.P)
print("")
print('this is K',K)
 #Return:   K: 2-d array
 #      State feedback gains
 #          S: 2-d array
 #      Solution to Riccati equation #          E: 1-d array
 #      Eigenvalues of the closed loop system
lyapF.alpha = 20.
lyapF.P = myMath.normalizeEllip(lyapF.P) * 5.
print("P nouvelle",lyapF.P)
xInit = plot.getV(lyapF.P, n=30, alpha=lyapF.alpha, endPoint=False) + refTraj.getX(0.)  # regular points on surface
print('xInt',xInit)
ff, aa = plt.subplots(1, 2, figsize=[1 + 2 * 4, 4])

plot.plotEllipse(aa[0], refTraj.getX(0.), lyapF.P, 1., faceAlpha=0.)
# Get the integration function
# def __call__(self, x:np.ndarray, u_:Union[np.ndarray,Callable], t:float=0., restrictInput:bool=True, mode:List[int]=[0,0], x0:np.ndarray=False):
fInt = lambda t, x: pendSys(x, u=-ndot(K,x),t=t, x0=refTraj.getX(0.))
#print('fInt', fInt(0.,xTraj(0.)))
maxStep = 0.1
print('penduleC',lyapF.C_)

# Define an end
fTerminalConv = lambda t, x: float(lyapF.evalV(x.reshape((lyapF.nq, -1)) - refTraj.getX(t), kd=False)) - 1e-5
print('fter,inalconv',fTerminalConv(100.,xTraj(100.)))
fTerminalConv.terminal = True
fTerminalDiverge = lambda t, x: float(lyapF.evalV(x.reshape((lyapF.nq, -1)) - refTraj.getX(t), kd=False)) - 20.
fTerminalDiverge.terminal = True

for k in range(xInit.shape[1]):
    sol = solve_ivp(fInt, [0., 1.], xInit[:, k], events=[fTerminalConv, fTerminalDiverge], vectorized=True, max_step=maxStep)

    aa[0].plot(sol.y[0, :], sol.y[1, :], 'k')
    aa[1].semilogy(sol.t, lyapF.evalV(sol.y - refTraj.getX(sol.t), kd=False), 'k')

#Plot the constraint
ff, aa = plt.subplots(1, 2, figsize=[1 + 2 * 4, 4])
plot.plotEllipse(aa[0], refTraj.getX(0.), lyapF.P, lyapF.alpha, faceAlpha=0.)
    # First get the taylor
fTaylorX0, gTaylorX0 = pendSys.getTaylorApprox(refTraj.getX(0.), 3)

    #Constraint
cstrPolySep = poly.polynomial(myrepr)
cstrPolySep.coeffs = lyapF.getCstrWithDeg(gTaylorX0, 3, 3)
# Objective
uCtrlStar = np.array([[1.]])  # Normalized input
uMonomStar = myrepr.varNumsUpToDeg[0]
objectArrayStar = lyapF.getObjectiveAsArray(fTaylorX0, gTaylorX0, taylorDeg=3, u=uCtrlStar, uMonom=uMonomStar)

uCtrlLin, uMonomLin = pendSys.ctrlInput.getU(narray([2]), 0., P=lyapF.P, PG0=ndot(lyapF.P_, gTaylorX0[0, :, :]),
                                             alpha=lyapF.alpha,
                                             monomOut=True)
objectArrayLin = lyapF.getObjectiveAsArray(fTaylorX0, gTaylorX0, taylorDeg=3, u=uCtrlLin, uMonom=uMonomLin)

Ngrid = 200
plot.plotEllipse(aa[0], refTraj.getX(0.), lyapF.P, 1., faceAlpha=0.)
aa[0].autoscale()
xx, yy = plot.ax2Grid(aa[0], Ngrid)
XX = np.vstack((xx.flatten(), yy.flatten()))
valCstr = cstrPolySep.eval2(XX - refTraj.getX(0.))
aa[0].contour(xx, yy, valCstr.reshape((Ngrid, Ngrid)), [0.])

plot.plt.show()