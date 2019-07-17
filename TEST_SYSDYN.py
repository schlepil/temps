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
from Lyapunov.lyapPropagators import lyapEvol
import trajectories as traj
import Lyapunov as lyap
from systems.pendulum import getSys, getUlims
from funnels.distributedFunnel import distributedFunnel as funnel
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
lyapF = lyap.quadraticLyapunovFunctionTimed(pendSys)
P, K = lyapF.lqrP(Q=np.identity(2), R=np.identity(1), x=refTraj.getX(0.))
print('getX(0.)',refTraj.getX(0.))
print("this is P",lyapF.P)
print("")
print('this is K',K)
 #Return:   K: 2-d array
 #      State feedback gains
 #          S: 2-d array
 #      Solution to Riccati equation #          E: 1-d array
 #      Eigenvalues of the closed loop system
print("P nouvelle",lyapF.P)
xInit = plot.getV(lyapF.P, n=30, alpha=lyapF.alpha, endPoint=False) + refTraj.getX(0.)  # regular points on surface
print('xInt',xInit)
ff, aa = plt.subplots(1, 2, figsize=[1 + 2 * 4, 4])

plot.plotEllipse(aa[0], refTraj.getX(0.), lyapF.P, 1., faceAlpha=0.)
lyapEvol=lyapEvol(dynSys=pendSys,refTraj=refTraj)
myFunnel=funnel(dynSys=pendSys,lyapFunc=lyapF,traj=refTraj,evolveLyap=lyapEvol,opts={})
initZone=[P,20.]
myFunnel.compute(0.,0.1,initZone)