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
#from systems.polySysTest import getSys, getUlims
import plotting as plot
from plotting import plt

from scipy.integrate import solve_ivp

from funnels import *

if __name__ == "__main__":
    # from parallelChecker import probSetter, solGetter, probQueues, solQueues
    # Complicate the problem
    thisRepr = poly.polynomialRepr(2, 4)
    
    # Get the dynamical system
    pendSys = getSys(thisRepr)  # "~/tmp/pendulumDict.pickle")
    
    # Get the trajectory
    xTraj = lambda t: narray([[np.pi], [0.]], dtype=nfloat)
    dxTraj = lambda t: narray([[0.], [0.]], dtype=nfloat)
    xTraj = lambda t:narray([[np.pi*t], [np.pi]], dtype=nfloat)
    dxTraj = lambda t:narray([[np.pi], [0.]], dtype=nfloat)
    # Compute necessary input (here 0.)
    # uRefTmp = pendSys.getUopt(xTraj(0), dxTraj(0), respectCstr=False, fullDeriv=True)
    uRefTmp = lambda t:pendSys.getUopt(xTraj(t), dxTraj(t), respectCstr=False, fullDeriv=True)
    #uRefTmp = lambda t:pendSys.getUopt(xTraj(t), dxTraj(t), respectCstr=False)
    # uTraj = lambda t: uRefTmp.copy()
    # uTraj=uRefTmp
    # def __int__(self, fX: Callable, fU: Callable, nx: int, nu: int, fXd: Callable = None, tMin: float = 0., tMax: float = 1.):
    refTraj = traj.analyticTrajectory(xTraj, uRefTmp, 2, 1, dxTraj)
    
    # Get the input constraints along the refTraj
    pendSys.ctrlInput = dynSys.constraints.boxInputCstrLFBG(thisRepr, refTraj, 1, *getUlims())
    
    lyapF = lyap.quadraticLyapunovFunctionTimed(pendSys)
    lyapF.opts_['zoneCompLvl'] = 1
    
    # Set the interpolator
    # lyapF.interpolate = lyap.standardInterpolNoDeriv
    lyapF.interpolate = lyap.standardInterpol
    
    # evolving the Lyapunov function along the trajectory
    # thisLyapEvol = lyap.noChangeLyap()
    thisLyapEvol = lyap.quadLyapTimeVaryingLQR(dynSys=pendSys, refTraj=refTraj)
    myFunnel = distributedFunnel(pendSys, lyapF, refTraj, thisLyapEvol, propagator=relax.propagators.dummyPropagator(), opts={'minConvRate':-0.})
    
    Pinit = lyapF.lqrP(0.01*np.identity(2), np.identity(1), refTraj.getX(0.))[0]
    # P=np.array([[1.,0.],[0.,1.]])
    
    # myFunnel.compute(0.0, 0.01, (lyapF.P, 1.))
    # Instead of computing an actually  stabilizable funnel, simply impose the shape to show the streamlines
    for at in np.linspace(0., 0.1, 5):
        lyapF.register(at, (Pinit, .25))
    opts_ = {'pltStyle':'proj', 'linewidth':1., 'color':[0.0, 0.0, 1.0, 1.0],
             'faceAlpha':0.0, 'linestyle':'-',
             'plotAx':np.array([0, 1]),
             'cmap':'viridis', 'colorStreams':'ang', 'nGrid':200, 'cbar':True,
             'modeDyn':[0, 0]}
    
    myFunnel.distributor.terminate()
    
    print(f"final funnel is \n P: \n {myFunnel.lyapFunc.getPnPdot(0., True)[0]} \n P: \n {myFunnel.lyapFunc.getPnPdot(0., True)[1]}")
    # plot.plt.show()
    Uminmax = getUlims()
    quiverScale = 0.02
    for at in np.linspace(0., 0.1, 5):
        # Compute the proof for each time point
        allTaylorApprox = [pendSys.getTaylorApprox(refTraj.getX(at))]
        doesConverge, results, resultsLin, timePoints = myFunnel.verify1(narray([at]), None, allTaylorApprox=allTaylorApprox)
        pltDict = plot.plot2dConv(myFunnel, at)
        for aSubProof in results:
            for aSubProofList in aSubProof:
                for aProof in aSubProofList:
                    yCritRel = lyapF.sphere2Ellip(at, aProof['xSol'])  # These are relative coords -> add current pos
                    yCritAbs = yCritRel + refTraj.getX(at)
                    #Get extremal derivatives
                    ydMin = pendSys(yCritAbs, np.tile(narray([Uminmax[0]]),(1,yCritAbs.shape[1])), 0, mode=[3, 3], x0=refTraj.getX(0.))
                    ydMax = pendSys(yCritAbs, np.tile(narray([Uminmax[1]]),(1,yCritAbs.shape[1])), 0, mode=[3, 3], x0=refTraj.getX(0.))

                    ydMinOrig = pendSys(yCritAbs, np.tile(narray([Uminmax[0]]), (1, yCritAbs.shape[1])), 0, mode=[0, 0], x0=refTraj.getX(0.))
                    ydMaxOrig = pendSys(yCritAbs, np.tile(narray([Uminmax[1]]), (1, yCritAbs.shape[1])), 0, mode=[0, 0], x0=refTraj.getX(0.))

                    # Get relative derivatives
                    ydMinRel = ydMin-refTraj.getDX(at)
                    ydMaxRel = ydMax-refTraj.getDX(at)
                    ydMinRelOrig = ydMinOrig-refTraj.getDX(at)
                    ydMaxRelOrig = ydMaxOrig-refTraj.getDX(at)
                    # Get the velocities using taylor approx
                    # Compute Taylor approx around reference point
                    allTaylorApprox = [pendSys.getTaylorApprox(refTraj.getX(at)) for i in range(yCritAbs.shape[1])]
                    # Get the vector of monomials from relative position
                    zCritRel = thisRepr.evalAllMonoms(yCritRel, maxDeg=pendSys.maxTaylorDeg)
                    ydSysDyn = np.hstack( [ndot(allTaylorApprox[i][0], zCritRel[:,[i]]) for i in range(yCritAbs.shape[1])] )
                    gTaylorContracted = [neinsum('kij,k->ij', allTaylorApprox[i][1], zCritRel[:,i]) for i in range(yCritRel.shape[1])]
                    # Rate them with respect to the inputs
                    ydInputMinDyn = np.hstack( [aG*Uminmax[0] for aG in gTaylorContracted] )
                    ydInputMaxDyn = np.hstack( [aG*Uminmax[1] for aG in gTaylorContracted] )
                    ydMinTaylor = ydSysDyn+ydInputMinDyn
                    ydMaxTaylor = ydSysDyn+ydInputMaxDyn
                    ydMinRelTaylor = ydMinTaylor - refTraj.getDX(at)
                    ydMaxRelTaylor = ydMaxTaylor - refTraj.getDX(at)
                    # Get the corresponding derivative of the Lyapunov function
                    VdMin = lyapF.evalVd(yCritRel, ydMinRel, at) #Use relative vel
                    VdMax = lyapF.evalVd(yCritRel, ydMaxRel, at)
                    VdMinTaylor = lyapF.evalVd(yCritRel, ydMinRelTaylor, at)  # Use relative vel
                    VdMaxTaylor = lyapF.evalVd(yCritRel, ydMaxRelTaylor, at)
                    pltDict['ax'].plot(yCritAbs[0, :], yCritAbs[1, :], 'dr')
                    plot.myQuiver(pltDict['ax'], yCritAbs, ydMinRel*quiverScale, c='b')
                    plot.myQuiver(pltDict['ax'], yCritAbs, ydMaxRel*quiverScale, c='r')

                    xxx, yyy, YYY = plot.ax2Grid(pltDict['ax'], 20, returnFlattened=True)
                    ydAllMinRel = pendSys(YYY, np.tile(narray([Uminmax[0]]), (1, YYY.shape[1])), 0, mode=[3, 3], x0=refTraj.getX(0.), dx0=refTraj.getDX(at))
                    ydAllMaxRel = pendSys(YYY, np.tile(narray([Uminmax[1]]), (1, YYY.shape[1])), 0, mode=[3, 3], x0=refTraj.getX(0.), dx0=refTraj.getDX(at))
                    ydAllZeroRel = pendSys(YYY, np.tile(narray([[0.]]), (1, YYY.shape[1])), 0, mode=[3, 3], x0=refTraj.getX(0.), dx0=refTraj.getDX(at))
                    
                    plot.myQuiver(pltDict['ax'], YYY, ydAllMinRel*quiverScale, c='b')
                    plot.myQuiver(pltDict['ax'], YYY, ydAllMaxRel*quiverScale, c='r')
                    plot.myQuiver(pltDict['ax'], YYY, ydAllZeroRel*quiverScale, c='g')
                    
                    # Check consistency of derivative values
                    # Using the lyap
                    VdMinAll = lyapF.evalVd(YYY-refTraj.getX(at), ydAllMinRel, at, kd=False) #Use relative vel
                    VdMaxAll = lyapF.evalVd(YYY-refTraj.getX(at), ydAllMaxRel, at, kd=False) #Use relative vel
                    VdZeroAll = lyapF.evalVd(YYY-refTraj.getX(at), ydAllZeroRel, at, kd=False) #Use relative vel
                    # Using the control dict
                    controlDictTest = lyapF.getCtrlDict(at, returnZone=False)
                    # Compute using control dict
                    thisPoly = poly.polynomial(myFunnel.repr)
                    thisPoly.coeffs = controlDictTest[-1][0] + controlDictTest[0][-1].copy()
                    VdMinCtrl = thisPoly.eval2(yCritRel)
                    VdMinAllCtrl = thisPoly.eval2(YYY-refTraj.getX(at))
                    thisPoly.coeffs = controlDictTest[-1][0].copy()+controlDictTest[0][1].copy()
                    VdMaxCtrl = thisPoly.eval2(yCritRel)
                    VdMaxAllCtrl = thisPoly.eval2(YYY-refTraj.getX(at))
                    thisPoly.coeffs = controlDictTest[-1][0].copy()
                    VdZeroAllCtrl = thisPoly.eval2(YYY-refTraj.getX(at))
                    
                    VdOptAll = np.minimum(VdMinAll,VdMaxAll).squeeze()
                    idxVdOptAllOk = VdOptAll<=0.
                    pltDict['ax'].plot(YYY[0,idxVdOptAllOk], YYY[1,idxVdOptAllOk], '.g')
                    
                    # Assert
                    if not np.allclose(VdMin, VdMinCtrl):
                        print("nope")
                    if not np.allclose(VdMax, VdMaxCtrl):
                        print("nope")
                    if not np.allclose(VdMinAll.squeeze(), VdMinAllCtrl.squeeze()):
                        print('nope')
                    if not np.allclose(VdMaxAll.squeeze(), VdMaxAllCtrl.squeeze()):
                        print('nope')
                    if not np.allclose(VdZeroAll.squeeze(), VdZeroAllCtrl.squeeze()):
                        print('nope')

                    

                    
                    

                    
                    
                    
        # plot.plot2dProof(myFunnel, at)
        # plot.plot2DCONV_and_2Dproof(plot.plot2dConv(myFunnel, 0.01),plot.plot2dProof(myFunnel, 0.01))
    #   myFunnel.compute(0.0, 0.5, (lyapF.P, 100.))
    #
    #     # Disable plot for timing
    #
    # # opts_ = {'pltStyle':'proj', 'linewidth':1., 'color':[0.0, 0.0, 1.0, 1.0],
    # #         'faceAlpha':0.0, 'linestyle':'-',
    # #         'plotAx':np.array([0, 1]),
    # #         'cmap':'viridis', 'colorStreams':'ang', 'nGrid':200, 'cbar':True,
    # #         'modeDyn':[0,0]}
    #   plot.plot2dConv(myFunnel, 0.0)
    #   plot.plot2dProof(myFunnel, 0.0)
    #
    # plot.plot2dConv(myFunnel, 0.05)
    # plot.plot2dProof(myFunnel, 0.05)
    
    # distributor.terminate()
    #
    # print(f"final funnel is \n P: \n {myFunnel.lyapFunc.getPnPdot(0., True)[0]} \n P: \n {myFunnel.lyapFunc.getPnPdot(0., True)[1]}")
    plot.plt.show()

