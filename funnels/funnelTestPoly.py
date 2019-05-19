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

import plotting as plot
from plotting import plt

from scipy.integrate import solve_ivp

from funnels import *

#from parallelChecker import probSetter, solGetter, probQueues, solQueues

def doTesting(funnel:distributedFunnel):
    import plotting as plot
    import relaxations as relax
    
    lyapFunc_ = funnel.lyapFunc
    dynSys_ = funnel.dynSys
    repr_ = funnel.dynSys.repr
    
    relLasserre = relax.lasserreRelax(repr_)
    thisPoly0 = polynomial(repr_)
    thisPoly1 = polynomial(repr_)
    thisPolyCstr = polynomial(repr_)
    
    # Fill up the Lyapunov function
    P0 = np.array([[2.1, 0.4], [0.4, 0.95]], dtype=nfloat)
    P1 = np.array([[1.5, -0.4], [-0.4, 1.25]], dtype=nfloat)

    lyapFunc_.reset()
    lyapFunc_.register(0., (P0,1.05))
    lyapFunc_.register(0.5, (P1,0.95))
    
    t = 0.133
    # Get the zone
    zone = lyapFunc_.getZone(0.133)
    
    x0,dx0,u0 = funnel.dynSys.ctrlInput.refTraj(t)
    
    # Taylor
    fTaylor, gTaylor = dynSys_.getTaylorApprox(x0)
    
    # Get the control dict
    ctrlDict, zoneP = lyapFunc_.getCtrlDict(t, fTaylor, gTaylor, True)
    # Project to unit sphere
    ctrlDictProj = dp(ctrlDict)
    funnel.doProjection(zone, [], ctrlDictProj)
    
    # Get the linear control law
    K, Kmonoms = dynSys_.ctrlInput.getU2(2*nones((dynSys_.nu,), dtype=nint), t, zone, gTaylor, x0, True, u0)
    
    # Get a grid to evaluate on
    ff,aa = plot.plt.subplots(1,2)
    lyapFunc_.plot(aa[0], t)
    aa[0].autoscale()
    aa[0].axis('equal')
    aa[1].autoscale()
    aa[1].axis('equal')
    
    xx,yy = plot.ax2Grid(aa[0],100)
    
    XX = np.vstack((xx.flatten(), yy.flatten()))
    ZZ = repr_.evalAllMonoms(XX)
    
    dXX = XX-x0
    dZX = repr_.evalAllMonoms(dXX)
    
    # Project onto sphere
    dYY = lyapFunc_.ellip2Sphere(t, dXX)
    dZY = repr_.evalAllMonoms(dYY)
    
    # Check if
    # Control input does not exceed limits
    Vxx = lyapFunc_.evalV(dXX, t, kd=False)
    idxInside = (Vxx <= zone[1]).squeeze()
    aa[0].plot(XX[0,idxInside], XX[1,idxInside], '.')
    # Compare with projection
    Vyy = norm(dYY, axis=0, keepdims=False)
    idxInsideY = Vyy <= 1.
    if not nall(idxInside == idxInsideY):
        idxFalse = idxInside != idxInsideY
        print(f"Failed on \n {dXX[:, idxFalse]} \n with \n {Vxx[:, idxFalse]} and \n {dYY[:, idxFalse]} \n with \n {Vyy[:, idxFalse]}")
    
    aa[1].plot(dYY[0, idxInside],dYY[1, idxInside],'.')
    
    # Compute control
    # First row of K is constant term
    Ulinx =ndot( K, dZX[:3,:] )
    # Check if limits are nowhere exceeded
    uMin,uMax = np.tile( dynSys_.ctrlInput.getMinU(t), (1,XX.shape[1]) ), np.tile( dynSys_.ctrlInput.getMaxU(t), (1,XX.shape[1]) )
    if not nall( np.logical_and( uMin[:, idxInside] <= Ulinx[:, idxInside], Ulinx[:, idxInside] <= uMax[:, idxInside] ) ):
        idx = np.logical_and( idxInside, np.any(np.logical_or( uMin <= Ulinx, Ulinx <= uMax ), axis=0) )
        print(f"Failed input check on :\n")
        for ii, aidx in enumerate(idx):
            if not aidx:
                continue
            print(f"X: {list(dXX[:, ii])}; V: {float(Vxx[ii])}; U: {list(Ulinx[:,ii])}")
    
    # Check consistency of convergence polynomials
    # Step compare projected and original polynomials
    for i in range(-1, dynSys_.nu):
        for type in range(3):
            try:
                thisPoly0.coeffs = ctrlDict[i][type]
                thisPoly1.coeffs = ctrlDictProj[i][type]
                if not np.allclose(thisPoly0.eval2(dZX), thisPoly1.eval2(dZY)):
                    print(f"Failed on {i}-{type}: {np.where(np.isclose(thisPoly0.eval2(dZX), thisPoly1.eval2(dZY)))}")
            except KeyError:
                print(f"Ignore keys {i}-{type}")
    
    # Now compare the control dict convergence with the actual convergence
    # Minimal input
    dxUmin = dynSys_(XX, u=uMin, restrictInput=False, mode=[3,3],x0=x0,dx0=dx0)
    # Maximal input
    dxUmax = dynSys_(XX, u=uMax, restrictInput=False, mode=[3,3],x0=x0,dx0=dx0)
    # Mixed [1,2]
    dxUmix = dynSys_(XX, u=np.vstack([uMin[0,:], Ulinx[1,:]]), restrictInput=False, mode=[3, 3], x0=x0, dx0=dx0)
    
    # Using dynSys
    convDynSysMinU = lyapFunc_.evalVd(dXX, dxUmin, t, False)
    convDynSysMaxU = lyapFunc_.evalVd(dXX, dxUmax, t, False)
    convDynSysMixU = lyapFunc_.evalVd(dXX, dxUmix, t, False)
    
    # Using ctrlDict
    convCtrlDictMinU = ctrlDict[-1][0].copy()
    convCtrlDictMinU += ctrlDict[0][-1]
    convCtrlDictMinU += ctrlDict[1][-1]
    thisPoly0.coeffs = convCtrlDictMinU
    convCtrlDictMinU = thisPoly0.eval2(dZX).reshape((-1,))
    
    convCtrlDictMaxU = ctrlDict[-1][0].copy()
    convCtrlDictMaxU += ctrlDict[0][1]
    convCtrlDictMaxU += ctrlDict[1][1]
    thisPoly0.coeffs = convCtrlDictMaxU
    convCtrlDictMaxU = thisPoly0.eval2(dZX).reshape((-1,))

    convCtrlDictMixU = ctrlDict[-1][0].copy()
    convCtrlDictMixU += ctrlDict[0][1]
    convCtrlDictMixU += ctrlDict[1][2]
    thisPoly0.coeffs = convCtrlDictMixU
    convCtrlDictMixU = thisPoly0.eval2(dZX).reshape((-1,))
    
    ff,aa = plot.plt.subplots(1,1)
    aa.plot(convDynSysMinU, 'r')
    aa.plot(convDynSysMaxU, 'b')
    aa.plot(convDynSysMixU, 'g')
    aa.plot(convCtrlDictMinU, '.-r')
    aa.plot(convCtrlDictMaxU, '--b')
    aa.plot(convCtrlDictMixU, '*-g')

    ff, aa = plot.plt.subplots(3, 1)
    aa[0].plot(convDynSysMinU, 'r')
    aa[0].plot(convCtrlDictMinU, '--b')
    aa[1].plot(convDynSysMaxU, 'r')
    aa[1].plot(convCtrlDictMaxU, '--b')
    aa[2].plot(convDynSysMixU, 'r')
    aa[2].plot(convCtrlDictMixU, '--b')
    
    
    if not np.allclose(convDynSysMinU, convCtrlDictMinU):
        print("Failed on min U")
    
    if not np.allclose(convDynSysMaxU, convCtrlDictMaxU):
        print("Failed on max U")
    
    if not np.allclose(convDynSysMixU, convCtrlDictMixU):
        print("Failed on mix U")
    
    return None
    


if __name__ == "__main__":
    
    # Complicate the problem
    cplx = 2
    shapeP = 2

    if cplx == 0:
        R = P = nidentity(2, dtype=nfloat)
        G = narray([[0.],[1.]], dtype=nfloat)
    elif cplx == 1:
        R = plot.Rot(45./180*np.pi).astype(dtype=nfloat)
        P = mndot([R.T, narray([[2,0],[0,1]], dtype=nfloat), R])
        G = narray([[0.],[1.]], dtype=nfloat)
    elif cplx == 2:
        R = plot.Rot(45. / 180 * np.pi)
        P = mndot([R.T, narray([[2, 0], [0, 1]], dtype=nfloat), R])
        G = nzeros((2,2), dtype=nfloat)
        G[0,1] = 1.
        G[1,0] = 0.6
        G[1,1] = 0.9
    else:
        raise NotImplementedError
    
    if shapeP == 0:
        Ps = nidentity(2)
        alpha = 0.8
    elif shapeP == 1:
        Ps = np.array([[1.1,0.05],[0.05, 0.95]], dtype=nfloat)
        alpha = 0.8
    elif shapeP == 2:
        Ps = np.array([[1.1,0.2],[0.2, 0.95]], dtype=nfloat)
        alpha = 0.8
    else:
        raise NotImplementedError
        


    
    # Get the polynomial representation which also decides on the maximal relaxation
    # Let use full here
    pSys = getSysStablePos(2,6,P=P,G=G)
    
    thisRepr = pSys.repr  # Todo debug digits. there is an error somewhere

    lyapF = lyap.quadraticLyapunovFunctionTimed(pSys)
    
    # Set the interpolator
    #lyapF.interpolate = lyap.standardInterpolNoDeriv
    lyapF.interpolate = lyap.standardInterpol
    
    # evolving the Lyapunov function along the trajectory
    thisLyapEvol = lyap.noChangeLyap()
    
    myFunnel = distributedFunnel(pSys, lyapF, pSys.ctrlInput.refTraj, thisLyapEvol)
    
    doTesting(myFunnel)
    
    myFunnel.compute(0.0, 0.3, (Ps, alpha))
    
    plot.plot2dConv(myFunnel, 0.0)
    
    plot.plot2dProof(myFunnel, 0.0)
    
    distributor.terminate()
    
    
    print(f"final funnel is \n P: \n {myFunnel.lyapFunc.getPnPdot(0.,True)[0]} \n P: \n {myFunnel.lyapFunc.getPnPdot(0.,True)[1]}")
    plot.plt.show()
    

