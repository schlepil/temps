from coreUtils import *
import polynomial as poly
import relaxations as relax

import matplotlib
# Adjust this if necessary, Fix for pycharm
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.collections as mcoll

from mpl_toolkits.mplot3d import Axes3D
from numpy.f2py.auxfuncs import isdouble

import funnels as fn

dynStringDict = {0:{'Q':'QP-ctrl', 'C':'SMC'}, 1:{'L':'Lin-Sys_dyn', 'P':'Poly-Sys_dyn', 'O':'NL-Sys_dyn'}, 2:{'L':'Lin-Ipt_dyn', 'P':'Poly-Ipt_dyn', 'O':'NL-Ipt_dyn', 'R':'Lin-Ipt_dyn_zoned'}}

# Unit vector in "x" direction
uVec = np.array([[1.], [0.]])


# Simple 2d rotation matrix
def Rot(alpha:float):
    R = np.zeros((2, 2))
    R[0, 0] = R[1, 1] = ncos(alpha)
    R[1, 0] = nsin(alpha)
    R[0, 1] = -R[1, 0]
    return nmatrix(R)


# Get regularly distributed points on the surface of {x|x.T.P.x <= alpha} in 2d
def getV(P:np.ndarray=np.identity(2), n:int=101, alpha:float=1, endPoint=True):
    Ang = np.linspace(0, 2*np.pi, n, endpoint=endPoint)
    V = np.zeros((2, Ang.size))
    C = np.linalg.inv(np.linalg.cholesky(P/alpha)).T
    for k in range(Ang.size):
        V[:, k] = np.dot(Rot(Ang[k]), uVec).squeeze()
    return ndot(C, V)

###############################################################################
def ax2Grid(aa, N):
    try:
        return np.meshgrid(np.linspace(aa.get_xlim()[0], aa.get_xlim()[1], N[0]), np.linspace(aa.get_ylim()[0], aa.get_ylim()[1], N[1]))
    except TypeError:
        return np.meshgrid(np.linspace(aa.get_xlim()[0], aa.get_xlim()[1], N), np.linspace(aa.get_ylim()[0], aa.get_ylim()[1], N))


###############################################################################
def projectEllip(P, T):
    # Return a "P" representing the projection of a matrix onto a affine subspace
    return np.linalg.inv(ndot(T.T, np.linalg.inv(P), T))


###############################################################################
def plot(x, y=None, z=None):
    if z is None:
        ff = plt.figure()
        aa = ff.add_subplot(111)
        if y is None:
            aa.plot(x)
        else:
            aa.plot(x, y)
    else:
        pass
    
    return aa


#########################################################################
# Plot {x|x.T.P.x <= alpha} in 2d
def getEllipse(pos, P, alpha=1., deltaPos=None):
    # center and shape to matplotlib patch
    v, E = eigh(P)
    if deltaPos is None:
        deltaPos = np.zeros((pos.shape))
    # tbd do calculations for offset
    orient = np.arctan2(E[1, 0], E[0, 0])*180.0/np.pi
    return Ellipse(xy=pos, height=2.0*np.sqrt(alpha)*1.0/np.sqrt(v[1]), width=2.0*np.sqrt(alpha)*1.0/np.sqrt(v[0]), angle=orient)


#########################################################################
def plotEllipse(ax, pos, P, alpha, plotAx=np.array([0, 1]), deltaPos=None, color=[0.0, 0.0, 1.0, 1.0], faceAlpha=0.5, pltStyle="proj", linewidth=1., linestyle='-'):
    """
    Plot the ellipsoid defined as {x|x'.P.y<=alpha} PROJECTED onto the plane defined by plotAx
    :param ax:
    :param pos:
    :param P:
    :param alpha:
    :param plotAx:
    :param deltaPos:
    :param color:
    :param faceAlpha:
    :param pltStyle:
    :param linewidth:
    :param linestyle:
    :return:
    """
    color = matplotlib.colors.colorConverter.to_rgba(color)
    color = np.array(dp(color));
    color[-1] = color[-1]*faceAlpha;
    color = list(color)
    
    if pltStyle == 'proj':
        if len(plotAx) == 2:
            T = np.zeros((P.shape[0], 2))
            T[plotAx[0], 0] = 1.
            T[plotAx[1], 1] = 1.
        else:
            assert (T.shape[0] == P.shape[0] and T.shape[1] == P.shape[1]), "No valid affine 2d sub-space"
        Pt = projectEllip(P, T)
    elif pltStyle == 'inter':
        Pt = P[np.ix_(plotAx, plotAx)]
    else:
        assert 0, "No valid pltStyle for ellip given"
    
    e = getEllipse(pos[plotAx], Pt, alpha)
    e.set_linewidth(1.)
    e.set_edgecolor(color[:3]+[1.])
    ax.add_patch(e)
    e.set_facecolor(color)
    e.set_linewidth(linewidth)
    e.set_linestyle(linestyle)
    
    return e

def getStreamColor(lyapFunc,XX,VV,t,opts):
    opts_ = {'colorStreams':'ang', 'nGrid':200}
    recursiveExclusiveUpdate(opts_,opts)

    # Compute streamline color
    try:
        streamColor = matplotlib.colors.to_rgb(opts_['colorStreams'])
    except:
        if opts_['colorStreams'] == 'conv':
            streamColor = lyapFunc.evalVd(XX, VV, t, kd=False)/(lyapFunc.evalV(XX, t, kd=False)+1e-30)
        elif opts_['colorStreams'] == 'mag':
            streamColor = norm(VV, ord=2, axis=0)
        elif opts_['colorStreams'] == 'dir':
            streamColor = np.arctan2(VV[1,:], VV[0,:])
        elif opts_['colorStreams'] == 'ang':
            streamColor = lyapFunc.convAng(XX, VV, t, kd=False)*180./np.pi
        else:
            raise NotImplementedError
        streamColor = streamColor.squeeze().reshape((opts_['nGrid'],opts_['nGrid']))

    return streamColor


def plot2dConv(funnel:fn.distributedFunnel, t=0.0, opts={}):
    opts_ = {'pltStyle':'proj', 'linewidth':1., 'color':[0.0, 0.0, 1.0, 1.0],
             'faceAlpha':0.0, 'linestyle':'-',
             'plotAx':np.array([0, 1]),
             'cmap':'viridis', 'colorStreams':'ang', 'nGrid':200, 'cbar':True,
             'modeCtrl':(1,nones((funnel.dynSys.nu,), dtype=nint)),
             'modeDyn':[3,3]}
    opts_.update(opts)
    
    # TODO define some asserts
    
    ff,aa = plt.subplots(1,1)
    
    # Plot the region
    funnel.lyapFunc.plot(aa, t, opts=opts_)
    aa.autoscale()
    
    # Plot the streamlines
    
    # Get the veloctiy
    xx, yy = ax2Grid(aa, opts_['nGrid'])
    XX = np.vstack((xx.flatten(), yy.flatten()))

    x0 = funnel.dynSys.ctrlInput.refTraj.getX(t)
    dx0 = funnel.dynSys.ctrlInput.refTraj.getDX(t)
    
    UU = funnel.lyapFunc.getCtrl(t, opts_['modeCtrl'], XX, x0)
    
    # __call__(self, x:np.ndarray, u:np.ndarray, mode:str='OO', x0:np.ndarray=None):
    VV = funnel.dynSys(XX, UU, opts_['modeDyn'], x0=x0, dx0=dx0)
    
    streamColor = getStreamColor(funnel.lyapFunc, XX,VV,t,opts_)
    
    thisStream = aa.streamplot(xx,yy,VV[0,:].reshape((opts_['nGrid'],opts_['nGrid'])),VV[1,:].reshape((opts_['nGrid'],opts_['nGrid'])), color=streamColor, cmap=opts_['cmap'])
    
    if opts_['cbar']:
        thisCBar = ff.colorbar(thisStream.lines)
    else:
        thisCBar = None
    
    return {'fig':ff, 'ax':aa, 'cbar':thisCBar}
    

def plot2dProof(funnel:fn.distributedFunnel, t=0.0, opts={}):

    from plotting.plottingCstr import plot2dCstr
    
    opts_ = {'zoneOpts':{'pltStyle':'proj', 'linewidth':1., 'color':[0.0, 0.0, 1.0, 1.0],
             'faceAlpha':0.0, 'linestyle':'-',
             'plotAx':np.array([0, 1])},
             'streamOpts':{'cmap':'viridis', 'colorStreams':'ang', 'nGrid':200, 'cbar':True, 'plotValidOnly':True, 'validEps':-.1},
             'modeDyn':[3,3],
             'cstrOpts':{'binaryPlot': True, 'filled': False, 'nGrid': 200, 'cbar': False, 'ctrOpts': {}}
             }
    recursiveExclusiveUpdate(opts_, opts)
    optsStream_ = opts_['streamOpts']

    # Helper stuff
    thisPoly = poly.polynomial(funnel.repr)
    thisRelax = relax.lasserreRelax(funnel.repr)

    #Get the subproof
    try:
        subProofList = funnel.proof_[t]
    except KeyError:
        keysT = narray(list(funnel.proof_.keys()), dtype=nfloat)
        tprime = keysT[ np.argmin( np.abs(keysT-t) ) ]
        print(f"Using time point {tprime} instead of {t}")
        t = tprime
        subProofList = funnel.proof_[t]

    x0, dx0, uRef = funnel.dynSys.ctrlInput.refTraj.getX(t), funnel.dynSys.ctrlInput.refTraj.getDX(t), funnel.dynSys.ctrlInput.refTraj.getU(t)

    allDict = {}
    for nSub, (subProof,_) in enumerate(subProofList):
        allDict[nSub] = {}
        thisDict = allDict[nSub]
        
        nProbs = len(subProof['origProb'])
    
        nax = [1,1]
    
        while True:
            if nax[0]*nax[1]>=nProbs:
                break
            nax[1] += 1
            if nax[0]*nax[1]>=nProbs:
                break
            nax[0] += 1
        
        ff,aa = plt.subplots(*nax, sharex=True, sharey=True)
        aa = narray(aa,ndmin=2)
        thisDict['fig'] = ff
        thisDict['ax'] = aa


        ff.suptitle(f"Proof {nSub} at {t:.4e}")
        #Loop over the significant problems and their solutions
        for k, (aVal, aProb) in enumerate(subProof['sigProbAndVals']):
            thisDict[k] = {}
            idx,idy = divmod(k, nax[1])
            aa[idx,idy].set_title( f"{aProb['sol']['primal objective']:.2e} : {list(aProb['origProb']['probDict']['u'].reshape((-1,)))}" )

            # Plot the zone
            zonePlot = funnel.lyapFunc.plot(ax=aa[idx,idy], t=t, x0=x0, opts=opts_['zoneOpts'])

            # Get the grid
            aa[idx, idy].autoscale()
            aa[idx, idy].axis('equal')

            xx,yy = ax2Grid(aa[idx, idy], opts_['streamOpts']['nGrid'])
            XX = np.vstack([xx.flatten(), yy.flatten()])
            DXX = XX-x0

            # Plot the additional contraints and check which points satisfy them
            idxValid = nones((DXX.shape[1],)).astype(np.bool_)

            for aCstrCoeff in aProb['origProb']['cstr'][1:]:
                thisPoly.coeffs = aCstrCoeff
                thisCstr = relax.lasserreConstraint(thisRelax, thisPoly)
                # Plot
                plot2dCstr(thisCstr, aa[idx,idy], x0=x0, opts=opts_['cstrOpts'])
                # Check feasibility of points
                idxValid = np.logical_and(idxValid, thisCstr.isValid(XX, atol=opts_['streamOpts']['validEps']))

            # Get the velocities
            UU = funnel.dynSys.ctrlInput.getUVal(DXX, aProb['origProb']['probDict']['u'], t, zone=funnel.lyapFunc.getZone(t),
                                                 gTaylor=funnel.dynSys.getTaylorApprox(x0)[1], uRef=uRef)
            VV = funnel.dynSys(XX, UU, mode=opts_['modeDyn'], x0=x0,dx0=dx0)

            streamColor = getStreamColor(funnel.lyapFunc, XX, VV, t, opts_['streamOpts'])

            thisStream = aa[idx, idy].streamplot(xx, yy, VV[0, :].reshape((optsStream_['nGrid'], optsStream_['nGrid'])), VV[1, :].reshape((optsStream_['nGrid'], optsStream_['nGrid'])),
                                       color=streamColor, cmap=optsStream_['cmap'])

            if optsStream_['cbar']:
                thisCBar = ff.colorbar(thisStream.lines)
            else:
                thisCBar = None

            thisDict[k]['zonePlot'] = zonePlot
            thisDict[k]['stream'] = thisStream
            thisDict[k]['cbar'] = thisCBar

    return allDict









        
    
    





