from coreUtils import *

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


def plot2dConv(funnel:fn.distributedFunnel, t=0.0, opts={}):
    opts_ = {'plotStyle':'proj', 'linewidth':1., 'color':[0.0, 0.0, 1.0, 1.0],
             'faceAlpha':0.0, 'linestyle':'-',
             'plotAx':np.array([0, 1]),
             'cmap':'viridis', 'colorStreams':'conv', 'nPt':200,
             'modeCtrl':(1,nones((funnel.dynSys.nu,), dtype=nint)),
             'modeDyn':[3,3]}
    opts_.update(opts)
    
    #assert opts_['colorStreams'] in ('conv', 'mag', 'dir')
    assert (opts_['ctrlMode'] is None) or ((opts_['ctrlMode'] is np.ndarray) and (opts_['ctrlMode'].size == funnel.dynSys.nu))
    
    ff,aa = plt.subplots(1,1)
    
    # Plot the region
    funnel.lyapFunc.plot(aa, t, opts_=opts_)
    aa.autoscale()
    
    # Plot the streamlines
    
    # Get the veloctiy
    xx, yy = ax2Grid(aa, opts_['nPt'])
    XX = np.vstack((xx.flatten(), yy.flatten()))

    x0 = funnel.dynSys.ctrlInput.refTraj.getX(t)
    xd0 = funnel.dynSys.ctrlInput.refTraj.getDX(t)
    
    UU = funnel.lyapFunc.getCtrl(t, opts_['modeCtrl'], XX, x0)
    
    # __call__(self, x:np.ndarray, u:np.ndarray, mode:str='OO', x0:np.ndarray=None):
    VV = funnel.dynSys(XX, UU, opts_['modeDyn'], x0, xd0)
    
    # Compute streamline color
    try:
        streamColor = plt.color.to_rgb(opts_['colorStreams'])
    except:
        if opts_['colorStreams'] == 'conv':
            streamColor = funnel.lyapFunc.evalVd(XX, VV, t)
        
        elif opts_['colorStreams'] == 'mag':
        elif opts_['colorStreams'] == 'dir':
        else:
            raise NotImplementedError
    
    aa.stream



    
    