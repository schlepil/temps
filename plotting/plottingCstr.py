from coreUtils import *
import plotting.plottingTools as pt

import relaxations as rel

def plot2dCstr(aObj:Union[rel.constraint, rel.polynomial, rel.convexProg], ax:pt.matplotlib.axes, opts={}, fig=None):
    
    opts_ = {'binaryPlot':True, 'filled':False, 'nGrid':200, 'cbar':True, 'ctrOpts':{}}
    opts_.update(opts)
    
    if isinstance(aObj, rel.polynomial):
        bRel = rel.lasserreRelax(aObj.repr)
        aObj = rel.lasserreConstraint(bRel, aObj)
    
    try:
        opts_['nGrid'][0]
    except TypeError:
        opts_['nGrid'] = (opts_['nGrid'], opts_['nGrid'])
    
    #Get points
    xg,yg = pt.ax2Grid(ax, opts_['nGrid'])
    XX = np.vstack([xg.flatten(),yg.flatten()])
    ZZ = aObj.poly.repr.evalAllMonoms(XX)

    if isinstance(aObj, rel.convexProg):
        YY = sum([aCstr.poly.eval2(ZZ) for aCstr in aObj.constraints.l+aObj.constraints.q+aObj.constraints.s])
    else:
        YY = aObj.poly.eval2(ZZ)
    
    if opts_['binaryPlot']:
        YY = (YY>=0.).astype(nfloat)
        opts_.update( {'levels':[0.,1.]} )
    
    if opts_['filled']:
        cc = ax.contourf(xg, yg, YY.reshape(xg.shape), **opts_['ctrOpts'])
    else:
        cc = ax.contour(xg, yg, YY.reshape(xg.shape), **opts_['ctrOpts'])
    
    if opts_['cbar']:
        if fig is None:
            print("Cannot print colorbar without figure")
        else:
            fig.colorbar(cc)

    return cc
