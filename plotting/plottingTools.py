from coreUtils import *
from matplotlib import pyplot as plt
import matplotlib

def myQuiver(ax, X, V, c=None, otherPlotOptDict={}):
    dim = X.shape[0]
    assert dim in (2, 3)

    XV = X + V

    if (c in ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']) or isinstance(c, (list, tuple)):
        cl = lambda x: c
    elif c == 'arctan':
        cmap = plt.get_cmap('viridis')
        cl = lambda x: cmap((np.arctan2(x[0], x[1]) + np.pi) / (2. * np.pi))
    else:
        assert 0

    if dim == 2:
        XX = np.vstack((X[0, :], XV[0, :]))
        YY = np.vstack((X[1, :], XV[1, :]))
        for k in range(X.shape[1]):
            ax.plot(XX[:, k], YY[:, k], color=cl(V[:, k]), **otherPlotOptDict)
    else:
        XX = np.vstack((X[0, :], XV[0, :]))
        YY = np.vstack((X[1, :], XV[1, :]))
        ZZ = np.vstack((X[2, :], XV[2, :]))
        for k in range(X.shape[1]):
            ax.plot(XX[:, k], YY[:, k], ZZ[:, k], color=cl(V[:, k]), **otherPlotOptDict)

    return 0

def ax2Grid(aa,N,returnFlattened=False):
    try:
        xx,yy = np.meshgrid(np.linspace(aa.get_xlim()[0],aa.get_xlim()[1], N[0]), np.linspace(aa.get_ylim()[0], aa.get_ylim()[1], N[1]))
    except TypeError:
        xx,yy = np.meshgrid(np.linspace(aa.get_xlim()[0],aa.get_xlim()[1],N),np.linspace(aa.get_ylim()[0],aa.get_ylim()[1],N))

    returnL = [xx, yy]
    if returnFlattened:
        returnL.append(np.vstack((xx.flatten(), yy.flatten())))
    
    return returnL