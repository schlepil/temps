from coreUtils import *
import Lyapunov as lyap
import plotting as plot
from plotting import plt

from systems.pendulum import getSysUnstablePos

pendSys = getSysUnstablePos(4)

P = np.identity(2)
alpha = 1.

sep = np.random.rand(2,2)-.5
#sep = np.identity(2)
while abs(ndot(sep[[0],:], sep[[1],:].T)) < .4:
    sep = np.random.rand(2,2)-.5
    sep /= norm(sep, axis=1, keepdims=True)
compSize = [(np.random.rand(2,)-.5)*1. for _ in range(sep.shape[0])]


x = plot.getV(P, n=101, alpha=alpha)


quadratV = lyap.quadraticLyapunovFunction(pendSys, P, alpha)
pieceQuadratV = lyap.piecewiseQuadraticLyapunovFunction(pendSys, P, alpha)
pieceQuadratV.setPlanesAndComp(sep, compSize)

y = pieceQuadratV.sphere2Ellip(x.copy()) # This is incorrect as the points can change sector during transformation -> iterate?

ff,aa = plt.subplots(1,1)
plot.myQuiver(aa, np.zeros((2,2)), sep.T*.5, c='k')


for k in range(sep.shape[0]):
    ppSep = np.hstack((ndot(plot.Rot(np.pi / 2.), sep[[k],:].T), ndot(plot.Rot(-np.pi / 2.), sep[[k],:].T)))
    aa.plot(ppSep[0,:], ppSep[1,:], '-b')


aa.plot(x[0,:], x[1,:], '.-g')
aa.plot(y[0,:], y[1,:], '.-r')

aa.axis('equal')


ff,aa = plt.subplots(1,2, figsize=(1+2*4,4))

xx,yy = np.meshgrid(np.linspace(-2,2,100), np.linspace(-2,2,100))
XX = np.vstack((xx.flatten(), yy.flatten()))

Vxq = quadratV.evalV(XX, kd=False)
Vxpq = pieceQuadratV.evalV(XX, kd=False)

lvlq = np.linspace(0., nmax(Vxq)**.5, 13)**2.
lvlpq = np.linspace(0., nmax(Vxpq)**.5, 13)**2.

aa[0].contour(xx,yy,Vxq.reshape((100,100)), lvlq)
aa[1].contour(xx,yy,Vxpq.reshape((100,100)), lvlpq)

plt.show()

