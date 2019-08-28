from polynomial import *
import plotting as plot
import time

np.random.seed(int(time.time()))

# Test gradient
thisRepr = polynomialRepr(4, 5)
thisPoly = polynomial(thisRepr, np.random.rand(thisRepr.nMonoms) - .5)
# Retrieve grad func
thisGradFun = thisPoly.getGradFunc()
thisHessFun = thisPoly.getHessFunc()

# Get a random direction
v = np.random.rand(4, 1) - 0.5
v /= norm(v.squeeze())  # Normalise

steps = np.linspace(-2, 2, 10000)
allX = v * steps

allY = thisPoly.eval2(allX).squeeze()
allGrad = thisGradFun(allX) #[dim,nPt]
allHess = thisHessFun(allX) #[nPt,dim,dim]
# Test
thisGradFun(allX[:,[0]])
thisHessFun(allX[:,[0]])

allDY = np.sum(np.multiply(allGrad, v), axis=0, keepdims=False)  # inner product
allDDY = neinsum("i,nij,j->n",v.squeeze(),allHess, v.squeeze())
allDYnum = np.diff(allY)/np.diff(steps)
allDDYnum = np.diff(allY,2)/np.diff(steps[1:],1)**2
midSteps = (steps[:-1]+steps[1:])/2.
midSteps2 = steps[1:-1]
if 0:
    ff, aa = plot.plt.subplots(1, 1)

    aa.plot(steps, allY, 'b')
    aa.plot(midSteps, allDYnum, 'r')
    aa.plot(steps, allDY, 'g')
    aa.plot(steps[:-1], allDY[:-1]-allDYnum, 'm')

    aa.plot(midSteps2, allDDYnum, '--r')
    aa.plot(steps, allDDY, '--g')
    aa.plot(midSteps2, allDDY[1:-1] - allDDYnum, '--m')

    plot.plt.show()
