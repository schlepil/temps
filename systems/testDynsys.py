from dynamicalSystems import *
from polynomial import polynomialRepr
from systems.acrobot import getSys

if __name__ == '__main__':
    
    thisRepr = polynomialRepr(4, 4)
    thisAcro = getSys(thisRepr)

    x = ((np.random.rand(4,1)-.5)*2.).astype(nfloat)

    thisAcro.evalTaylor(x)