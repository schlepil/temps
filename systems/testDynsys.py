from dynamicalSystems import *
from polynomial import polynomialRepr
from systems.acrobot import getSys

if __name__ == '__main__':
    
    thisRepr = polynomialRepr(4, 4)
    thisAcro = getSys(thisRepr)