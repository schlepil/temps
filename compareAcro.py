from polynomial import polynomialRepr
from systems import acrobot

import sys

sys.path.append("/home/elfuius/ownCloud/thesis/RoAAcc3/RoA/src")

import acrobot as acroOld

polyRepr = polynomialRepr(4, 4)
acroNew = acrobot.getSys(polyRepr)

