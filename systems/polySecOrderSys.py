from coreUtils import *
from dynamicalSystems.dynamicalSystems import secondOrderSys
from dynamicalSystems.inputs import boxInputCstrLFBG
from polynomial import polynomialRepr

# NOTE
# Using the second order system class is somewhat pointless here as computations could be done more efficiently
# due to the scalar mass "matrix" (state-independent)
# As it is however a second oreder sys we still use this class

# Zero is stable

def getUlims():
    return -1.,1. #Usually -2,2

def getSys(repr: polynomialRepr, fileName:str=None):
    ##Taken from Drake - RobotLocomotion @ CSAIL
    # Implements the dynamics representing the inverted pendulum

    if fileName is None:

        # Constraints -> It is important the keep a certain margin between the reference
        # input and the input limits to allow for stabilitzation
        
        # Get the state-space variables
        u = sy.symbols('u')
        uM = sy.Matrix([[u]])
        q = sy.symbols('q:2')
        qM = sy.Matrix([[q[0]],[q[1]]])

        M = sy.Matrix([[1.+q[0]-q[0]**2+q[1]**3]])
        F = sy.Matrix([[-q[0]+q[1]**3]])
        gInput = sy.Matrix([[1.+q[0]-.1*q[1]**3]])

        dynSys = secondOrderSys(repr,M,F,gInput,qM,uM)
    else:
        secondOrderSys.fromFileOrDict(fileName)

    return dynSys

def getSysBasePos(deg:int, isMoving=False):
    from coreUtils import np
    import polynomial as poly
    import trajectories as traj

    repr = poly.polynomialRepr(2, deg)

    # Get the dynamical system
    polySecSys = getSys(repr)

    # Get the trajectory
    if isMoving:
        xTraj = lambda t: narray([[0.1*t], [0.1]], dtype=nfloat)
        dxTraj = lambda t: narray([[0.1], [0.]], dtype=nfloat)
    else:
        xTraj = lambda t: narray([[0.], [0.]], dtype=nfloat)
        dxTraj = lambda t: narray([[0.], [0.]], dtype=nfloat)

    # Compute necessary input (here 0.)
    uTraj = lambda t: polySecSys.getUopt(xTraj(t), dxTraj(t), respectCstr=False, fullDeriv=True)

    # def __int__(self, fX: Callable, fU: Callable, nx: int, nu: int, fXd: Callable = None, tMin: float = 0., tMax: float = 1.):
    refTraj = traj.analyticTrajectory(xTraj, uTraj, 2, 1, dxTraj)
    print(refTraj(0.))

    # Get the input constraints along the refTraj
    polySecSys.ctrlInput = boxInputCstrLFBG(repr, refTraj, 1, *getUlims())

    return polySecSys

