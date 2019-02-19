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
    return -10.,10.

def getSys(repr: polynomialRepr, fileName:str=None):
    ##Taken from Drake - RobotLocomotion @ CSAIL
    # Implements the dynamics representing the inverted pendulum

    if fileName is None:
        m = 1;  # % kg
        l = .5;  # % m
        b = 0.1;  # % kg m^2 /s
        lc = .5;  # % m
        I = .25;  # %m*l^2; % kg*m^2
        g = 9.81;  # % m/s^2

        # Constraints -> It is important the keep a certain margin between the reference
        # input and the input limits to allow for stabilitzation
        uLim = [-10,10]

        # Get the state-space variables
        u = sy.symbols('u')
        uM = sy.Matrix([[u]])
        q = sy.symbols('q:2')
        qM = sy.Matrix([[q[0]],[q[1]]])

        M = sy.Matrix([[I]])
        F = sy.Matrix([[m*g*lc*sy.sin(qM[0,0])+b*qM[1,0]]])
        gInput = sy.Matrix([[1.]])

        dynSys = secondOrderSys(repr,M,-F,gInput,qM,uM)
    else:
        secondOrderSys.fromFileOrDict(fileName)
    
    return dynSys

def getSysUnstablePos(deg:int):
    from coreUtils import np
    import polynomial as poly
    import trajectories as traj

    repr = poly.polynomialRepr(2, deg)

    # Get the dynamical system
    pendSys = getSys(repr)

    # Get the trajectory
    xTraj = lambda t: narray([[np.pi], [0.]], dtype=nfloat)
    dxTraj = lambda t: narray([[0.], [0.]], dtype=nfloat)

    # Compute necessary input (here 0.)
    uRefTmp = pendSys.getUopt(xTraj(0, ), dxTraj(0.), respectCstr=False, fullDeriv=True)
    uTraj = lambda t: uRefTmp.copy()

    # def __int__(self, fX: Callable, fU: Callable, nx: int, nu: int, fXd: Callable = None, tMin: float = 0., tMax: float = 1.):
    refTraj = traj.analyticTrajectory(xTraj, uTraj, 2, 1, dxTraj)
    print(refTraj(0.))

    # Get the input constraints along the refTraj
    pendSys.ctrlInput = boxInputCstrLFBG(repr, refTraj, 1, *getUlims())

    return pendSys

