from coreUtils import *
from dynamicalSystems.dynamicalSystems import secondOrderSys
from polynomial import polynomialRepr

# NOTE
# Using the second order system class is somewhat pointless here as computations could be done more efficiently
# due to the scalar mass "matrix" (state-independent)
# As it is however a second oreder sys we still use this class

# Zero is stable

def getUlims():
    return -10.,10.

def getSys(repr: polynomialRepr):
    ##Taken from Drake - RobotLocomotion @ CSAIL
    # Implements the dynamics representing the inverted pendulum
    
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
    
    return dynSys

