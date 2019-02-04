from coreUtils import *
from dynamicalSystems.dynamicalSystems import secondOrderSys
from polynomial import polynomialRepr

def getSys(repr:polynomialRepr):
    ##Taken from Drake - RobotLocomotion @ CSAIL
    # Implements the dynamics representing the acrobot
    # Geometry
    l1 = 1
    l2 = 2
    # Mass
    m1 = 1
    m2 = 1
    # Damping
    b1 = .1
    b2 = .1
    # Gravity center and inertial moment
    lc1 = l1/2.
    lc2 = l2/2.
    Ic1 = .083
    Ic2 = .33
    # Gravity
    g = 9.81
    
    # Constraints -> It is important the keep a certain margin between the reference
    # input and the input limits to allow for stabilitzation
    uLim = [-10,10]
    
    # Get the dynamical system
    # Original definition
    # ===============================================================================
    #       m2l1lc2 = m2*l1*lc2;  % occurs often!
    #
    #       c = cos(q(1:2,:));  s = sin(q(1:2,:));  s12 = sin(q(1,:)+q(2,:));
    #
    #       h12 = I2 + m2l1lc2*c(2);
    #       H = [ I1 + I2 + m2*l1^2 + 2*m2l1lc2*c(2), h12; h12, I2 ];
    #
    #       C = [ -2*m2l1lc2*s(2)*qd(2), -m2l1lc2*s(2)*qd(2); m2l1lc2*s(2)*qd(1), 0 ];
    #       G = g*[ m1*lc1*s(1) + m2*(l1*s(1)+lc2*s12); m2*lc2*s12 ];
    #
    #       % accumulate total C and add a damping term:
    #       C = C*qd + G + [b1;b2].*qd;
    #
    #       B = [0; 1];
    # ===============================================================================
    # H is the mass matrix
    # The dynamics
    # M.qdd + C_qdq.qd + g_q = B.tau
    
    # Get the state-space variables
    u = sy.symbols('u')
    uM = sy.Matrix([[u]])
    q = sy.symbols('q:4')
    qM = sy.Matrix([[q[0]],[q[1]],[q[2]],[q[3]]])
    
    # helper
    I1 = Ic1+m1*lc1**2
    I2 = Ic2+m2*lc2**2
    
    h12 = I2+m2*l1*lc2*sy.cos(q[1]);
    
    M = sy.Matrix([[I1+I2+m2*l1**2+2*m2*l1*lc2*sy.cos(q[1]),h12],[h12,I2]])
    
    G = g*sy.Matrix([m1*lc1*sy.sin(q[0])+m2*(l1*sy.sin(q[0])+lc2*sy.sin(q[0]+q[1])),m2*lc2*sy.sin(q[0]+q[1])]) #gravity
    # old
    # C = sMa( [[ -2*m2*l1*lc2*sympy.sin(x1)*x3, -m2*l1*lc2*sympy.sin(x1)*x3], [m2*l1*lc2*sympy.sin(x1)*x2, 0 ]] )
    # xdd = -Mi*(C*x[2:,0]+G)
    # new
    C = sy.Matrix([-2.*m2*l1*lc2*sy.sin(q[1])*q[3]*q[2]-m2*l1*lc2*sy.sin(q[1])*q[3]*q[3]+b1*q[2],m2*l1*lc2*sy.sin(q[1])*q[2]*q[2]+b2*q[3]]) #nonlinear effects
    
    # input mapping
    gInput = sy.Matrix([[0.],[1.]])
    
    dynSys = secondOrderSys(repr, M, G+C, gInput, qM, uM)
    
    return dynSys
    
