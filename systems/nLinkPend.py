from coreUtils import *
from dynamicalSystems.dynamicalSystems import secondOrderSys
from dynamicalSystems.inputs import boxInputCstrLFBG
from polynomial import polynomialRepr


from sympy.physics.mechanics import *

def getSys(nLink=5):
    from sympy import symbols

    n = 5
    q = dynamicsymbols('q:'+str(n+1))  # Generalized coordinates
    u = dynamicsymbols('u:'+str(n+1))  # Generalized speeds
    f = dynamicsymbols('f')  # Force applied to the cart
    mom = dynamicsymbols('mom')  # Force applied to the cart

    m = symbols('m:'+str(n+1))  # Mass of each bob
    l = symbols('l:'+str(n))  # Length of each link
    g, t = symbols('g t')  # Gravity and time

    I = ReferenceFrame('I')  # Inertial reference frame
    O = Point('O')  # Origin point
    O.set_vel(I, 0)  # Origin's velocity is zero

    P0 = Point('P0')  # Hinge point of top link
    P0.set_pos(O, q[0]*I.x)  # Set the position of P0
    P0.set_vel(I, u[0]*I.x)  # Set the velocity of P0
    Pa0 = Particle('Pa0', P0, m[0])

    frames = [I]  # List to hold the n + 1 frames
    points = [P0]  # List to hold the n + 1 points
    particles = [Pa0]  # List to hold the n + 1 particles
    forces = [(P0, f*I.x-m[0]*g*I.y)]  # List to hold the n + 1 applied forces, including the input force, f
    kindiffs = [q[0].diff(t)-u[0]]  # List to hold kinematic ODE's

    for i in range(n):
        Bi = I.orientnew('B'+str(i), 'Axis', [q[i+1], I.z])  # Create a new frame
        Bi.set_ang_vel(I, u[i+1]*I.z)  # Set angular velocity
        frames.append(Bi)  # Add it to the frames list

        Pi = points[-1].locatenew('P'+str(i+1), l[i]*Bi.x)  # Create a new point
        Pi.v2pt_theory(points[-1], I, Bi)  # Set the velocity
        points.append(Pi)  # Add it to the points list

        Pai = Particle('Pa'+str(i+1), Pi, m[i+1])  # Create a new particle
        particles.append(Pai)  # Add it to the particles list

        forces.append((Pi, -m[i+1]*g*I.y))  # Set the force applied at the point

        kindiffs.append(q[i+1].diff(t)-u[i+1])  # Define the kinematic ODE:  dq_i / dt - u_i = 0

    kane = KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)  # Initialize the object
    fr, frstar = kane.kanes_equations(particles, forces)  # Generate EoM's fr + frstar = 0
    fr, frstar = kane.kanes_equations(particles, forces+[(frames[2], mom*frames[2].z)])  # Generate EoM's fr + frstar = 0
    fr, frstar = kane.kanes_equations(particles, forces+[(frames[2], mom*frames[2].z), (frames[3], -mom*frames[3].z)])  # Generate EoM's fr + frstar = 0

    return kane

def getSysWhatIsWrong(repr: polynomialRepr, nLink:int=3, input:List[int]=None, fileName:str=None):
    import sympy.physics.mechanics as mech

    """
    Creates the n-link pendulum plane pendulum. indices of controlled inputs are given in input
    if none, all links are actuated
    :param repr:
    :param n:
    :param input:
    :param fileName:
    :return:
    """

    # All links are equal
    m = 1. #[kg] total mass of each link
    l = 1. #[m] length of each link
    g = -9.81
    b = 0. # [N*m*s/rad] viscous friction

    input = np.arange(nLink,dtype=nint) if input is None else np.sort(narray(input, dtype=nint))
    assert input.size == np.unique(input).size and nall(0<=input) and nall(input<nLink)
    m = input.size
    n = 2*nLink

    # Work some sympy magic to get the equations
    # First get the usual variables
    q = sy.symbols(f"q:{n}")
    u = [sy.symbols(f"u{aInd}") for aInd in input]
    #Get the matrix variable
    qM = sy.Matrix(nzeros((n,1)))
    uM = sy.Matrix(nzeros((m,1)))
    for i in range(n):
        qM[i,0] = q[i]
    for i,au in enumerate(u):
        uM[i,0] = au

    # Get the dynamic variables used for kanes later on
    ms = sy.symbols("m")
    t = sy.symbols("t")

    qs = mech.dynamicsymbols(f"q:{n}") # State
    us = [mech.dynamicsymbols(f"u{aInd}") for aInd in input]

    # Kanes cannot separate input and system dynamics
    # therefore we have to solve twice, once without u
    # once setting ms=0 so only the input dynamic remains

    # For now we use a point mass located at the end of each link
    # TODO code a switch between pointmass and cylinder (rigid body)

    # Origin
    I = mech.ReferenceFrame('I')  # Inertial reference frame
    O = mech.Point('O')  # Origin point
    O.set_vel(I, 0)

    frames = [I]
    bodyFrames = []
    pointsJoints = [O]
    pointsPart = []
    partOrBody = []
    forces = []
    torques = [(I,us[0]*I.z)] if input[0] == 0 else []
    kindiffs = []

    # Loop over segments
    fOld = I # Current frame
    cIn = len(torques) # Next input number
    for i in range(nLink):
        fNew = fOld.orientnew(f"F{i}", 'Axis', [qs[i], fOld.z])  # Create a new frame, turn around z, accumulate changes
        fNew.set_ang_vel(fOld, qs[nLink+i]*fOld.z)  # Set angular velocity with respect to last frame
        frames.append(fNew)  # Add it to the frames list
        #recursion
        bodyFrames.append(fNew)
        frames.append(fNew)

        Pi = pointsJoints[-1].locatenew(f"P{i}", -l*fNew.y)  # Create a new point. For q=0 hanging
        Pi.v2pt_theory(pointsJoints[-1], I, fNew)  # Set the velocity both points (mass point and joint point) are fixed in fNew
        #Rigid body, this point is not to move within its frame
        #Pi.set_vel(fNew, 0.*fNew.x)
        pointsJoints.append(Pi)  # Add it to the points list <-> masses and joints coincide

        Pai = mech.Particle(f"Pa{i}", Pi, ms)  # Create a new particle
        partOrBody.append(Pai)  # Add it to the particles list

        forces.append((Pi, ms*g*I.y))  # Set the force applied at the point

        # The torques between the links
        # friction
        if b != 0.:
            if i>0:
                # Acts in reverse direction on last frame
                torques.append( (bodyFrames[i-1], +b*qs[nLink+i]*bodyFrames[i-1].z) ) # Both z axis are equal
            torques.append( (bodyFrames[i], -b*qs[nLink+i]*bodyFrames[i].z) )

        # input
        if (i>0) and (i in input):
            torques.append( (bodyFrames[i-1], -us[cIn]*bodyFrames[i-1].z) ) # Both z axis are equal
            torques.append( (bodyFrames[i], us[cIn]*bodyFrames[i].z) )
            cIn += 1 #next input

        kindiffs.append(qs[i].diff(t)-qs[nLink+i])  # Define the kinematic ODE:  dq_i / dt - u_i = 0

        # Update
        fOld = fNew

    kane = mech.KanesMethod(I, q_ind=qs[:nLink], u_ind=us, kd_eqs=kindiffs)
    f,fr = kane.kanes_equations(partOrBody, forces+torques)
    print(f,fr)

    return kane




if __name__ == "__main__":
    nLink = 3
    thisRepr = polynomialRepr(2*nLink, 3)
    thisSys = getSys(thisRepr, nLink)
    test()






































