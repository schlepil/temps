from coreUtils import *
from dynamicalSystems.dynamicalSystems import secondOrderSys
from dynamicalSystems.inputs import boxInputCstrLFBG
from polynomial import polynomialRepr


def getSys(repr:polynomialRepr, nLink:int=5, input:List[int]=None):
    """

    :param repr:
    :param nLink:
    :param input:
    :return:
    """
    import sympy.physics.mechanics as mech

    #
    mass = 1. # [kg] mass each link is equal
    length = 1. # [m] link length; particles and joints coincide
    g = 9.81 # gravity

    input = np.arange(nLink,dtype=nint) if input is None else np.sort(narray(input, dtype=nint))
    assert input.size == np.unique(input).size and nall(0<=input) and nall(input<nLink)
    m = input.size
    n = 2*nLink

    # symbols
    qS = mech.dynamicsymbols(f"qS:{nLink}")  # Generalized coordinates
    qdS = mech.dynamicsymbols(f"qd:{nLink}")  # Generalized speeds

    t = sy.symbols('t')  # Gravity and time

    I = mech.ReferenceFrame('I')  # Inertial reference frame
    O = mech.Point('O')  # Origin point
    O.set_vel(I, 0)  # Origin's velocity is zero

    frames = [I]  # List to hold the nLink frames
    points = [O]  # List to hold the n+1 points for the origin plus particles
    particles = []  # List to hold the nLink particles
    forces = []  # List to hold the nLink applied forces; Inputs will be treated separately
    kindiffs = []  # List to hold kinematic ODE's

    for i in range(nLink):
        Bi = I.orientnew('B'+str(i), 'Axis', [sum(qS[:i+1]), I.z])  # Create a new frame that accumulates all the rotations
        Bi.set_ang_vel(I, sum(qdS[:i+1])*I.z)  # Set angular velocity z-axis common to all frames
        frames.append(Bi)  # Add it to the frames list

        Pi = points[-1].locatenew(f"P{i}", -length*Bi.y)  # Create a new point hanging position for q=0
        Pi.v2pt_theory(points[-1], I, Bi)  # Set the velocity particle and joint are both fixed in this frame
        points.append(Pi)  # Add it to the points list

        Pai = mech.Particle(f"Pa{i}", Pi, m)  # Create a new particle
        particles.append(Pai)  # Add it to the particles list

        forces.append((Pi, -mass*g*I.y))  # Set the force applied at the point due to gravity

        kindiffs.append(qS[i].diff(t)-qdS[i])  # Define the kinematic ODE:  dq_i / dt - u_i = 0

    kane = mech.KanesMethod(I, q_ind=qS, u_ind=qdS, kd_eqs=kindiffs)  # Initialize the object
    fr, frstar = kane.kanes_equations(particles, forces)  # Generate EoM's fr + frstar = 0

    # Extract mass matrix
    massMat = kane.mass_matrix
    # and system dynamics
    fDyn = kane.forcing

    # The type is ok, but we have to replace the dynamic symbols with normal ones for further usage
    q = sy.symbols(f"q:{n}")
    u = [sy.symbols(f"u{aI}") for aI in input]
    qM = sy.Matrix(nzeros((n,1)))
    uM = sy.Matrix(nzeros((m,1)))
    # format
    for i,aq in enumerate(q):
        qM[i,0] = aq
    for i,au in enumerate(u):
        uM[i,0] = au
    #replace using dict
    replDict = dict( zip(qS+qdS, q) )
    massMat = massMat.subs(replDict)
    fDyn = fDyn.subs(replDict)

    # Finally get the input mapping which is simply a selection matrix
    gInput = nzeros((nLink, m))
    gInput[input, np.arange(m)] = 1.
    gInput = sy.Matrix(gInput)

    # Now construct the sys
    dynSys = secondOrderSys(repr, massMat, fDyn, gInput, qM, uM)

    return dynSys

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
    import time
    nLink = 3
    T = time.time()
    thisRepr = polynomialRepr(2*nLink, 3)
    T = time.time()-T
    print(f"It took {T} sec to build up repr")

    T = time.time()
    thisSys = getSys(thisRepr, nLink)
    T = time.time() - T
    print(f"It took {T} sec to build up dyn eq")






































