from coreUtils import *

@njit(cache=True)
def evalPolyLyap_Numba(P, monomP, f, monomF, g, monomG, dx0, monomDX0, u, monomU, idxMat, coeffsOut, Pdot):
    assert 0, "call and code needs debugging"
    """
    Generates the polynomial corresponding to the derivative of the Lyapunov function
    V = z'.P.z
    V = 2.*z'.P.dz + z'.Pdot.z
    where z is the vector of monomials stored in monomP
    f and monomF have to store the derivatives of z (system dynamics)
    g and monomG have to store the derivatives of z (input dynamics)
    u and monomU respresent the control law
    Pdot : Time-derivative of the Lyapunov region. If None: P != func(t)
    #Attention for efficiency, the monomials have to be given as the variable number
    # TODO check if this is computationally intensive and if so do some precomputations
    :param P:
    :param monomP:
    :param fTaylor:
    :param monomF:
    :param gTaylor:
    :param monomG:
    :param dx0:
    :param monomDX0:
    :param uOpt:
    :param uMonom:
    :param idxMat:
    :param coeffsOut:
    :param Pdot:
    :return:
    """
    
    #Part one
    #Compute z'*P*f*y
    # z'*P*f*y = sum_j sum_i sum_l z_i*P[i,j]*A[j,l]*y[l]
    tmpValPij = 0.0 # "declare" before
    for j in range(P.shape[0]):
        for i,amz in enumerate(monomP): #Enumerate is equally fast (maybe even faster) then looping + accessing
            tmpValPij = P[i,j] #Avoid repeated access
            if tmpValPij != 0.0:
                for l,amy in enumerate(monomF):
                    if [j,l] != 0.0:
                        coeffsOut[idxMat[amz,amy]]=tmpValPij*f[j,l]

    # Part two
    # Compute z'*P*g*y*u*w
    # z -> monomials of P
    # y -> monomials of input dynamics
    # w -> monomials of control law
    
    # Compute z'*P*g*y*u*w = sum_a sum_b P[b,a] sum_c sum_d g[c,a,d] (z[b]y[c]) sum_e u[d,e] w[e]
    # a -> final inner prod
    # b -> monom z monomP
    # c -> monom y monomG
    # d -> mat mul input dynamics
    # e -> monom w monomU
    tmpValPba = 0.0 # "declare" before
    for a in range(P.shape[0]):
        for b,amz in enumerate(monomP):
            tmpValPba = P[b,a]
            if tmpValPba == 0.0:
                continue #skip
            for c,amy in enumerate(monomG):
                for d in range(g.shape[1]):
                    tmpValPbaGcad = tmpValPba*g[c,a,d]
                    if tmpValPbaGcad == 0.0:
                        continue #skip
                    tmpVarNumXBYC = idxMat[amz,amy]
                    for e,amw in enumerate(monomU):
                        tmpValUde = u[d,e]
                        if tmpValUde == 0.0:
                            continue
                        coeffsOut[idxMat[tmpVarNumXBYC,amw]] += tmpValPbaGcad*tmpValUde
    
    # Part three
    # Contribution of reference velocity
    # Vdot = (x-x_ref)'.P.(x_dot - x_ref_dot)
    for a,amDX in enumerate(monomDX0):
        for b,amz in enumerate(monomP):
            # No need to check for zeros, zeros are not very likely and there is not much to gain
            coeffsOut[idxMat[amDX,amz]] += P[b,a]*dx0[a,0]
    
    # Multiply by two
    coeffsOut *= 2.
    
    # Add time-derivative
    # Use symmetry
    for a,ampa in enumerate(monomP):
        # Diagonal
        coeffsOut[idxMat[ampa,ampa]] += Pdot[a,a]
        for b,ampb in enumerate(monomP[a+1:]):
            #Strict Upper triang
            coeffsOut[idxMat[ampa,ampb]] += Pdot[a,b]
    
    return coeffsOut


@njit(cache=True)
def evalPolyLyapAsArray_Numba(P, monomP, f, monomF, g, monomG, dx0, monomDX0, u, monomU, idxMat, coeffsOut, Pdot):
    """
    Generates the polynomial corresponding to the derivative of the Lyapunov function
    V = z'.P.z
    V = 2.*z'.P.dz + z'.Pdot.z
    where z is the vector of monomials stored in monomP
    f and monomF have to store the derivatives of z (system dynamics)
    g and monomG have to store the derivatives of z (input dynamics)
    u and monomU respresent the control law
    Pdot : Time-derivative of the Lyapunov region. If None: P != func(t)
    #Attention for efficiency, the monomials have to be given as the variable number
    # TODO check if this is computationally intensive and if so do some precomputations
    :param P:
    :param monomP:
    :param fTaylor:
    :param monomF:
    :param gTaylor:
    :param monomG:
    :param dx0:
    :param monomDX0:
    :param uOpt:
    :param uMonom:
    :param idxMat:
    :param coeffsOut:
    :param Pdot:
    :return:
    """

    # Part one -> System dynamics
    # Compute z'*P*f*y
    # z'*P*f*y = sum_j sum_i sum_l z_i*P[i,j]*A[j,l]*y[l]
    tmpValPij = 0.0  # "declare" before
    for j in range(P.shape[0]):
        for i, amz in enumerate(monomP):  # Enumerate is equally fast (maybe even faster) then looping + accessing
            tmpValPij = P[i, j]  # Avoid repeated access
            if tmpValPij != 0.0:
                for l, amy in enumerate(monomF):
                    if f[j, l] != 0.0:
                        coeffsOut[0,idxMat[amz, amy]] = tmpValPij * f[j, l]

    # Part two
    # Compute z'*P*g*y*u*w
    # z -> monomials of P
    # y -> monomials of input dynamics
    # w -> monomials of control law

    # Compute z'*P*g*y*u*w = sum_a sum_b P[b,a] sum_c sum_d g[c,a,d] (z[b]y[c]) sum_e u[d,e] w[e]
    # a -> final inner prod
    # b -> monom z monomP
    # c -> monom y monomG
    # d -> mat mul input dynamics
    # e -> monom w monomU
    tmpValPba = 0.0  # "declare" before
    for a in range(P.shape[0]):
        for b, amz in enumerate(monomP):
            tmpValPba = P[b, a]
            if tmpValPba == 0.0:
                continue  # skip
            for c, amy in enumerate(monomG):
                for d in range(g.shape[2]):
                    tmpValPbaGcad = tmpValPba * g[c, a, d]
                    if tmpValPbaGcad == 0.0:
                        continue  # skip
                    tmpVarNumXBYC = idxMat[amz, amy]
                    for e, amw in enumerate(monomU):
                        tmpValUde = u[d, e]
                        if tmpValUde == 0.0:
                            continue
                        coeffsOut[1+d,idxMat[tmpVarNumXBYC, amw]] += (tmpValPbaGcad * tmpValUde) # TODO this causes a lot of jumping around in the array..

    # Part three
    # Contribution of reference velocity
    # Vdot = (x-x_ref)'.P.(x_dot - x_ref_dot)
    for a, amDX in enumerate(monomDX0):
        for b, amz in enumerate(monomP):
            # No need to check for zeros, zeros are not very likely and there is not much to gain
            coeffsOut[0,idxMat[amDX, amz]] -= P[b, a] * dx0[a, 0]

    # Multiply by two
    coeffsOut *= 2.

    # Add time-derivative (if necessary)
        # Use symmetry
    for a, ampa in enumerate(monomP):
        # Diagonal
        coeffsOut[0,idxMat[ampa, ampa]] += Pdot[a, a]
        for b, ampb in enumerate(monomP[a + 1:]):
            # Strict Upper triang
            coeffsOut[0,idxMat[ampa, ampb]] += (2.*Pdot[a, a+1+b])

    return coeffsOut
    
    
    
@njit(cache=True)
def compPolyCstr_Numba(monomsP:np.ndarray, PG:np.ndarray, monomsG:np.ndarray, which:np.ndarray, idxMat:np.ndarray, coeffsOut:np.ndarray):
    """
    Computes the coefficients of the polynomial constraints defining the optimal input regions
    PG is each taylor term of the input dynamics premultiplied with P
    :param monomsP:
    :param PG:
    :param monomsG:
    :param which:
    :param idxMat:
    :param coeffsOut:
    :return:
    """
    # Wait for numba 0.43
    #if __debug__:
    #    assert which.size == coeffsOut.shape[0]
    
    # x'*PG_c[:,idx]*y_c
    for k, aIdx in enumerate(which):
        for a, amg in enumerate(monomsG):
            for b, amp in enumerate(monomsP):
                tmpVarNum = idxMat[amp,amg]
                for aIdx in which:
                    coeffsOut[k, tmpVarNum] += PG[a,b,aIdx] #which holds to index of the input to be considered. The aIdx-th input extracts the
                    # corresponding column of PG
    
    return coeffsOut
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    