from coreUtils import *
import Lyapunov.quadraticLyapunovFunction as quadLyap

from dynamicalSystems import dynamicalSystem
from trajectories import referenceTrajectory

from scipy.integrate import solve_ivp

class stopperFunction:
    def __init__(self):
        self.terminal=True

    def reset(self, t0, P0):
        pass

    def __call__(self, t:float, z:np.ndarray):
        return 1.

class limitChangeStopper(stopperFunction):
    def __init__(self, P0:np.ndarray, deltaPMax:float, diffType:'ev'):
        super(type(self), self).__init__()

        self.deltaPMax = deltaPMax
        self.diffType = diffType

        dim = int(P0.size**0.5)
        self.P0 = P0.reshape((dim,dim))

    def reset(self, t0, P0):
        super(type(self), self).__reset__(t0, P0)
        self.P0 = P0.reshape(self.P0.shape)

    def __call__(self, t: float, z: np.ndarray):

        P = z.reshape(self.P0.shape)

        if (self.diffType == 'ev'):
            diffE = nmax(nabs(eigvalsh(P-self.P0)))-self.deltaPMax
        else:
            raise RuntimeError

        return diffE

###########################

class lyapEvol():

    def __init__(self, dynSys:dynamicalSystem, refTraj:referenceTrajectory):
        self.dynSys=dynSys
        self.refTraj = refTraj

    def __call__(self, tStart:float, tDeltaMax:float, initialShape):
        pass

###########################

class noChangeLyap(lyapEvol):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, tStart: float, tDeltaMax: float, lastShape):
        return tStart-tDeltaMax, lastShape

###########################
class quadLyapTimeVaryingLQR(lyapEvol):
    def __init__(self, dynSys:dynamicalSystem, refTraj:referenceTrajectory, Q:np.ndarray=None, R:np.ndarray=None, reshape=False, restart=False):

        super(quadLyapTimeVaryingLQR, self).__init__(dynSys, refTraj)

        if Q is None:
            self.Q = np.identity(self.dynSys.nq)
        else:
            self.Q = Q
        if R is None:
            self.R = 0.01 * np.identity(self.dynSys.nu)
        else:
            self.R = R
        self.lastK = None
        self.reshape = reshape
        self.restart = restart
        self.interSteps = None

        self.limK = True
        self.ctrlSafeFac = 0.9
        self.retAll = False #Return all option for finer interpolation

        self.stopFct = stopperFunction() # type: stopperFunction

    def dz(self, z, t, xx):
        # Current P matrix
        P = z.reshape((self.dynSys.nq, self.dynSys.nq))
        # Current linearised system
        #A = self.dynSys.getLinAFast(xx)
        #B = self.dynSys.gEval(xx)
        A = self.dynSys.getTaylorApprox(xx, 1)[0][:,1:] #First column is the constant term
        B = self.dynSys.getTaylorApprox(xx, 1)[1][0,:,:] # extract matrix

        #K = np.linalg.lstsq(self.R, ndot(B.T, P))[0] #TODO check and comment this line
        #Wrong in general case
        K = mndot([inv(self.R), B.T, P])

        if self.limK:
            Cpi = inv(cholesky(P, lower=True))
            Kstar = K
            Kscale = ndot(K, Cpi)
            KscaleNorm = norm(Kscale, axis=1).reshape((self.dynSys.nu, 1))#colWiseNorm(Kscale.T).reshape((self.dynSys.nu, 1))
            uRef = self.dynSys.ctrlInput.refTraj.getU(t)#self.refTraj.uref(t).reshape((self.dynSys.nu, 1))
            uLim = nminimum(np.abs(self.dynSys.ctrlInput.getMinU(t) - uRef), np.abs(self.dynSys.ctrlInput.getMaxU(t) - uRef))#np.minimum(np.abs(self.dynSys.inputCstr.getMinU(t) - uRef), np.abs(self.dynSys.inputCstr.getMaxU(t) - uRef))
            Kstar = nmultiply(ndivide(Kstar, KscaleNorm), uLim) * self.ctrlSafeFac
            K = Kstar
        if 1:
            dP = -(ndot(A.T, P) + ndot(P, A) - 2. * mndot([P, B, K]) + self.Q)
        else:
            # TODO legacy; clean up
            # Project Q into the principal axis of dP and check if Q not too large
            dP = -(ndot(A.T, P) + ndot(P, A) - 2. * ndot(P, B, K))
            e, v = np.linalg.eigh(dP)
            Qe = np.diag(ndot(v.T, self.Q, v))
            Qe = np.minimum(np.abs(e) * .5, Qe)
            dP = dP - np.diag(ndot(v, np.diag(Qe), v.T))

        dP = (dP + dP.T) / 2. #Make symmetric
        return dP.reshape((dP.size,))

    def getDz(self, refTraj, allT, allZ):
        allXX = refTraj.xref(allT)
        allDz = np.zeros_like(allZ)
        for k in range(allZ.shape[1]):
            allDz[:, k] = self.dz(allZ[:, k], allT[k], allXX[:, [k]])
        return allDz

    # self.allShapes[n] = self.shapeGen(self.refTraj(tTest[0])[0], self.tSteps[[n,n+1]], self.refTraj, lastShape=self.allShapes[n+1])
    def __call__(self, tStart:float, tDeltaMax:float, lastShape):
        nq = self.dynSys.nq
        nu = self.dynSys.nu

        if self.retAll:
            lastShape = [lastShape[0][0][0], lastShape[1]]

        lastShape = dp(lastShape) #Make dependency free

        if self.reshape:
            M = lastShape[0]
            [w, v] = eigh(M)
            M = ndot(v.T, np.diag(np.square(w)), v)
            z0 = M.reshape((nq ** 2,))
        else:
            z0 = lastShape[0].reshape((nq ** 2,))


        if self.restart:
            z0 = z0 / lastShape[1]
            lastShape[1] = 1.
        
        # Reset the stopperFunction
        self.stopFct.reset(tStart, z0.reshape((nq,nq)))

        if self.interSteps is None:
            #z = scipy.integrate.odeint(lambda thisZ, thisT: self.dz(thisZ, thisT, refTraj.xref(thisT)), z0, [tSpan[1], tSpan[0]])[-1, :]
            sol = solve_ivp(lambda thisT, thisZ: self.dz(thisZ, thisT, self.refTraj.getX(thisT)), [tStart, tStart-tDeltaMax], z0, events=self.stopFct)
            z = sol.y[:,-1]
            tFinal = sol.t[-1]
            allZ = z.reshape((-1,1))
        else:
            allT = np.linspace(tStart, tStart-tDeltaMax, self.interSteps)
            sol = solve_ivp(lambda thisT, thisZ: self.dz(thisZ, thisT, self.refTraj.getX(thisT)), [tStart, tStart-tDeltaMax], z0, t_eval=allT, events=self.stopFct)
            allZ = sol.y
            tFinal = sol.t[-1]

            z = allZ[:,-1]
            # Flip to the time in order
            allT = np.flip(allT, 0)
            allZ = np.fliplr(allZ)
        
        if __debug__:
            for k in range(allZ.shape[1]):
                aP = allZ[:, k].reshape((nq, nq)).copy()
                try:
                    assert np.allclose(aP, aP.T)
                    # Check positiveness
                    cholesky(aP)
                except:
                    print(f"failed for {k} with \n {aP}")

        if self.restart:
            retVal = 1.
        else:
            retVal = lastShape[1]
        if self.retAll:
            allDz = self.getDz(refTraj, allT, allZ)
            allZM = [allZ[:, k].reshape((nq,nq)) for k in range(allZ.shape[1])]
            allDzM = [allDz[:, k].reshape((self.dynSys.nx, self.dynSys.nx)) for k in range(allZ.shape[1])]
            return tFinal, [[allZM, allDzM, allT], retVal]
        else:
            return tFinal, [z.reshape((nq,nq)), retVal]