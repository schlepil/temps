from coreUtils.coreUtilsImport import np

np.set_printoptions(precision=6, linewidth=225, floatmode='maxprec_equal')

floatEps = np.finfo(np.float_).eps*2.

doPlot = False

absTolCstr = 1.e-6
numericEpsPos = -1.e-6

usePenaltyOrdering = True # Order proofs with respect to "last" result

alphaAbsMin = 1.e-7 # If the size of the regions goes lower than this, consider the system to be non-stabilisable
cholDiagMin = 1.e-7 # If any diagonal elements of the cholesky decomp are smaller than this, P is no-longer consider psd
cholDiagMax = 1.e8 # If any diagonal elements of the cholesky decomp are larger than this, the solver runs into problems and also the zone is
# insanely large

doTiming = True