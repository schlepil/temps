from coreUtils.coreUtilsImport import np

np.set_printoptions(precision=6, linewidth=225, floatmode='maxprec_equal')

floatEps = np.finfo(np.float_).eps*2.

doPlot = False

absTolCstr = 1.e-6
numericEpsPos = -1.e-6

usePenaltyOrdering = True # Order proofs with respect to "last" result