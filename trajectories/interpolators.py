from numpy import array, maximum, all, searchsorted

from scipy import interpolate

class leftNeighboor():
    """
    Emulates scipy.interp1d("left") which does not exist
    """
    def __init__(self, t, x):
        self.t = array(t).squeeze()
        self.x = array(x).reshape((-1, self.t.size))

        assert (all(t[1::] - t[0:-1] > 0.))
        self.dim = self.x.shape[0]

    def __call__(self, t):
        """
        get the left neighboor for each element in t
        :param t:
        :return:
        """
        t = array(t).squeeze()
        thisInd = maximum(searchsorted(self.t, t) - 1, 0)
        x = self.x[:, thisInd]

        return x
