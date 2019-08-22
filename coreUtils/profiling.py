#Precise/fast enough?
from time import time

def countedtimer(fn):
    def wrapper(*args, **kwargs):
        wrapper.called += 1
        t_ = time()
        res = fn(*args, **kwargs)
        wrapper.time += time()-t_
        return res
    wrapper.called = 0
    wrapper.time = 0.
    wrapper.__name__ = fn.__name__
    return wrapper