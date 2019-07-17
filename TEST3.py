import numpy as np
import polynomial as poly
myrepr = poly.polynomialRepr(2,2)
u=np.array([1,1])
print('c',u.size)
o=u.shape
print(o)
l=len(u.shape)
print('l',l)
u=u.reshape(-1,1)
print(u)
print(u.shape)
p=[1,2,3]
print(p[1:])
print(u.size)
print(len(u.shape))