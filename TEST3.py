import numpy as np
import polynomial as poly
c=np.array([[1,2,3],[4,5,6]])
b=np.zeros(c.shape)
for i in range(c.shape[0]):
    b[i,:]=c[i,:]
print(c[1,])
print(b.shape[0])
isValid=np.ones((5,), dtype=np.bool_)
print(isValid)
a=np.array([1,2,3,4,5,6,7,8,9,10])
a*=a[:a.shape[0]]
print(a)
print(a[:a.shape[0]])
print(c[:,:c.shape[1]])
print(c.shape)
d=np.array([1,2])
print(c.reshape((-1,d.shape[0]))*d)
print(c[1,:])