import numpy as np
import polynomial as poly
a=10
b=3
c={0:{'0':1,'g':10},1:{'0':3,'bb':11}}
for i in range(len(c)):
    print(c[i]['0'])
g=(1,2,3)
h=(4,5,6)
# print(a%b)
# print(a//b)
# r=[1,2,1,4,4]
# print(r[1:])
print(g.__add__(h))