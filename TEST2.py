import copy
import numpy as np
from scipy import sparse
from sympy import plot_implicit,symbols,Eq
x, y = symbols('x y')
p1=plot_implicit(Eq(1.0*x**2+1.*y**2,1),show=False)
p2=plot_implicit(Eq(1.0*(x+1)**2+1.*y**2,3),show=False)
p3=plot_implicit(Eq(1.0*x**2+1.*y**2,5),show=False)
p1.append(p2[0])
p1.append(p3[0])
# p2=plot_implicit(Eq(x**2 + y**2, 5),(x, -5, 5), (y, -2, 2),adaptive=False, points=400)
# p1.show()
g=bool(1==2)
gg=bool(1==1) and bool(1==2)
print(gg)
s=bool(-0.00001<0<0.00001)
print(s)
sss=bool(-0.00001<10<0.00001)
print(sss)