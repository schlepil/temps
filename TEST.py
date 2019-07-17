import numpy as np
import polynomial as poly
import plotting as plot
import relaxations as relax

Ngrid = 100
#Creation d'une "representation"
myrepr = poly.polynomialRepr(2,4)
# Creation d'un polynome a partir de la represetation
mypoly = poly.polynomial(myrepr)
relLasserre = relax.lasserreRelax(myrepr)



center = 2. * (np.random.rand(2, 1) - .5)
P = 1.5 * (np.random.rand(2, 2) - .5)
P = np.dot(P.T, P) + .5 * np.identity(2)

mypoly.setEllipsoidalConstraint(center, 1., P)
print('_coeffs',mypoly._coeffs)
print("ggg",mypoly.getMaxDegree())
print("ggg",mypoly.repr.maxDeg)
print("ggg",mypoly.repr.listOfMonomials)
print("ggg",mypoly._coeffs)
print("ggg",mypoly.repr.listOfMonomials[int(np.argwhere(mypoly._coeffs != 0.)[-1])].sum())
lass_cstr = relax.lasserreConstraint(relLasserre,mypoly)
ff, aa = plot.plt.subplots(1,1)
plot.plotEllipse(aa, center, P, 1., faceAlpha=0.)

aa.autoscale()
print('aa',aa)
aa.axis('equal')

xx, yy, XX = plot.ax2Grid(aa, Ngrid, True)
#print("xx",xx)
#print("yy",yy)
#print("XX",XX)
z = lass_cstr.poly.eval2(XX).reshape((Ngrid, Ngrid))

aa.contour(xx, yy, z, [-0.1, 0., 0.01])

is_valid = lass_cstr.isValid(XX, simpleEval=False)
is_n_valid = np.logical_not(is_valid)

aa.plot(XX[0, is_valid], XX[1, is_valid], '.g')
aa.plot(XX[0, is_n_valid], XX[1, is_n_valid], '.r')

aa.plot(center[0, :], center[1, :], 'sk')
plot.plt.show()
