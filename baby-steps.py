import numpy
import theano.tensor as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

print(f)
print(f(2, 3))
print(numpy.allclose(f(16.3, 12.1), 28.4))




x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)

print(f([[1, 2], [3, 4]], [[10, 20], [30, 40]]))




