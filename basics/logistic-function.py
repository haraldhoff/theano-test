import theano
import theano.tensor as T
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)

print(logistic([[0, 1], [-1, -2]]))



s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = theano.function([x], s2)

print(logistic2([[0, 1], [-1, -2]]))



a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = theano.function([a, b], [diff, abs_diff, diff_squared])

print(f([[1, 1], [1, 1]], [[0, 1], [2, 3]]))






