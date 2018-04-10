from core import *
from layers import *
from activations import *
from loss import *
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
n = 50000
n_ = int(n*0.8)
X = np.linspace(-1, 1, n)
np.random.shuffle(X)
Y = 0.5 * X**2 - 0.6*X + 2 + np.random.normal(0, 0.015, (n, ))
plt.scatter(X, Y)
#plt.show()
X = np.expand_dims(X, axis=1)
Y = np.expand_dims(Y, axis=1)

X_train, Y_train = X[:n_], Y[:n_]
X_test, Y_test = X[n_:], Y[n_:]

a = Input(n_input=1, n_output=4)
a = Linear(a)
a = Dense(8, a)
a = Tanh(a)
a = Dense(16, a)
a = Tanh(a)
a = Dense(8, a)
a = Tanh(a)
a = Dense(1, a)
a = Tanh(a)
a = Dense(1, a)
a = Linear(a)
a = Output(a)
model = Model(a)
model.compile(optimizer='SGD', loss='rms', lr=0.03)
#print(model)
model.update(X_train, Y_train, batch_size=4, trainig_epoch=800, X_val=X_test, Y_val=Y_test, validate_every_n_epoch=200)
#print(model)
print(X_test[1:4])
print(model.forward(X_test[1:4]))
print(Y_test[1:4])
plt.scatter(X, Y)
#plt.plot(X_test[:500], model.forward(X_test[:500]), color='green')
for i in range(1, 501):
	plt.plot(X_test[:i], model.forward(X_test[:i]), 'ro', ms=2)
plt.show()