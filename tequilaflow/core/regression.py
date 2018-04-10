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
Y = 0.0001*X**4 - 0.02 * X**3 + 0.1 * X**2 - 0.3*X + 2 + np.random.normal(0, 0.03, (n, ))
plt.scatter(X, Y)
#plt.show()
X = np.expand_dims(X, axis=1)
Y = np.expand_dims(Y, axis=1)

X_train, Y_train = X[:n_], Y[:n_]
X_test, Y_test = X[n_:], Y[n_:]

a = Input(n_input=1, n_output=2)
a = Linear(a)
a = Dense(4, a)
a = Tanh(a)
a = Dense(8, a)
a = Tanh(a)
a = Dense(8, a)
a = Tanh(a)
a = Dense(1, a)
a = Linear(a)
a = Output(a)
model = Model(a)
model.compile(optimizer='SGD', loss='rms', lr=0.008)
#print(model)
model.update(X_train, Y_train, batch_size=4, trainig_epoch=2400, X_val=X_test, Y_val=Y_test, validate_every_n_epoch=200)
#print(model)
print(X_test[1:4])
print(model.forward(X_test[1:4]))
print(Y_test[1:4])
plt.scatter(X_train, Y_train, color='blue', s=3)
#plt.plot(X_test[:500], model.forward(X_test[:500]), color='green')

plt.scatter(X_test, Y_test, color='green', s=2)
plt.scatter(X_test, model.forward(X_test), color='red', s=1)

plt.show()