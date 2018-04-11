from __future__ import absolute_import
from .core import *
from .exhaustion import *
import numpy as np
import copy


class Input(layer):
	def __init__(self, n_input, n_output, 
				 kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.03,
				 bias_initializer='Ones', bias_mean=0, bias_std=0.03):
		super().__init__(n_input=n_input, n_output=n_output, layer_type='Input', 
						 kernel_initializer=kernel_initializer, kernel_mean=kernel_mean, kernel_std=kernel_std,
						 bias_initializer=bias_initializer, bias_mean=bias_mean, bias_std=bias_std)
		self.n_output = n_output

	def forward(self, X_): 
		#print(self.matrix)
		n = X_.shape[0]
		add_ = np.array([[1]]*n)
		ret = np.hstack((X_, add_))
		ret = ret.dot(self.matrix)

		return ret
	def __str__(self): return super().__str__()

class Output(layer):
	def __init__(self, last_layer):
		super().__init__(n_input=last_layer.n_output, n_output=last_layer.n_output, last_layer=last_layer, layer_type='Output')

	def forward(self, X_):
		return X_

	def __str__(self): return super().__str__()		


class Dense(layer):
	def __init__(self, n_output, last_layer, n_input=None, 
				 kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.1,
				 bias_initializer='Ones', bias_mean=0, bias_std=0.1):
		super().__init__(n_input=last_layer.n_output, n_output=n_output, last_layer=last_layer, layer_type='Dense', 
						 kernel_initializer=kernel_initializer, kernel_mean=kernel_mean, kernel_std=kernel_std,
						 bias_initializer=bias_initializer, bias_mean=bias_mean, bias_std=bias_std)

	def forward(self, X_):
		
		n = X_.shape[0]
		add_ = np.array([[1]]*n)
		ret = np.hstack((X_, add_))
		ret = ret.dot(self.matrix)
		#print(self.matrix)
		#ret = np.reshape(ret, (1, len(ret)))
		return ret
		
	def __str__(self): return super().__str__()


class Activation(layer):
	def __init__(self, last_layer):
		super().__init__(n_input=last_layer.n_output, n_output=last_layer.n_output, last_layer=last_layer, layer_type='Activation')

	def kernel(self, X_):
		raise NotImplementedError()

	def diff(self, X_):
		raise NotImplementedError()

	def forward(self, X_): 
		raise NotImplementedError()
		
	def __str__(self, activation_type): return super().__str__(activation_type)



if __name__ == '__main__':
	X = np.random.random((6, 3))
	a = Input(n_input=3, n_output=2)
	a = Dense(4, a)#, kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.001, bias_initializer='Zeros')
	a = Dense(2, a)
	a = Dense(3, a)
	a = Dense(1, a)#, kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.001,bias_initializer='Zeros')
	model = Model(a)
	print(model)
	print(model.forward(X))