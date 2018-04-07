from __future__ import absolute_import
from core import *
from exhaustion import *
from layers import *
import numpy as np
import copy
import math

class Relu(Activation):
	def __init__(self, last_layer): 
		super().__init__(last_layer)

	def kernel(self, X_):
		X_ret = copy.deepcopy(X_)
		for i, v in enumerate(X_ret[0]):
			if v<=0: X_ret[0][i] = 0
		return X_ret

	def diff(self,X_):
		ret = copy.deepcopy(X_)
		for i, v in enumerate(ret[0]):
			ret[0][i] = 0 if v < 0 else 1
		return ret		

	def forward(self, X_): 
		self.before_ = X_
		self.after_ = self.kernel(X_)
		return self.after_

	def __str__(self): 
		return super().__str__('ReLU')

class Linear(Activation):
	def __init__(self, last_layer): 
		super().__init__(last_layer)

	def kernel(self, X_): 
		return X_

	def diff(self,X_):
		ret = copy.deepcopy(X_)
		for i, v in enumerate(ret[0]):
			ret[0][i] = 1
		return ret	

	def forward(self, X_): 
		self.before_ = X_
		self.after_ = self.kernel(X_)
		return self.after_

	def __str__(self): 
		return super().__str__('Linear')

class Softmax(Activation):
	def __init__(self, last_layer): 
		super().__init__(last_layer)

	def kernel(self, X_): 
		_X = X_[0]
		X_ret = copy.deepcopy(X_)
		divided_by = sum([math.exp(i) for i in _X])
		for i, v in enumerate(X_ret[0]):
			X_ret[0][i] = math.exp(v)/divided_by
		return X_ret

	def diff(self,X_):
		# https://en.wikipedia.org/wiki/Activation_function
		X_ret = self.kernel(X_)[0]
		table = np.zeros( (X_ret.shape[0], X_ret.shape[0]) )
		for i, v1 in enumerate(X_ret):
			for j, v2 in enumerate(X_ret):
				if i == j: k_delta = 1
				else: k_delta = 0
				# {\partialf_i_{x}}/{\partialx_j}
				table[i][j] = X_ret[i] * (k_delta-X_ret[j])
		return table

	def forward(self, X_): 
		self.before_ = X_
		self.after_ = self.kernel(X_)
		return self.after_

	def __str__(self): 
		return super().__str__('Softmax')

class Tanh(Activation):
	def __init__(self, last_layer): 
		super().__init__(last_layer)

	def kernel(self, X_): 
		_X = X_[0]
		X_ret = copy.deepcopy(X_)
		for i, v in enumerate(X_ret[0]):
			X_ret[0][i] = math.tanh(v)
		return X_ret

	def diff(self,X_):
		ret = copy.deepcopy(X_)
		for i, v in enumerate(ret[0]):
			ret[0][i] = 1 - math.tanh(v)**2
		return ret	

	def forward(self, X_): 
		self.before_ = X_
		self.after_ = self.kernel(X_)
		return self.after_

	def __str__(self): 
		return super().__str__('tanh')

class Sigmoid(Activation):
	def __init__(self, last_layer): 
		super().__init__(last_layer)

	def kernel(self, X_): 
		_X = X_[0]
		X_ret = copy.deepcopy(X_)
		for i, v in enumerate(X_ret[0]):
			X_ret[0][i] = 1/(1+math.exp(-v))
		return X_ret

	def diff(self,X_):
		ret = copy.deepcopy(X_)
		for i, v in enumerate(ret[0]):
			f = 1/(1+math.exp(-v))
			ret[0][i] = f * (1-f)
		return ret	

	def forward(self, X_): 
		self.before_ = X_
		self.after_ = self.kernel(X_)
		return self.after_

	def __str__(self): 
		return super().__str__('Sigmoid')

class LeakyRelu(Activation):
	def __init__(self, last_layer): 
		super().__init__(last_layer)

	def kernel(self, X_): 
		_X = X_[0]
		X_ret = copy.deepcopy(X_)
		for i, v in enumerate(X_ret[0]):
			X_ret[0][i] = v if v > 0 else 0.01 * v
		return X_ret

	def diff(self,X_):
		ret = copy.deepcopy(X_)
		for i, v in enumerate(ret[0]):
			ret[0][i] = 0.01 if x < 0 else 1
		return ret	

	def forward(self, X_): 
		self.before_ = X_
		self.after_ = self.kernel(X_)
		return self.after_

	def __str__(self): 
		return super().__str__('LeakyReLU')




if __name__ == '__main__':
	X = np.random.random((1, 10))
	a = Input(n_input=10, n_output=16)
	a = Dense(32, a)#, kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.001, bias_initializer='Zeros')
	a = Relu(a)
	a = Dense(64, a)
	a = Dense(32, a)
	a = Dense(8, a)#, kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.001,bias_initializer='Zeros')
	a = Softmax(a)
	model = Model(a)
	print(model)
	print(model.forward(X))