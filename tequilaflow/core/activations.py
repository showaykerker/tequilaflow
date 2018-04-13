from __future__ import absolute_import
from .core import *
from .exhaustion import *
from .layers import *
import numpy as np
import copy
import math

class Relu(Activation):
	def __init__(self, last_layer): 
		super().__init__(last_layer)
		self.sub_type = 'Relu'

	def kernel(self, X_):
		X_ret = copy.deepcopy(X_)
		X_ret[X_ret<0] = 0
		#print('In')
		#print(X_)
		#print('Out')
		#print(ret1)
		#input(X_ret)
		return X_ret

	def diff(self,X_):
		#ret = copy.deepcopy(X_)
		#for i, v in enumerate(ret[0]):
		#	ret[0][i] = 0 if v < 0 else 1
		X_ret  = copy.deepcopy(X_)
		X_ret[X_ret<0] = 0
		X_ret[X_ret!=0] =  1
		#ret2 = super().diff(X_, self.kernel)
		return X_ret		

	def forward(self, X_): 
		return self.kernel(X_)

	def __str__(self): 
		return super().__str__('ReLU')

class Linear(Activation):
	def __init__(self, last_layer): 
		super().__init__(last_layer)
		self.sub_type = 'Linear'

	def kernel(self, X_): 
		return X_

	def diff(self,X_):
		ret = copy.deepcopy(X_)
		ret.fill(1)
		return ret	

	def forward(self, X_): 
		return self.kernel(X_)

	def __str__(self): 
		return super().__str__('Linear')

class Softmax(Output):
	def __init__(self, last_layer): 
		super().__init__(last_layer)
		self.sub_type = 'Softmax'

	def kernel(self, X_): 
		X_ret = np.zeros(X_.shape)
		for i,row in enumerate(X_):
			exps = np.exp(row-row.max())
			X_ret[i] = exps/np.sum(exps)
		return X_ret

	def diff(self,X_):
		# https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
		# https://en.wikipedia.org/wiki/Activation_function
		
		d = 1e-6
		ret = []
		X_cp = copy.deepcopy(X_)
		X_mod = copy.deepcopy(X_)

		for i in range(X_.shape[1]):
			X_a = copy.deepcopy(X_)
			X_m = copy.deepcopy(X_)
			
			X_a[0][:][i] += d
			X_m[0][:][i] -= d

	
			v = ((self.kernel(X_a)-self.kernel(X_m))/2/d)[0][0]

			ret.append(v)

		ret = np.array([ret])
		#input(ret)
		return ret

	def forward(self, X_): 
		return self.kernel(X_)

	def __str__(self): 
		return super().__str__('Softmax Output')

class Tanh(Activation):
	def __init__(self, last_layer): 
		super().__init__(last_layer)
		self.sub_type = 'Tanh'

	def kernel(self, X_): 
		return np.tanh(X_)

	def diff(self,X_):
		return 1 - self.kernel(X_)**2		

	def forward(self, X_): 
		return self.kernel(X_)

	def __str__(self): 
		return super().__str__('tanh')

class Sigmoid(Activation):
	def __init__(self, last_layer): 
		super().__init__(last_layer)
		self.sub_type = 'Sigmoid'

	def kernel(self, X_): 
		X_ret = copy.deepcopy(X_)
		X_ret = np.clip(X_ret, -500, 500)
		X_ret = 1/(1+np.exp(-X_ret))
		
		return X_ret

	def diff(self,X_):
		ret1 = self.kernel(X_)*(1-self.kernel(X_))
		#ret2 = super().diff(X_, self.kernel)
		
		return ret1

	def forward(self, X_): 
		return self.kernel(X_)

	def __str__(self): 
		return super().__str__('Sigmoid')

class LeakyRelu(Activation):
	def __init__(self, last_layer): 
		super().__init__(last_layer)
		self.sub_type = 'LeakyRelu'

	def kernel(self, X_): 
		X_ret = copy.deepcopy(X_)
		X_ret[X_ret<0] *= 0.01
		return X_ret

	def diff(self,X_):
		X_ret = copy.deepcopy(X_)
		X_ret[X_ret<0] = 0.01
		X_ret[X_ret>0] = 1

		return X_ret	

	def forward(self, X_): 
		return self.kernel(X_)

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