from __future__ import absolute_import
from exhaustion import *
import numpy as np
import copy

class node: 
	# Single node to multiple output
	# type_ : ['regular', 'bias', 'activation']
	def __init__(self, n_output, node_type='regular', initializer='Gaus', mean=0, std=0.1):
		if initializer not in INITIALIZERS: raise ValueError('Initializer %s not recognized.'%(str(initializer)))
		self.n_output = n_output
		self.node_type = node_type
		if   initializer == 'Gaus'  : self.vec = np.random.normal(mean, std, (1, n_output))
		elif initializer == 'Zeros' : self.vec = np.zeros((1, n_output))
		elif initializer == 'Ones'  : self.vec = np.ones((1, n_output))
	def get_neurons(self):
		return self.vec
	def __str__(self):
		return '\t\t<Class node> node_type=%7s, shape=%s\n' % (self.node_type, self.vec.shape)

class layer:
	def __init__(self, n_input=None, n_output=None, last_layer=None, layer_type='Dense', 
					kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.1,
					bias_initializer='Gaus', bias_mean=0, bias_std=0.1):
		if layer_type not in LAYERS: raise ValueError('Layer type %s not recognized.' % ( str(type_) ))
		self.n_output = n_output
		self.n_input = self.n_nodes = n_input
		self.last_layer = copy.deepcopy(last_layer)
		self.layer_type = layer_type
		self.nodes = []
		for i in range(0, self.n_nodes): 
			nd = node(n_output, node_type='regular', initializer=kernel_initializer, mean=kernel_mean, std=kernel_std,)
			self.nodes.append(nd)
			if not hasattr(self, 'matrix'): self.matrix = nd.get_neurons()
			else: self.matrix = np.append(self.matrix, nd.get_neurons(), axis=0)
		if layer_type in BIAS_LAYERS: 
			self.n_nodes += 1
			nd = node(n_output, node_type='bias', initializer=bias_initializer, mean=bias_mean, std=bias_std,)
			self.nodes.append(nd)
			self.matrix = np.append(self.matrix, nd.get_neurons(), axis=0)
		if last_layer == None and layer_type=='Input' : self.layer_list = [self] # Input layer
		else: 
			self.layer_list = copy.deepcopy( last_layer.layer_list )
			self.layer_list.append(self)

	def forward(self):
		raise NotImplementError()

	def __str__(self, type_):
		ret = '\t<Class %s layer> n_nodes=%d, shape=%s\n' % (self.layer_type, self.n_nodes, self.matrix.shape)
		for i in self.nodes: ret += i.__str__()
		return ret

class Input(layer):
	def __init__(self, n_input, n_output, 
				 kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.1,
				 bias_initializer='Gaus', bias_mean=0, bias_std=0.1):
		super().__init__(n_input=n_input, n_output=n_output, layer_type='Input', 
						 kernel_initializer=kernel_initializer, kernel_mean=kernel_mean, kernel_std=kernel_std,
						 bias_initializer=bias_initializer, bias_mean=bias_mean, bias_std=bias_std)
		self.n_output = n_output

	def forward(self, X_): return np.append(X_, [1]).dot(self.matrix)
	def __str__(self): return super().__str__(super)

class Dense(layer):
	def __init__(self, n_output, last_layer, n_input=None, 
				 kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.1,
				 bias_initializer='Gaus', bias_mean=0, bias_std=0.1):
		super().__init__(n_input=last_layer.n_output, n_output=n_output, last_layer=last_layer, layer_type='Dense', 
						 kernel_initializer=kernel_initializer, kernel_mean=kernel_mean, kernel_std=kernel_std,
						 bias_initializer=bias_initializer, bias_mean=bias_mean, bias_std=bias_std)

	def forward(self, X_):return np.append(X_, [1]).dot(self.matrix)
	def __str__(self): return super().__str__(super)

class Model:
	def __init__(self, input_layers):
		self.layers = copy.deepcopy(input_layers.layer_list)
		self.input_shape = (1, self.layers[0].matrix.shape[0]-1)
		self.output_shape = (1, self.layers[-1].matrix.shape[1])

	def forward(self, X_):
		vec = copy.deepcopy(X_)
		for layer in self.layers:
			vec = layer.forward(vec)
		return vec


	def __str__(self):
		ret = '<Class Model> Input Shape = %s, Output Shape = %s\n' % (self.input_shape, self.output_shape)
		for i in self.layers:
			ret += i.__str__()
		return ret


if __name__ == '__main__':
	X = np.array([[1,2,3]])
	a = Input(n_input=3, n_output=5)
	a = Dense(5, a, kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.001, bias_initializer='Zeros')
	a = Dense(3, a, kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.001,bias_initializer='Zeros')
	model = Model(a)
	print(model)
	print(model.forward(X))
	