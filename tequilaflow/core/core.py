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
		self.value = 0
		if node_type == 'Activation' :
			self.vec = np.ones((1,1))
		else:
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
		self.nodes=[]
		if layer_type == 'Activation':
			for i in range(0, self.n_nodes): 
				nd = node(n_output, node_type=layer_type)
				self.nodes.append(nd)
				if not hasattr(self, 'matrix'): self.matrix = nd.get_neurons()
				else: self.matrix = np.append(self.matrix, nd.get_neurons(), axis=0)
		else:
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
		raise NotImplementedError()


	def __str__(self, activation_type=None):
		if activation_type == None:
			ret = '\t<Class %s layer> n_nodes=%d, shape=%s\n' % (self.layer_type, self.n_nodes, self.matrix.shape)
		else:
			ret = '\t<Class %s --%s-- layer> n_nodes=%d\n' % (self.layer_type, activation_type, self.n_nodes)
		for i in self.nodes: ret += i.__str__()
		return ret

class Model:
	def __init__(self, input_layers):
		self.layers = copy.deepcopy(input_layers.layer_list)
		self.input_shape = (1, self.layers[0].matrix.shape[0]-1)
		self.output_shape = (1, self.layers[-1].matrix.shape[1])
		self.compiled = None

	def predict(self, x):
		self.forward(x)

	def forward(self, x):
		vec = copy.deepcopy(x)
		for layer in self.layers:
			#print('\033[1;31m', layer, vec.shape, '\033[0m')
			vec = layer.forward(vec)
		#print('\033[1;31m', layer, vec.shape, '\033[0m')
		return vec

	def update(self, X_, Y_, batch_size=0, trainig_epoch=10):
		if not self.compiled: raise RuntimeError('Model Not Compiled.')
		for epoch in range(trainig_epoch):
			idx = np.random.choice(np.arange(len(X_)), batch_size, replace=False)
			X, Y = X_[idx], Y_[idc]
			self.init_grad_table()
			loss_vec = self.get_loss_vector(X, Y, batch_size)
			for data in range(batch_size):
				# X[data], Y[data]
				self.forward_pass(X[data])
				self.backward_pass()


	def compile(self, optimizer=None, loss=None, lr=0.01):
		self.n_input = self.layers[0].n_input
		self.n_output = self.layers[-1].n_output
		self.optimizer = optimizer
		self.lr = lr
		self.loss = loss
		self.compiled = True

	def __str__(self):
		ret = '<Class Model> Input Shape = %s, Output Shape = %s\n' % (self.input_shape, self.output_shape)
		for i in self.layers:
			ret += i.__str__()
		return ret

if __name__ == '__main__':
	from layers import *
	from activations import *
	X = np.random.normal(0, 1.2, (3,5))
	print('X=',X)
	Y = np.array([[1],[2],[3]])
	a = Input(n_input=5, n_output=3)
	a = Dense(5, a, kernel_initializer='Gaus', kernel_mean=1, kernel_std=0.1, bias_initializer='Zeros')
	a = Relu(a)
	a = Dense(6, a, kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.1,bias_initializer='Zeros')
	model = Model(a)
	print(model)
	print(model.forward(X))
	model.compile()
	#print(model.get_loss_vector(X, Y, 1))
	
