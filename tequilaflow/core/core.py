from __future__ import absolute_import
from exhaustion import *
import numpy as np
import copy
from loss import *

class node: 
	# Single node to multiple output
	# type_ : ['regular', 'bias', 'activation']
	def __init__(self, n_output, node_type='regular', initializer='Gaus', mean=0, std=0.1):
		if initializer not in INITIALIZERS: raise ValueError('Initializer %s not recognized.'%(str(initializer)))
		self.n_output = n_output
		self.type = node_type
		self.value = None
		if node_type == 'Activation' :
			self.vec = np.ones((1,1))
		else:
			if   initializer == 'Gaus'  : self.vec = np.random.normal(mean, std, (1, n_output))
			elif initializer == 'Zeros' : self.vec = np.zeros((1, n_output))
			elif initializer == 'Ones'  : self.vec = np.ones((1, n_output))

	# Call Before Back Propagation.
	def init_grad(self):
		self.grad_forward = np.zeros(self.vec.shape)
		self.grad_backward = np.zeros(self.vec.shape)
		self.grad = np.zeros(self.vec.shape)
		self.value = 1

	# Fill self.value for forward_pass
	def forward_pass(self):
		self.grad_forward.fill(self.value)

	def get_neurons(self):
		return self.vec

	def __str__(self):
		return '\t\t<Class node> type=%7s, value=%5.2f, shape=%s\t\t%s\n' % (self.type, self.value, self.vec.shape, str(self.vec))

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
		if layer_type == 'Activation': # Don't need bias node.
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

	# Init node's value. Call before back propagation
	def init_grad_table(self):
		for node in self.nodes: node.init_grad()

	def forward(self):
		raise NotImplementedError()


	def __str__(self, activation_type=None):
		if activation_type == None:
			ret = '\t<Class %s layer> n_nodes=%d, shape=%s\n' % (self.layer_type, self.n_nodes, self.matrix.shape)
		else:
			ret = '\t<Class %s layer> ===%s=== n_nodes=%d\n' % (self.layer_type, activation_type, self.n_nodes)
		for i in self.nodes: ret += i.__str__()
		return ret

class Model:
	def __init__(self, input_layers):
		self.layers = copy.deepcopy(input_layers.layer_list)
		self.input_shape = (1, self.layers[0].matrix.shape[0]-1)
		self.output_shape = (1, self.layers[-1].matrix.shape[1])
		self.compiled = None

	# Do exactly same thing as Model.forward.
	def predict(self, x):
		self.forward(x)

	# Feed Forward
	def forward(self, x):
		vec = copy.deepcopy(x)
		for layer in self.layers:
			vec = layer.forward(vec)
		return vec

	# Initialize Grad Table. Call before Back Propagation
	def init_grad_table(self):
		for layer in self.layers: layer.init_grad_table()

	# Calculate loss
	def get_loss_vector(self, X, Y, batch_size): 
		self.loss.get_vector(self.forward(X), Y, batch_size)

	# Simply make grad the value of the input node
	def forward_pass(self, x):
		vec_now = x
		for layer in self.layers:
			for v, node in zip(x[0], layer.nodes):
				if node.type == 'regular':
					node.value = v
					node.forward_pass()
				elif node.type == 'bias':
					node.value = 1
					node.forward_pass()
			if layer.layer_type == 'Activation':
				layer.diff(vec_now)
			vec_now = layer.forward(vec_now)

	def backward_pass(self, y_pred, y_true):
		self.loss.get_pCpy(y_pred, y_true)

	# Update Weights using Back Propagation
	def update(self, X_, Y_, batch_size=0, trainig_epoch=10):
		if not self.compiled: raise RuntimeError('Model Not Compiled.')
		for epoch in range(trainig_epoch):
			idx = np.random.choice(np.arange(len(X_)), batch_size, replace=False)
			X, Y = X_[idx], Y_[idx]
			#input((X, Y))
			self.init_grad_table()
			loss_vec = self.get_loss_vector(X, Y, batch_size)
			for x, y in zip(X, Y):
				x = np.reshape(x, (1, x.shape[0]))
				y = np.reshape(y, (1, y.shape[0]))
				# X[data], Y[data]
				self.forward_pass(x)
				self.backward_pass(self.forward(x), y)

	# Make sure witch optimizer and loss to use.
	def compile(self, optimizer=None, loss=None, lr=0.01):
		self.compiled = True
		self.n_input = self.layers[0].n_input
		self.n_output = self.layers[-1].n_output
		self.optimizer = optimizer
		self.lr = lr
		self.loss = Model_loss(loss_func=loss)
		

	def __str__(self):
		ret = '<Class Model> Input Shape = %s, Output Shape = %s\n' % (self.input_shape, self.output_shape)
		for i in self.layers:
			ret += i.__str__()
		return ret

if __name__ == '__main__':
	from layers import *
	from activations import *
	np.random.seed(1)
	X = np.random.normal(0, 1.2, (3,5))
	#print('X=',X)
	Y = np.array([[1,2,3],[2,4,6],[3,6,9]])
	a = Input(n_input=5, n_output=3)
	a = Dense(5, a, kernel_initializer='Gaus', kernel_mean=1, kernel_std=0.1, bias_initializer='Ones')
	a = Relu(a)
	a = Dense(3, a, kernel_initializer='Gaus', kernel_mean=0, kernel_std=0.1, bias_initializer='Ones')
	a = Softmax(a)
	model = Model(a)
	##print(model)
	print('model.forward(X)=\n', model.forward(X))
	#print('Y=', Y)
	model.compile(optimizer=None, loss='cross_entropy')
	model.update(X, Y, batch_size=1)
	print(model)
