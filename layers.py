from core import *
import copy



class Input(layer):
	def __init__(self, n_input, n_output):
		super().__init__(n_nodes=n_input, layer_type='Input')
		self.n_input = n_input
		self.initialize(n_output)
		self.matrics = [self.matrix]
		self.whole = np.append(self.matrix, np.zeros((self.matrix.shape[0],1)), axis=1)
		print(self)

	def __str__(self):
		return '<Class Input> n_nodes=%d, n_output=%d\n' % (self.n_nodes, self.n_output) + str(self.matrix)


class Dense(layer):
	def __init__(self, inp_layer, n_output, n_input=None):
		super().__init__(n_nodes=inp_layer.n_output, layer_type='Dense')
		self.initialize(n_output)
		self.matrics = inp_layer.matrics + [self.matrix]
		self.whole = inp_layer.whole.dot(self.matrix)
		self.whole = np.append(self.whole, np.zeros((self.whole.shape[0],1)), axis=1 )
		print(self)

	def __str__(self):
		return '<Class Dense> n_nodes=%d, n_output=%d\n' % (self.n_nodes, self.n_output) + str(self.matrix)

class Output(layer):
	def __init__(self, inp_layer):
		super().__init__(n_nodes=inp_layer.n_output, layer_type='Output')
		self.initialize(inp_layer.n_output)
		if inp_layer.get_layer_type() == 'Dense':
			self.matrics = copy.deepcopy(inp_layer.matrics)
			self.whole = np.delete(inp_layer.whole, (-1), axis=1)
		print(self)

	def __str__(self):
		return '<Class Output> n_nodes=%d, n_output=%d\n' % (self.n_nodes, self.n_output) 

class Model:
	def __init__(self, inp_layer):
		self.matrics = copy.deepcopy(inp_layer.matrics)
		self.whole = inp_layer.whole
		#self.n_input = len(self.matrics[0])-1
		#self.n_output = len(self.whole)
		#print('Modle.n_input=%d'%(self.n_input))

	def forward(self, X):
		X_ = np.append(X, [[1]], axis=1)
		return X_.dot(self.whole)




if __name__ == '__main__':
	inp = np.array([[4,5]])
	print('Input Shape:', inp.shape)
	x = Input(n_input=2, n_output=18)
	x = Dense(x, 12)
	x = Dense(x, 48)
	x = Dense(x, 22)
	x = Dense(x, 15)
	x = Dense(x, 15)
	x = Output(x)

	model = Model(x)
	
	out = model.forward(inp)
	
	print('Output Shape:', out.shape)
	print('Output:', out)