

from core import *
from layers import *

class activation(layer):
	def __init__(self, inp_layer, n_output):
		super().__init__(n_nodes=n_output, layer_type='activation')
		self.initialize(n_output=n_output)

	def activate(self, vec):
		raise NotImplementError()

class relu(activation):
	def __init__(self, inp_layer, n_output):
		super().__init__(inp_layer=inp_layer, n_output=n_output);
		self.inp_layer = inp_layer
		self.n_output = n_output

	def activate(self, vec):
		vec_length = vec.shape[1]
		print(vec_length)
		if vec_length != self.n_output:
			raise ValueError('vec_length != self.n_output')

if __name__=='__main__':
	inp = np.array([[4,5]])
	print(inp.shape)
	x = Input(n_input=2, n_output=3)
	x = Dense(x, 1)
	x = Output(x)
	model = Model(x)
	model.forward(inp)

