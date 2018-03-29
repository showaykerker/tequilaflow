

from core import *
from layers import *

class activation(layer):
	def __init__(self, inp_layer, n_output):
		super().__init__(n_nodes=n_output, layer_type='activation')
		self.initialize(n_output=n_output)
		


if __name__=='__main__':
	inp = np.array([[4,5]])
	x = Input(n_input=2, n_output=18)
	x = Dense(x, 48)
