from __future__ import absolute_import
import numpy as np

class DNN:
	def __init__(self, n_input):
		self.structure = []
		self.structure.append(np.ones([n_input, n_input]))
		self.last_output = n_input
		self.n_input = n_input
		self.matrics = self.structure[-1]

	def add_Dense(self, n_output, n_input=None):
		if n_input is None: n_input = self.last_output
		elif n_input != self.last_output: 
			raise ValueError('n_input != self.last_output ( %d! = %d )'%(n_input, self.last_output))
		new = np.random.normal(0, 0.01, (n_input, n_output))
		self.structure.append(np.random.normal(0, 0.01, (n_input, n_output)))
		self.matrics = self.matrics.dot(self.structure[-1])
		self.last_output = n_output

	def forward(self, X):
		if X.shape != (1, self.n_input): raise ValueError('X.shape', X.shape, ' is not fit.')
		return X.dot(self.matrics)

	def __str__(self):
		info = {}
		info['structure'] = [mat.shape for mat in self.structure[1:]]
		info['matrics_shape'] = self.matrics.shape
		return str(info)

if __name__ == '__main__':
	model = DNN(3)
	model.add_Dense(5)
	model.add_Dense(5)
	print(model)
	input_ = np.array([1,2,3]).reshape(1,3)
	print(model.forward(input_))

