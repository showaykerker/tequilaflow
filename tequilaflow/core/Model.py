class Model:
	def __init__(self, inp_layer):
		self.matrics = copy.deepcopy(inp_layer.matrics)
		#self.whole = inp_layer.whole
		#self.n_input = len(self.matrics[0])-1
		#self.n_output = len(self.whole)
		#print('Modle.n_input=%d'%(self.n_input))

	def forward(self, X):
		X_ = np.append(X, [[1]], axis=1)
		current = X_
		for i in self.matrics:
			if (i.get_layer_type()=='activation'):
				
			else:
				current = X_.dot(i)

		return X_.dot(self.whole)
