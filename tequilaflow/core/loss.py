from __future__ import absolute_import
import numpy as np
import copy


class Model_loss: # wrapper
	def __init__(self, loss_func='rms'):
		self.l = loss_func
		if   loss_func == 'rms': self.loss_func = root_mean_square_error()
		elif loss_func == 'mse': self.loss_func = mean_square_error()
		elif loss_func == 'se' : self.loss_func = square_error()
		elif loss_func == 'cross_entropy': self.loss_func = cross_entropy()

	def set_batch_size(self, b_s):
		self.loss_func.batch_size = b_s

	def get_vector(self, Y_predict, Y_true, batch_size=None):
		return self.loss_func.get_vector(Y_predict, Y_true, batch_size)

	def get_pCpy(self, y_pred, y_true, idx):
		return self.loss_func.get_pCpy(y_pred, y_true, idx)

	def get_performance(self, Y_pred, Y_true):
		return self.loss_func.get_performance(Y_pred, Y_true)

	def __str__(self):
		return '<class Model_loss> loss_func=%s' % (self.l)


class loss_function(object): # for inherit
	def __init__(self, ):
		self.partial_output = None
		self.batch_size = None

	def get_vector(self, Y_predict, Y_true, batch_size=None):
		if Y_predict.shape != Y_true.shape : raise ValueError('Y shape mismatch. %s and %s'%(str(Y_predict.shape), str(Y_true.shape)))
		if batch_size == None : 
			self.batch_size = Y_predict.shape[0] 
			#raise ValueError('batch_size is None')
		elif batch_size is not Y_predict.shape[0]:
			raise ValueError('batch_size error')
		#else: self.batch_size = batch_size

	def get_pCpy(self, Y_predict, Y_true, get_v, idx):
		d = 1e-12
		Y_pred_p = copy.deepcopy(Y_predict)
		Y_pred_m = copy.deepcopy(Y_predict)
		Y_pred_p[idx] += d
		Y_pred_m[idx] -= d
		plus = get_v(Y_pred_p, Y_true, 1, pass_=True)
		minus = get_v(Y_pred_m, Y_true, 1, pass_=True)
		return np.expand_dims( (plus-minus)/(2*d) , axis=0 )
	
	def get_performance(self, Y_pred, Y_true):
		raise NotImplementedError('get_acc not implemented')


class root_mean_square_error(loss_function):
	def __init__(self):
		super().__init__()

	def get_vector(self, Y_predict, Y_true, batch_size=None, pass_=False):
		if not pass_: super().get_vector(Y_predict, Y_true, batch_size)
		s_loss = (Y_predict - Y_true)**2
		rm_loss = (np.sum(s_loss, axis=0)/self.batch_size)**0.5
		tot_loss = np.expand_dims(rm_loss, axis=0)
		return tot_loss

	def get_pCpy(self, Y_predict, Y_true, idx=None):
		ret = super().get_pCpy(Y_predict, Y_true, get_v=self.get_vector, idx=idx)
		ret = ret.flatten()
		ret = np.expand_dims(ret, axis=0)
		return ret

	def get_performance(self, Y_pred, Y_true):
		ret = self.get_vector(Y_pred, Y_true, batch_size=None)
		return ret.mean()


class mean_square_error(loss_function):
	def __init__(self):
		super().__init__()

	def get_vector(self, Y_predict, Y_true, batch_size=None, pass_=False):
		if not pass_: super().get_vector(Y_predict, Y_true, batch_size)
		loss = (Y_predict - Y_true)**2
		tot_loss = np.expand_dims(np.sum(loss, axis=0)/self.batch_size, axis=0)

		return tot_loss

	def get_pCpy(self, Y_predict, Y_true, idx=None):
		return np.expand_dims(Y_predict[idx]-Y_true[idx], axis=0)

	def get_performance(self, Y_pred, Y_true):
		ret = self.get_vector(Y_pred, Y_true, batch_size=None)
		return ret.mean()

class square_error(loss_function):
	def __init__(self):
		super().__init__()

	def get_vector(self, Y_predict, Y_true, batch_size=None, pass_=False):
		if not pass_: super().get_vector(Y_predict, Y_true, batch_size)
		
		#print(Y_predict)
		#print(Y_true)
		'''
		[[0. 0. 0.]
		 [0. 0. 0.]
		 [0. 0. 0.]
		 [0. 0. 0.]]
		[[0.1362 0.1362 0.1362]
		 [0.1362 0.1362 0.1362]
		 [0.1362 0.1362 0.1362]
		 [0.1362 0.1362 0.1362]]
		'''
		loss = (Y_predict - Y_true)**2
		#print(loss)
		'''
		[[0.0171089  0.0171089  0.0171089 ]
		 [0.01710898 0.01710898 0.01710898]
		 [0.01710898 0.01710898 0.01710898]
		 [0.01710898 0.01710898 0.01710898]]
		'''
		tot_loss = np.expand_dims(np.sum(loss, axis=0), axis=0)
		#print(tot_loss)
		#input()
		
		'''
		[[0.0684356 0.0684356 0.0684356]]
		'''

		return tot_loss

	def get_pCpy(self, Y_predict, Y_true, idx=None):
		#input( np.expand_dims((2*Y_predict-2*Y_true)[idx], axis=0))
		return np.expand_dims((2*Y_predict-2*Y_true)[idx], axis=0)

	def get_performance(self, Y_pred, Y_true):
		ret = self.get_vector(Y_pred, Y_true, batch_size=None)
		return ret.mean()

class cross_entropy(loss_function):
	def __init__(self):
		super().__init__()

	def get_vector(self, Y_predict, Y_true, batch_size=None, pass_=False):
		nclass = Y_predict.shape[1]
		E = - (Y_true * np.log(Y_predict+1e-12)).sum(axis=1)
		E = np.expand_dims(E, axis=1)
		E = np.append(E, E, axis=1)
		return E

	def get_pCpy(self, Y_predict, Y_true, idx=None):
		ret = np.expand_dims( ((Y_predict-Y_true)/( (Y_predict+1e-12)*(1-Y_predict+1e-12)))[idx], axis=0)
		#print(Y_predict[idx])
		#print(Y_true[idx])
		#input(ret)
		return ret

	def get_performance(self, Y_pred, Y_true):
		ret = self.get_vector(Y_pred, Y_true, batch_size=None)
		#print(ret)
		#input(ret.mean())
		return ret.mean()