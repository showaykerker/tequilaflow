from __future__ import absolute_import
import numpy as np


class Model_loss:
	def __init__(self, loss_func='rms'):
		self.l = loss_func
		if   loss_func == 'rms': self.loss_func = root_mean_square_error()
		elif loss_func == 'mse': self.loss_func = mean_square_error()
		elif loss_func == 'se' : self.loss_func = square_error()
		elif loss_func == 'cross_entropy': self.loss_func = cross_entropy()

	def get_vector(self, Y_predict, Y_true, batch_size=None):
		return self.loss_func.get_vector(Y_predict, Y_true, batch_size)

	def get_pCpy(self, y_pred, y_true):
		return self.loss_func.get_pCpy(y_pred, y_true)

	def __str__(self):
		return '<class Model_loss> loss_func=%s' % (self.l)


class loss_function(object):
	def __init__(self, ):
		self.partial_output = None
		self.batch_size = None

	def get_vector(self, Y_predict, Y_true, batch_size=None):
		if Y_predict.shape != Y_true.shape : raise ValueError('Y shape mismatch. %s and %s'%(str(Y_predict.shape), str(Y_true.shape)))
		

	def get_pCpy(self, y_pred, y_true, get_v):
		d = 1e-12
		plus = get_v(y_pred + d, y_true, 1)
		minus = get_v(y_pred - d, y_true, 1)
		return np.expand_dims( (plus-minus)/(2*d) , axis=0 )
		


class root_mean_square_error(loss_function):
	def __init__(self):
		super().__init__()

	def get_vector(self, Y_predict, Y_true, batch_size=None):
		tot_loss = 0
		super().get_vector(Y_predict, Y_true, batch_size)
		if batch_size == None : raise ValueError('batch_size is None')
		else: self.batch_size = batch_size

		for y_p, y_t in zip(Y_predict, Y_true):
			err = y_p - y_t
			tot_loss += err ** 2

		tot_loss = (tot_loss/self.batch_size)**0.5
		return tot_loss

	def get_pCpy(self, y_pred, y_true):
		return super().get_pCpy(y_pred, y_true, self.get_vector)


class mean_square_error(loss_function):
	def __init__(self):
		super().__init__()

	def get_vector(self, Y_predict, Y_true, batch_size=None):
		tot_loss = 0
		super().get_vector(Y_predict, Y_true, batch_size)
		if batch_size == None : raise ValueError('batch_size is None')
		else: self.batch_size = batch_size

		for y_p, y_t in zip(Y_predict, Y_true):
			err = y_p - y_t
			tot_loss += err ** 2

		tot_loss = (tot_loss/batch_size)

		return tot_loss

	def get_pCpy(self, y_pred, y_true):
		return np.expand_dims(y_pred-y_true, axis=0)


class square_error(loss_function):
	def __init__(self):
		super().__init__()

	def get_vector(self, Y_predict, Y_true, batch_size=None):
		tot_loss = 0
		super().get_vector(Y_predict, Y_true, batch_size)
		if batch_size == None : raise ValueError('batch_size is None')
		else: self.batch_size = batch_size

		for y_p, y_t in zip(Y_predict, Y_true):
			err = y_p - y_t
			tot_loss += err ** 2

		return tot_loss

	def get_pCpy(self, y_pred, y_true):
		return 2*y_pred-2*y_true



class cross_entropy(loss_function):
	def __init__(self):
		super().__init__()

	def get_vector(self, Y_predict, Y_true, batch_size=None):
		tot_loss = 0
		super().get_vector(Y_predict, Y_true, batch_size)
		if batch_size == None : raise ValueError('batch_size is None')
		else: self.batch_size = batch_size

		for y_p, y_t in zip(Y_predict, Y_true):
			err = y_p - y_t
			inside_ = 0 
			for k in range(len(y_t)): 
				inside_ += y_t * np.log(y_p) 
			tot_loss += inside_ 
		return tot_loss

	def get_pCpy(self, y_pred, y_true):
		return super().get_pCpy(y_pred, y_true, self.get_vector)