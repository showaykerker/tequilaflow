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
		self.loss_func.get_vector(Y_predict, Y_true, batch_size)

	def get_partial(self,):
		self.loss_func.get_partial()

	def __str__(self):
		return '<class Model_loss> loss_func=%s' % (self.l)


class loss_function(object):
	def __init__(self, ):
		self.partial_output = None
		self.batch_size = None

	def get_vector(self, Y_predict, Y_true, batch_size=None):
		if Y_predict.shape != Y_true.shape : raise ValueError('Y shape mismatch. %s and %s'%(str(Y_predict.shape), str(Y_true.shape)))
		

	def get_partial(self, ):
		raise NotImplementError('')


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