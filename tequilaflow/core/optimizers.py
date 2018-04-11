from __future__ import absolute_import
import numpy as np

class Model_optimizer:
	def __init__(self, **kwargs):
		self.dict = kwargs
		if kwargs['type'] == 'SGD':
			self.opt = SGD(**kwargs)
		else: raise ValueError('Optimizer %s not recognized'%kwargs['type'])

	def optimize(self, origin, err):
		return self.opt.kernel(origin, err)

	def get_lr(self):
		return self.opt.lr

	def __str__(self):
		return self.opt.__str__(**self.dict)

class optimizer:
	def __init__(self, **kwargs):
		if 'type' not in kwargs.keys():
			kwargs['type'] = 'SGD'
		self.type = kwargs['type']

	def __str__(self, **kwargs):
		return "<class optimizer> %s" % kwargs

class SGD(optimizer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		if 'lr' not in kwargs.keys(): self.lr = 0.01
		else: self.lr = kwargs['lr']
		if 'decay_rate' not in kwargs.keys():self.decay_rate=1
		else: self.decay_rate=kwargs['decay_rate']

	def kernel(self, origin, err):
		ret = origin - self.lr * err
		self.lr *= self.decay_rate
		return ret 

	def __str__(self, **kwargs):
		return super().__str__(**kwargs)


if __name__ == '__main__':
	opt = Model_optimizer(type='SGD', lr=0.01, showay='cool')
	print(opt)
