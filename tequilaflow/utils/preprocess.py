from __future__ import absolute_import
import numpy as np
import copy

def to_catagorize(Y, n = None):
	label = {}
	for i,v in enumerate(np.unique(Y)): label[v] = i
	if len(label) != n: raise ValueError('len(label) != n')
	ret = []

	for y in Y:
		vec = [0] * n
		vec[label[y[0]]] = 1
		ret.append(copy.deepcopy(vec))

	ret = np.reshape(ret, (1, len(Y), n))
	return ret, label

def reverse_catagorize(ret, label):
	res = dict((v,k) for k,v in label.items())
	Y = []
	for i in ret[0]:
		y = res[np.argmax(i)]
		Y.append([y])
	

	return Y


if __name__ == '__main__':
	a = np.array([['apple'], ['cat'], ['banana'], ['showay'], ['fish']])
	a, label = to_catagorize(a, 5)
	print(a)
	a = reverse_catagorize(a, label)
	print(a)