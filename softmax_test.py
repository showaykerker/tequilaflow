import numpy as np

def softmax(X):
	print(X.shape)
	X_ret = np.zeros(X.shape)
	for i,row in enumerate(X):
		exps = np.exp(row-row.max())
		X_ret[i] = exps/np.sum(exps)
	return X_ret

a = np.array([[1,6,3,9,5]])
print(softmax(a))

b = np.array([[1,6,3,9,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8]])
print(softmax(b))