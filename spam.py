from tequilaflow.core.core import *
from tequilaflow.core.layers import *
from tequilaflow.core.activations import *
from tequilaflow.core.loss import *
import numpy as np
from scipy.io import loadmat
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

D_3 = False

def get_data():
	data = loadmat('spam_data.mat')
	#print(data)
	#print(data.keys())
	
	X_train, Y_train = data['train_x'], data['train_y']
	X_test, Y_test = data['test_x'], data['test_y']
	randomize = np.arange(len(X_train))
	np.random.shuffle(randomize)
	X_train = X_train[randomize]
	Y_train = Y_train[randomize]

	return X_train, Y_train, X_test, Y_test

def get_model():
	a = Input(n_input=40, n_output=36, kernel_initializer='Gaus', kernel_mean=0., kernel_std=0.45, bias_initializer='Ones')
	a = Linear(a)
	a = Dense(32, a, kernel_initializer='Gaus', kernel_mean=0., kernel_std=0.1, bias_initializer='Ones')
	a = Linear(a)
	a = Dense(2, a, kernel_initializer='Gaus', kernel_mean=0., kernel_std=0.1, bias_initializer='Ones' )
	a = Linear(a)
	a = Dense(2, a, kernel_initializer='Gaus', kernel_mean=0., kernel_std=0.1, bias_initializer='Ones')
	a = Linear(a)
	a = Softmax(a)
	model = Model(a)
	model.compile(optimizer='SGD', lr=0.001, decay_rate=0.99998 , loss='cross_entropy')
	return model

if __name__ == '__main__':
	X_train, Y_train, X_test, Y_test = get_data()
	X_train += 0.1
	X_test += 0.1
	model = get_model()


	hist = model.update(X_train, Y_train, batch_size=32, trainig_epoch=3000, 
				X_val=X_test, Y_val=Y_test, validate_every_n_epoch=50, record_every_n_epoch=5, latenet=2)

	C1_10, C2_10 = [[],[],[]], [[],[],[]]
	C1_m, C2_m = [[],[],[]], [[],[],[]]
	C1_L, C2_L = [[],[],[]], [[],[],[]]
	#try:

	last = len(hist['acc'])

	mid = int((last-2-10)/2)
	latent10 = model.latent_monitor[10]
	latentm = model.latent_monitor[mid]
	latentL = model.latent_monitor[last-2]



	for i,v in enumerate(latent10['class']):
		if v == 0 : 
			if D_3: x, y, z = latent10['val'][i]
			else: x, y = latent10['val'][i]
			C1_10[0].append(x)
			C1_10[1].append(y)
			if D_3: C1_10[2].append(z)
		else: 
			if D_3: x, y, z = latent10['val'][i]
			else: x, y = latent10['val'][i]
			C2_10[0].append(x)
			C2_10[1].append(y)
			if D_3: C2_10[2].append(z)

	for i,v in enumerate(latentm['class']):
		if v == 0 : 
			if D_3: x, y, z = latentm['val'][i]
			else: x, y = latentm['val'][i]
			C1_m[0].append(x)
			C1_m[1].append(y)
			if D_3: C1_m[2].append(z)
		else: 
			if D_3: x, y, z = latentm['val'][i]
			else: x, y = latentm['val'][i]
			C2_m[0].append(x)
			C2_m[1].append(y)
			if D_3: C2_m[2].append(z)

	for i,v in enumerate(latentL['class']):
		if v == 0 : 
			if D_3: x, y, z = latentL['val'][i]
			else: x, y = latentL['val'][i]
			C1_L[0].append(x)
			C1_L[1].append(y)
			if D_3: C1_L[2].append(z)
		else: 
			if D_3: x, y, z = latentL['val'][i]
			else: x, y = latentL['val'][i]
			C2_L[0].append(x)
			C2_L[1].append(y)
			if D_3: C1_L[2].append(z)


	print(model.predict(X_test[:10]))
	input(Y_test[:10])

	Y_test_pred = hist['best'].predict(X_test).argmax(axis=1)
	Y_train_pred = hist['best'].predict(X_train).argmax(axis=1)
	Y_train = Y_train.argmax(axis=1)
	Y_test = Y_test.argmax(axis=1)


	print('Lowest Loss:', hist['best_loss'])

	fig = plt.figure()
	
	ax0 = plt.subplot2grid((2, 3), (0, 0))
	ax1 = plt.subplot2grid((2, 3), (0, 1))
	ax2 = plt.subplot2grid((2, 3), (0, 2))
	

	#fig, ax = plt.subplots(2,2)
	ax0.plot(hist['loss'], color='green', linewidth=1)
	ax0.set_title('Training loss')
	ax1.plot([1-i for i in hist['train_acc']], color='red', linewidth=1)
	ax1.set_title('Training error rate')
	ax2.plot([1-i for i in hist['acc']], color='red', linewidth=1)
	ax2.set_title('Testing error rate')


	if D_3 == True:

		ax3 = plt.subplot2grid((2, 3), (1, 0), projection='3d')
		ax4 = plt.subplot2grid((2, 3), (1, 1), projection='3d')
		ax5 = plt.subplot2grid((2, 3), (1, 2), projection='3d')

		ax3.scatter(x=C1_10[0], y=C1_10[1], z=C1_10[2], c='red', label='Class 1', alpha=0.6, edgecolors='white')
		ax3.scatter(x=C2_10[0], y=C2_10[1], z=C1_10[2], c='green', label='Class 2', alpha=0.6, edgecolors='white')
		ax3.set_title('10 Epoch')
		ax3.legend()

		ax4.scatter(x=C1_m[0], y=C1_m[1], z=C1_m[2], c='red', label='Class 1', alpha=0.6, edgecolors='white')
		ax4.scatter(x=C2_m[0], y=C2_m[1], z=C1_m[2], c='green', label='Class 2', alpha=0.6, edgecolors='white')
		ax4.set_title('%d Epoch'%mid)
		ax4.legend()

		ax5.scatter(x=C1_L[0], y=C1_L[1], z=C1_L[2], c='red', label='Class 1', alpha=0.6, edgecolors='white')
		ax5.scatter(x=C2_L[0], y=C2_L[1], z=C1_L[2], c='green', label='Class 2', alpha=0.6, edgecolors='white')
		ax5.set_title('%d Epoch'%(len(hist['loss'])*5))
		ax5.legend()

	else:
		ax3 = plt.subplot2grid((2, 3), (1, 0))
		ax4 = plt.subplot2grid((2, 3), (1, 1))
		ax5 = plt.subplot2grid((2, 3), (1, 2))

		ax3.scatter(x=C1_10[0], y=C1_10[1], c='red', label='Class 1', alpha=0.6, edgecolors='white')
		ax3.scatter(x=C2_10[0], y=C2_10[1], c='green', label='Class 2', alpha=0.6, edgecolors='white')
		ax3.set_title('10 Epoch')
		ax3.legend()

		ax4.scatter(x=C1_m[0], y=C1_m[1], c='red', label='Class 1', alpha=0.6, edgecolors='white')
		ax4.scatter(x=C2_m[0], y=C2_m[1], c='green', label='Class 2', alpha=0.6, edgecolors='white')
		ax4.set_title('%d Epoch'%mid)
		ax4.legend()

		ax5.scatter(x=C1_L[0], y=C1_L[1], c='red', label='Class 1', alpha=0.6, edgecolors='white')
		ax5.scatter(x=C2_L[0], y=C2_L[1], c='green', label='Class 2', alpha=0.6, edgecolors='white')
		ax5.set_title('%d Epoch'%(len(hist['loss'])*5))
		ax5.legend()

	'''
	ax[1][0].plot(Y_train.flatten(), 's', color='blue', label='label', marker='x', markersize=0.8)
	ax[1][0].plot(Y_train_pred.flatten(),'s', color='orange', label='predict', marker='x', markersize=0.5)
	ax[1][0].set_title('Heat load for training dataset')
	ax[1][0].legend(loc='upper left')
	ax[1][1].plot(Y_test.flatten(), 's', color='blue', label='label', marker='x', markersize=0.8)
	ax[1][1].plot(Y_test_pred.flatten(), 's', color='orange', label='predict', marker='x', markersize=0.5)
	ax[1][1].set_title('Heat load for testing dataset')
	ax[1][1].legend(loc='upper left')
	'''
	plt.show()