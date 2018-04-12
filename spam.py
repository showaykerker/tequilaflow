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
import mpl_toolkits.mplot3d 
import copy

def get_data():
	data = loadmat('spam_data.mat')
	#print(data)
	#print(data.keys())
	X_train, Y_train = data['train_x'], data['train_y']
	X_test, Y_test = data['test_x'], data['test_y']
	return X_train, Y_train, X_test, Y_test

def get_model():
	a = Input(n_input=40, n_output=32, kernel_initializer='Gaus', kernel_mean=0.1, kernel_std=0.5, bias_initializer='Ones')
	a = Linear(a)
	a = Dense(16, a, kernel_initializer='Gaus', kernel_mean=0., kernel_std=0.3, bias_initializer='Ones')
	a = Tanh(a)
	a = Dense(8, a, kernel_initializer='Gaus', kernel_mean=0., kernel_std=0.3, bias_initializer='Ones')
	a = Linear(a)
	a = Dense(8, a, kernel_initializer='Gaus', kernel_mean=0., kernel_std=0.3, bias_initializer='Ones')
	a = Linear(a)
	a = Dense(4, a, kernel_initializer='Gaus', kernel_mean=0., kernel_std=0.3, bias_initializer='Ones')
	a = Linear(a)
	a = Dense(2, a, kernel_initializer='Gaus', kernel_mean=0., kernel_std=0.3, bias_initializer='Ones')
	a = Linear(a)
	a = Softmax(a)
	model = Model(a)
	model.compile(optimizer='SGD', lr=0.0003, decay_rate=1 , loss='cross_entropy')
	return model

if __name__ == '__main__':
	X_train, Y_train, X_test, Y_test = get_data()
	X_train += 0.1
	X_test += 0.1
	model = get_model()
	#print(model)
	#input()
	#input(X_test)

	hist = model.update(X_train, Y_train, batch_size=16, trainig_epoch=2400000, 
				X_val=X_test, Y_val=Y_test, validate_every_n_epoch=25, record_every_n_epoch=5)
	try:

		latent10 = model.latent_monitor[10]
		latentL = model.latent_monitor[-1]
		C1_10, C2_10 = [], []
		C1_L, C2_L = [], []
		for i,v in enumerate(latent10['class']):
			if v == 0 : C1_10.append(latent10['val'][i])
			else: C2_10.append(latent10['val'][i])

		for i,v in enumerate(latentL['class']):
			if v == 0 : C1_L.append(latentL['val'][i])
			else: C2_L.append(latentL['val'][i])

	except:
		pass

	print(model.predict(X_test[:10]))
	input(Y_test[:10])
	Y_test_pred = hist['best'].predict(X_test).argmax(axis=1)
	Y_train_pred = hist['best'].predict(X_train).argmax(axis=1)
	Y_train = Y_train.argmax(axis=1)
	Y_test = Y_test.argmax(axis=1)
	#input(Y_test_pred)
	#input(Y_train_pred)

	print('Lowest Loss:', hist['best_loss'])

	fig = plt.figure()
	
	ax0 = plt.subplot2grid((2, 3), (0, 0))
	ax1 = plt.subplot2grid((2, 3), (0, 1))
	ax2 = plt.subplot2grid((2, 3), (0, 2))
	ax3 = plt.subplot2grid((2, 3), (1, 0))
	ax4 = None

	#fig, ax = plt.subplots(2,2)
	ax0.plot(hist['loss'], color='green', linewidth=1)
	ax0.set_title('Training loss')
	ax1.plot([1-i for i in hist['train_acc']], color='red', linewidth=1)
	ax1.set_title('Training error rate')
	ax2.plot([1-i for i in hist['acc']], color='red', linewidth=1)
	ax2.set_title('Testing error rate')

	



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