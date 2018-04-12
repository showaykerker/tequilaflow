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
	a = Input(n_input=40, n_output=36)
	a = Tanh(a)
	a = Dense(18, a)
	a = Linear(a)
	a = Dense(12, a)
	a = Tanh(a)
	a = Dense(4, a)
	a = Tanh(a)
	a = Dense(2, a)
	a = Linear(a)
	a = Softmax(a)
	model = Model(a)
	model.compile(optimizer='SGD', lr=0.03, decay_rate=1 , loss='cross_entropy')
	return model

if __name__ == '__main__':
	X_train, Y_train, X_test, Y_test = get_data()
	X_train += 0.1
	X_test += 0.1
	model = get_model()
	#print(model)
	#input()
	#input(X_test)

	hist = model.update(X_train, Y_train, batch_size=8, trainig_epoch=2400000, 
				X_val=X_test, Y_val=Y_test, validate_every_n_epoch=500, record_every_n_epoch=20)

	latent10 = model.latent_monitor[10]
	latent500 = model.latent_monitor[500]
