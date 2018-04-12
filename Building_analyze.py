from tequilaflow.core.core import *
from tequilaflow.core.layers import *
from tequilaflow.core.activations import *
from tequilaflow.core.loss import *
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
import copy

cname = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 
		'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution']

one_hot_columns = ['Glazing Area Distribution', 'Orientation']

remove_c = ['Wall Area', 'Orientation', 'Glazing Area', 'Glazing Area Distribution']

leaving = list(set(cname)-set(remove_c))

input_shape = len(leaving)

def preprocess(df, H):
	print('Start Preprocessing...')
	cname = list(df)
	X = df.as_matrix()
	Y = H.as_matrix()

	Y_min_max_scaler = preprocessing.MinMaxScaler()
	Y_minmax = Y_min_max_scaler.fit_transform(Y)
	#Y = Y_min_max_scaler.inverse_transform(Y_minmax)
	scalars = {'Y':Y_min_max_scaler}
	X_T = X.T
	for i,x in enumerate(X_T):
		
		scalar = preprocessing.MinMaxScaler()
		x_min, x_max = x.min(), x.max()
		X_T[i] = (x-x_min)/(x_max-x_min)	
		scalars[i] = (x_min, x_max)
	#print(scalars)
	X = X_T.T
	#print(X.shape, Y.shape, 'In preprocess')
	return X, Y, scalars

def get_data():
	print('Start to get data...')
	data = pd.read_csv('energy_efficiency_data.csv')
	df = copy.deepcopy(data)
	df = df.rename(columns={'# Relative Compactness':'Relative Compactness'})
	df = df.sample(frac=1).reset_index(drop=True)

	H = df['Heating Load'].to_frame()
	C = df['Cooling Load'].to_frame()
	del df['Heating Load']
	del df['Cooling Load']

	for c in remove_c: del df[c]

	for one_hot_c in one_hot_columns:
		if one_hot_c in leaving:
			df = pd.get_dummies(data=df, columns=one_hot_c)

	#df = pd.concat([df, back], axis=1)
	X, Y, scalars = preprocess(df, H)
	#print(X.shape, Y.shape, 'In get_data')
	return X, Y, scalars

def get_model(input_size):
	print('Getting Model.')
	a = Input(n_input=input_size, n_output=6)
	a = Linear(a)
	a = Dense(8, a)
	a = Linear(a)
	a = Dense(4, a)
	a = Sigmoid(a)
	a = Dense(2, a)
	a = Sigmoid(a)
	a = Dense(2, a)
	a = Linear(a)
	a = Dense(2, a)
	a = Linear(a)
	a = Dense(1, a, kernel_initializer='Gaus', kernel_mean=1.2, kernel_std=0.3, bias_initializer='Ones')
	a = Output(a)

	model = Model(a)
	model.compile(optimizer='SGD', lr=0.001, decay_rate=0.9999996 , loss='se', estimator='rms')
	#print(model)
	return model

def main():
	X, Y, scalars = get_data()
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	X = X[randomize]
	Y = Y[randomize]
	
	X_train, Y_train = X[:576], Y[:576]
	X_test,  Y_test  = X[576:], Y[576:]
	model = get_model(input_shape)

	print('Start Training.')
	hist = model.update(X_train, Y_train, batch_size=4, trainig_epoch=240000, 
				X_val=X_test, Y_val=Y_test, validate_every_n_epoch=16000, record_every_n_epoch=100)


	'''
	Y_test_pred = scalars['Y'].inverse_transform(model.predict(X_test))
	Y_train_pred = scalars['Y'].inverse_transform(model.predict(X_train))
	Y_train = scalars['Y'].inverse_transform(Y_train)
	Y_test = scalars['Y'].inverse_transform(Y_test)
	'''
	Y_test_pred = hist['best'].predict(X_test)
	Y_train_pred = hist['best'].predict(X_train)
	Y_train = Y_train
	Y_test = Y_test
	#input(Y_test_pred)
	#input(Y_train_pred)

	print('Lowest Loss:', hist['best_loss'])

	fig = plt.figure()
	ax = [[[],[]],[[],[]]]
	ax[0][0] = plt.subplot2grid((3, 4), (0, 0), colspan=2)
	ax[0][1] = plt.subplot2grid((3, 4), (0, 2), colspan=2)
	ax[1][0] = plt.subplot2grid((3, 4), (1, 0), colspan=4, rowspan=1)
	ax[1][1] = plt.subplot2grid((3, 4), (2, 0), colspan=4, rowspan=1)

	#fig, ax = plt.subplots(2,2)
	ax[0][0].plot(hist['acc'], color='red', linewidth=1)
	ax[0][0].set_title('acc')
	ax[0][1].plot(hist['loss'], color='green', linewidth=1)
	ax[0][1].set_title('loss')
	ax[1][0].plot(Y_train.flatten(), color='blue', label='label', linewidth=1.2)
	ax[1][0].plot(Y_train_pred.flatten(), color='orange', label='predict', linewidth=.6)
	ax[1][0].set_title('Heat load for training dataset')
	ax[1][0].legend(loc='upper left')
	ax[1][1].plot(Y_test.flatten(), color='blue', label='label', linewidth=1.5)
	ax[1][1].plot(Y_test_pred.flatten(), color='orange', label='predict', linewidth=1)
	ax[1][1].set_title('Heat load for testing dataset')
	ax[1][1].legend(loc='upper left')
	plt.show()

if __name__ == '__main__':
	main()