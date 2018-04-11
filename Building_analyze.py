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
	return X, Y

def get_data():
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
	X, Y = preprocess(df, H)
	#print(X.shape, Y.shape, 'In get_data')
	return X, Y

def get_model(input_size):
	a = Input(n_input=input_size, n_output=5)
	a = Linear(a)
	a = Dense(7, a)
	a = Tanh(a)
	a = Dense(5, a)
	a = Tanh(a)
	a = Dense(1, a)
	a = Linear(a)
	a = Output(a)
	model = Model(a)
	model.compile(optimizer='SGD', lr=0.00003, decay_rate=0.99996 , loss='se', estimator='rms')
	#print(model)
	return model

def main():
	X, Y = get_data()
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	X = X[randomize]
	Y = Y[randomize]
	
	X_train, Y_train = X[:576], Y[:576]
	X_test,  Y_test  = X[576:], Y[576:]
	model = get_model(input_shape)

	hist = model.update(X_train, Y_train, batch_size=3, trainig_epoch=240000, 
				X_val=X_test[:100], Y_val=Y_test[:100], validate_every_n_epoch=2000, record_every_n_epoch=100)

	plt.plot(hist['acc'], color='red', linewidth=1)
	plt.plot(hist['loss'], color='green', linwidth=1, linestyle='dashed')

if __name__ == '__main__':
	main()