from trainingSet import init_dataSet
import numpy as np
import tensorflow.keras
from tensorflow.compat.v1.initializers import glorot_normal
from tensorflow.python.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
import pandas as pd
import matplotlib.pyplot as plt
import pickle


"""
Can be training with three type of images GREY-RGB-RGBA
"""
def train(X_train, Y_train, X_test, Y_test, batch_size, epochs, input_channels, model_name,history_name):
	
	#Define Model
	model=Sequential()
	model.add(Convolution2D(96, kernel_size=7,strides=(2, 2), padding='same',
		               input_shape=(250,250,input_channels),kernel_initializer = glorot_normal()))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((3,3),strides=(2, 2),padding='same'))
	model.add(Lambda(tensorflow.nn.local_response_normalization))


	model.add(Convolution2D(384, kernel_size=5,strides=(2, 2), padding='same', 
		                kernel_initializer = glorot_normal()))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((3,3),strides=(2, 2), padding='same'))
	model.add(Lambda(tensorflow.nn.local_response_normalization))

	model.add(Convolution2D(384, kernel_size=5,strides=(2, 2), padding='same',
		                kernel_initializer =glorot_normal()))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((3,3),strides=(2, 2),padding='same'))
	model.add(Lambda(tensorflow.nn.local_response_normalization))

	model.add(Convolution2D(384, kernel_size=3,strides=(2, 2), padding='same',
		                kernel_initializer = glorot_normal()))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((3,3),strides=(2, 2), padding='same'))
	model.add(Lambda(tensorflow.nn.local_response_normalization))

	model.add(Flatten())

	model.add(Dense(4096, activation='relu',
		        kernel_initializer = glorot_normal()))
	model.add(Dropout(0.4))
	model.add(Dense(4096, activation='relu',
		        kernel_initializer = glorot_normal()))
	model.add(Dropout(0.4))
	model.add(Dense(5, activation='softmax'))



	model.compile(loss='categorical_crossentropy',
		      optimizer=SGD(lr=0.001),
		      metrics=['acc'])

	model.summary()

	callbacks_list = [tensorflow.keras.callbacks.ModelCheckpoint(filepath=model_name+'_best.h5',monitor='val_loss',save_best_only=True,)]

	history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
		        callbacks=callbacks_list , validation_data=(X_test,Y_test))

	# Evaluate model on test data
	score = model.evaluate(X_test, Y_test, verbose=0)
	print ("score")

	with open(history_name, 'wb') as handle: # saving the history of the model
	   pickle.dump(history.history, handle)

	return model, history

def plot_accuracy_and_loss(history):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()

def labels_as_emotions(labels):
	labels[labels == 0] = 'anger'
	labels[labels == 1] = 'fear'
	labels[labels == 2] = 'happiness'
	labels[labels == 3] = 'neutral'
	labels[labels == 4] = 'sadness'
	return labels 

def print_confusion_matrix(model, X_test , Y_test):
	pred = model.predict(np.array(X_test))
	actual_results = []
	predicted_results = []
	for i in range(len(Y_test)):
		actual_results.append(np.argmax(Y_test[i]))
		predicted_results.append(np.argmax(pred[i]))

	actual_labels=labels_as_emotions(predicted_results)
	predicted_labels=labels_as_emotions(predicted_results)
	
	actual_labels=pd.Series(actual_labels, name="Actual")	
	predicted_labels=pd.Series(predicted_labels, name="Predicted")	
	
	print(pd.crosstab(actual_labels,predicted_labels))

"""
Prepares data for training.
For GREY : (kind_of_data, noOfchannels)=("specs", 1)
For RGB : (kind_of_data, noOfchannels)=("RGB", 3)
For RGBa : (kind_of_data, noOfchannels)=("RGBa", 4)
"""
def data_preperation(kind_of_data, noOfchannels):
  
	X_train, Y_train, X_test, Y_test=init_dataSet(kind_of_data)

	X_train=np.asarray(X_train)
	print("X train :"+ str(X_train.shape))

	Y_train=np.asarray(Y_train)
	print("Y train :"+ str(Y_train.shape))

	X_test=np.asarray(X_test)
	print("X test :"+ str(X_test.shape))

	Y_test=np.asarray(Y_test)
	print("Y test :"+ str(Y_test.shape))

	X_train=X_train.reshape((-1, 250, 250,noOfchannels))
	X_test=X_test.reshape((-1, 250, 250,noOfchannels))
	print(X_train[0])
	print("X train :"+ str(X_train.shape))
	print("X test :"+ str(X_test.shape))

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	#X_train /= 255
	#X_test /= 255
	#print(X_train[0])

	Y_train = utils.np_utils.to_categorical(Y_train, 5)
	#print(Y_train[0])
	Y_test = utils.np_utils.to_categorical(Y_test, 5)
	print("Y test :"+ str(Y_test.shape))

	return X_train, Y_train, X_test, Y_test


#Training with RGBA spectrograms
X_train, Y_train, X_test, Y_test=data_preperation("RGBa", 4)
 
model , history= train(X_train, Y_train, X_test, Y_test, batch_size=64, epochs=300, input_channels=4, 					
                           model_name= 'RGBA_model', history_name='RGBA_history')  




