from keras.models import load_model
from trainingSet import*
from train_model import*
import os
import numpy as np
from scipy.io import wavfile
import tensorflow
from pickle import load
from keras import models
import matplotlib.pyplot as plt

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

def smooth_curve(points, factor=0.8):
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous * factor + point * (1 - factor))
		else:
			smoothed_points.append(point)
	return smoothed_points


# Recreate the exact same model, including its weights and the optimizer from exercise2_a
model = keras.models.load_model('RGBA_model_best.h5')
model.summary()

with open('RGBA_history', 'rb') as handle: # loading old history 
    oldhstry = load(handle)

#train_model_with_RGBa.plot_accuracy_and_loss(oldhstry)

X_train, Y_train, X_test, Y_test=train_model.data_preperation("RGBa", 4)

loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
epochs=300
plt.plot(smooth_curve(oldhstry['acc']), 'bo', label='Smoothed training acc')
plt.plot(smooth_curve(oldhstry['val_acc']), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(smooth_curve(oldhstry['loss']), 'bo', label='Smoothed training loss')
plt.plot(smooth_curve(oldhstry['val_loss']), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

print_confusion_matrix(model, X_test , Y_test)
