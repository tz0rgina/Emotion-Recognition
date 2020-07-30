from keras.models import load_model
from trainingSet import*
import os
import numpy as np
from scipy.io import wavfile
import train_model_with_RGBa 
import keras
from pickle import load
from keras import models
import tensorflow as tf
import matplotlib.pyplot as plt

def print_figure(figure_name):
    
    figure_path = os.path.join(os.path.join(os.getcwd(), "figures"))
    
    if os.path.isdir(figure_path):
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    else:
        os.mkdir(figure_path)
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    
    return

def visualizeFirstLayer(model,example):

	#Instantiating a model from an input tensor and a list of output tensors
	layer_outputs = [layer.output for layer in model.layers[:1]] #Extracts the outputs of the top layer
	print(layer_outputs)
	activation_model = models.Model(inputs=model.input, outputs=layer_outputs)


	#Running the model in predict mode
	layer_name=(model.layers[:1][0]).name
	first_layer_activation  = activation_model.predict(example)
	images_per_row = 16
	print(first_layer_activation.shape)
	n_features = first_layer_activation.shape[-1]
	size = first_layer_activation.shape[1]
	n_cols = n_features // images_per_row
	display_grid = np.zeros((size * n_cols, images_per_row * size))

	for col in range(n_cols):
		for row in range(images_per_row):

			channel_image = first_layer_activation[0,
			:, :,
			col * images_per_row + row]
			channel_image -= channel_image.mean()
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size,
				row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	plt.figure(figsize=(scale * display_grid.shape[1],
		scale * display_grid.shape[0]))
	plt.title(layer_name)
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='Reds')

	plt.show()

def plot_outputs(model, no_of_layers , example , emotion):

    layer_outputs = [layer.output for layer in model.layers[:no_of_layers]] # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

    activations = activation_model.predict(example) # Returns a list of six Numpy arrays: one array per layer activation 
    
    layer_names = []

    for layer in model.layers[:no_of_layers]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
      
    images_per_row = 16
    grids=[] 
    
    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                
                channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
               
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                             row * size : (row + 1) * size] = channel_image
        grids.append(display_grid)
        scale = 1. / size

        plt.figure(figsize=(scale * display_grid.shape[1],
        scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
        print_figure(layer_name)

#Preprocessing a single wavfile
fs, signal = wavfile.read(os.path.join("C:/MachineLearningPractice/emovo/EMOVOdata/sadness", "tri-f1-b1.wav"))
exampleRGBa=prepareData([signal],[fs], "RGBa")
#print(exampleRGBa[0].shape)

plt.imshow(exampleRGBa[0],cmap='jet')
plt.show()

exampleRGBa=np.asarray(exampleRGBa)
model = tf.keras.models.load_model('RGBA_model_best.h5' , compile=True)
visualizeFirstLayer(model,exampleRGBa)
plot_outputs(model, 24 , exampleRGBa , "sadness")

