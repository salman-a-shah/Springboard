"""
This module contains the tools to implement the evolution algorithm.

Create a ModelBuilder class
Fetch the dataset using build_dataset()
Randomize the hyperparameters (within reason) using randomize_hyperparameters()
Build the model using build_model()
Display results using appropriate functions
"""

from tqdm import tqdm 
import keras
from keras.layers import Conv2D, UpSampling2D, Input, Add
from keras.models import Model
from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from numpy.random import randint
import numpy as np
import utils
import sys

class ModelBuilder():

	def __init__(self):
		# hyperparameter
		self.h = {
			"l1_parameter" : 0.01,		# for l1 regularizer. [0, 0.02]
			"l2_parameter" : 0.01,		# for l2 regularizer. [0, 0.02]
			"num_residual_blocks" : 4,	# number of residual blocks to use. [2,5]
			"num_conv_blocks" : 2, 		# number of conv blocks to use inside a residual block. [2,4]
			"num_final_conv_blocks" : 2,
			"num_epochs" : 1,			# train everything at 100 epochs for now
			"batch_size" : 16,			# lower this number if ResourceExhaustion errors occur
			"num_filters" : 64,			# [16,32,64,128]
			"learning_rate" : 0.0001,	# parameter for adam optimizer. [0.0001, 0.001]
			"beta_1" : 0.9,				# parameter for adam optimizer. ignore for now
			"beta_2" : 0.999,			# parameter for adam optimizer. ignore for now
		}
		self.h.update({"optimizer" : keras.optimizers.Adam(lr=self.h['learning_rate'], 
												   beta_1=self.h['beta_1'], 
												   beta_2=self.h['beta_2'], 
												   amsgrad=False),
			'regularizer' : l1_l2(l1=self.h['l1_parameter'], l2=self.h['l2_parameter'])})
		self.model = None
		self.hist = None
		self.dataset = None
		self.accuracy = 0.0

	# randomizes the hyperparameters
	def randomize_hyperparameters(self):
		# Note: N evenly spaced random points in interval [a,b) is given by:
		# a + (b - a) * randint(N)/N
		self.h.update({'l1_parameter' : 0.02 * randint(10) / 10,
				  'l2_parameter' : 0.02 * randint(10) / 10,
				  'num_residual_blocks' : np.random.choice([2,3,4,5]),
				  'num_conv_blocks' : np.random.choice([2,3,4]),
				  'num_final_conv_blocks' : np.random.choice([2,3,4]),
				  'num_filters' : np.random.choice([16,32,64,128]),
				  'learning_rate' : 0.00005 + (0.001 - 0.00005) * randint(20) / 20
			})
		self.h.update({"optimizer" : keras.optimizers.Adam(lr=self.h['learning_rate'], 
														   beta_1=self.h['beta_1'], 
														   beta_2=self.h['beta_2'], 
														   amsgrad=False),
					'regularizer' : l1_l2(l1=self.h['l1_parameter'], l2=self.h['l2_parameter'])})

	# a residual block
	def residual_block(self, input_layer, activation='relu', kernel_size=(3,3)):
		layer = input_layer
		for i in range(self.h['num_conv_blocks']):
			layer = Conv2D(self.h['num_filters'], 
						   kernel_size, 
						   padding='same', 
						   activation=activation, 
						   activity_regularizer=self.h['regularizer'])(layer)
		conv_1x1 = Conv2D(3, (1,1), padding='same')(layer)
		return Add()([conv_1x1, input_layer])

	# final convolution blocks
	def conv_block(self, input_layer, activation='relu', kernel_size=(3,3)):
		layer = input_layer
		for i in range(self.h['num_final_conv_blocks']):
			layer = Conv2D(self.h['num_filters'],
						   kernel_size, 
						   padding='same', 
						   activation=activation)(layer)
		return layer

	# upsamples 2x
	def upsample(self, layer):
		return UpSampling2D(size=(2,2))(layer)

	# builds model based on hyperparameter specs
	def build_model(self):
		input_layer = Input(shape=(150,150,3)) # Todo: remove hardcoded input shape
		layer = input_layer
		for i in range(self.h['num_residual_blocks']):
			layer = self.residual_block(layer)
		layer = self.upsample(layer)
		layer = self.conv_block(layer)
		output_layer = Conv2D(3, (1,1), padding='same')(layer)
		return Model(inputs=input_layer, outputs=output_layer)

	# returns dataset in (x_train, y_train), (x_test, y_test) format
	# by default, uses the downscaled image files
	# (150,150,3) to train, (300,300,3) to test
	def build_dataset(self, directory="./dataset/downscaled/"):
		# initialize variables
		filenames = utils.get_filenames(directory)
		X = []
		Y = []

		# collect images from directory
		print("Processing image files...")
		for filename in tqdm(filenames):
			image = Image.open(directory + filename)
			image_large = np.array(image)
			dim = image_large.shape
			image_small = np.array(image.resize((int(dim[0]/2.0), int(dim[1]/2.0))))
			Y.append(image_large)
			X.append(image_small)

		# convert to matrices
		X = np.asarray(X)
		X = X.astype('float32')
		X /= 255
		Y = np.asarray(Y)
		Y = Y.astype('float32')
		Y /= 255

		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

		# following keras convention for load_data() 
		return ((x_train, y_train), (x_test, y_test))

	def train_model(self, verbose=True):
		if self.dataset == None:
			if verbose:
				print("Loading dataset...")
			self.dataset = self.build_dataset()
		(x_train, y_train), (x_test, y_test) = self.dataset
		
		if verbose:
			print("Randomizing hyperparameters...")
		self.randomize_hyperparameters()

		if verbose:
			print("Building model...")
		self.model = self.build_model()
		if verbose:
			self.model.summary()
			print("Compiling...")
		self.model.compile(loss='mse', optimizer=self.h['optimizer'], metrics=['accuracy'])
		self.hist = self.model.fit(x_train,
								   y_train,
								   batch_size=self.h['batch_size'],
								   epochs=self.h['num_epochs'],
								   verbose=verbose,
								   validation_data=(x_test, y_test))

	def run_algorithm(self, iteration=1, verbose=True):
		if verbose:
			print("Running algorithm for", iteration, "iterations.")
		for i in range(iteration):
			if verbose:
				print("Iteration", i,)
			self.train_model()
			if verbose:
				print("Training complete. Recording accuracy...")
			self.accuracy = self.model.evaluate(x_test, y_test)[1]	# record the last validation accuracy
			if verbose:
				print("Saving results...")
			self.save_model_summary()
			self.save_error_plots()
			self.save_results()
			if verbose:
				print("Serializing model...")
			self.save_model_to_disk()
			if verbose:
				print("Iteration complete\n")

	# show results of a trained model 
	def display_results(self, n=10, img_size=10):
		if model == None:
			print("No results to display.")
			return
		if n > 1:
			fig, axs = plt.subplots(n, 3, figsize=(3 * img_size, n * img_size))
			for index, ax in enumerate(axs):
				ax[0].imshow(x_test[index])
				ax[0].axis('off')
				ax[1].imshow(y_test[index])
				ax[1].axis('off')
				y_pred = self.model.predict(x_test[index:index+1])[0]
				ax[2].imshow(y_pred)
				ax[2].axis('off')
			plt.show()
		else:
			print("n must be at least 2")

	# plot accuracy and error over epochs
	def display_error_plots(self):
		if self.hist is not None:
			acc = self.self.hist.history['acc']
			val_acc = self.self.hist.history['val_acc']
			loss = self.self.hist.history['loss']
			val_loss = self.self.hist.history['val_loss']

			fig, axs = plt.subplots(1,2, figsize=(15,5))
			axs[0].plot(acc, label='Training accuracy')
			axs[0].plot(val_acc, label='Validation accuracy')
			axs[0].legend(loc='lower right')
			axs[0].set_title("Accuracy")
			axs[1].plot(loss, label='Training error')
			axs[1].plot(val_loss, label='Validation error')
			axs[1].legend()
			axs[1].set_title("Error")
			plt.show()
		else:
			print("Could not find training history.")

	# save error and accuracy plots
	def save_error_plots(self):
		if self.hist is not None:
			acc = self.hist.history['acc']
			val_acc = self.hist.history['val_acc']
			loss = self.hist.history['loss']
			val_loss = self.hist.history['val_loss']

			fig, axs = plt.subplots(1,2, figsize=(15,5))
			axs[0].plot(acc, label='Training accuracy')
			axs[0].plot(val_acc, label='Validation accuracy')
			axs[0].legend(loc='lower right')
			axs[0].set_title("Accuracy")
			axs[1].plot(loss, label='Training error')
			axs[1].plot(val_loss, label='Validation error')
			axs[1].legend()
			axs[1].set_title("Error")

			plt.savefig('plots.png')
		else:
			print("Could not find training history.")

	# save final image comparisons
	def save_results(self):
		if model == None:
			print("No results to save.")
			return
		fig, axs = plt.subplots(10, 3, figsize=(30,100))
		for index, ax in enumerate(axs):
			ax[0].imshow(x_test[index])
			ax[0].axis('off')
			ax[1].imshow(y_test[index])
			ax[1].axis('off')
			y_pred = self.model.predict(x_test[index:index+1])[0]
			ax[2].imshow(y_pred)
			ax[2].axis('off')
		plt.savefig('./results/results.png')

	def save_model_summary(self, filepath='./results/model_summary.txt'):
		# save model summary by redirecting stdout to a file
		orig_stdout = sys.orig_stdout
		f = open(filepath, 'w')
		sys.stdout = f
		print("Hyperparameters:")
		for key, value in self.h.items():
			print(key, ":", value)
		print("\nFinal accuracy:", self.accuracy, "\n")
		self.model.summary()
		sys.stdout = orig_stdout
		f.close()

	def save_model_to_disk(self, directory='./results/'):
		model_json = self.model.to_json()
		with open(directory + "model.json", "w") as json_file:
			json_file.write(model_json)
		model.save_weights(directory + "model.h5")



