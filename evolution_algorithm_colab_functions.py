import keras
from keras.layers import Conv2D, UpSampling2D, Input, Add
from keras.models import Model
from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from PIL import Image
from numpy.random import randint
import numpy as np
import os

"""
The functions below are helper functions used in google colab
to train the super-resolution neural network with random parameters.

These functions are useful for automatic the process of hyperparameter
serach.
"""

# collection of all hyperparameters
h = {
	"l1_parameter" : 0.01,		# for l1 regularizer. [0, 0.02]
	"l2_parameter" : 0.01,		# for l2 regularizer. [0, 0.02]
	"num_residual_blocks" : 4,	# number of residual blocks to use. [2,5]
	"num_conv_blocks" : 2, 		# number of conv blocks to use inside a residual block. [2,4]
	"num_final_conv_blocks" : 2,
	"num_epochs" : 100,			# train everything at 100 epochs for now
	"batch_size" : 16,			# lower this number if ResourceExhaustion errors occur
	"num_filters" : 64,			# [16,32,64,128]
	"learning_rate" : 0.0001,	# parameter for adam optimizer. [0.0001, 0.001]
	"beta_1" : 0.9,				# parameter for adam optimizer. ignore for now
	"beta_2" : 0.999,			# parameter for adam optimizer. ignore for now
	"optimizer" : keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, amsgrad=False),
	'regularizer' : l1_l2(l1=h['l1_parameter'], l2=h['l2_parameter'])
}


def randomize_hyperparamters():
	"""
	Randomizing hyperparameters from within a reasonable range of random numbers
	"""
	global h
	# Note: N evenly spaced random points in interval [a,b) is given by:
	# a + (b - a) * randint(N)/N
	h.update({'l1_parameter' : 0.02 * randint(10) / 10,
			  'l2_parameter' : 0.02 * randint(10) / 10,
			  'num_residual_blocks' : np.random.choice([2,3,4,5]),
			  'num_conv_blocks' : np.random.choice([2,3,4]),
			  'num_final_conv_blocks' : np.random.choice([2,3,4]),
			  'num_filters' : np.random.choice([16,32,64,128]),
			  'learning_rate' : 0.00005 + (0.001 - 0.00005) * randint(20) / 20
		})
	h.update({"optimizer" : keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, amsgrad=False),
			  'regularizer' : l1_l2(l1=h['l1_parameter'], l2=h['l2_parameter'])})


def residual_block(input_layer, activation='relu', kernel_size=(3,3)):
	"""
	A residual block
	"""
	global h
	layer = input_layer
	for i in range(h['num_conv_blocks']):
		layer = Conv2D(h['num_filters'], kernel_size, padding='same', activation=activation, activity_regularizer=h['regularizer'])(layer)
	conv_1x1 = Conv2D(3, (1,1), padding='same')(layer)
	return Add()([conv_1x1, input_layer])


def conv_block(input_layer, activation='relu', kernel_size=(3,3)):
	"""
	Final convolution block
	"""
	global h
	layer = input_layer
	for i in range(h['num_final_conv_blocks']):
		layer = Conv2D(h['num_filters'], kernel_size, padding='same', activation=activation)(layer)
	return layer


def upsample(layer):
	"""
	Upsampling block. Upsamples 2x
	"""
	return UpSampling2D(size=(2,2))(layer)


def build_model():
	"""
	Builds model using chosen hyperparameters
	"""
	global h
	input_layer = Input(shape=(150,150,3))
	layer = input_layer
	for i in range(h['num_residual_blocks']):
		layer = residual_block(layer)
	layer = upsample(layer)
	layer = conv_block(layer)
	output_layer = Conv2D(3, (1,1), padding='same')(layer)
	return Model(inputs=input_layer, outputs=output_layer)


def build_dataset(directory):
	"""
	Returns dataset in (x_train, y_train), (x_test, y_test) format
	"""
	filenames = get_filenames(directory)
	X = []
	Y = []

	# collect images from directory
	for filename in filenames:
		print("Processing", filename)
		image = Image.open(directory + filename)
		image_large = np.array(image)
		image_small = np.array(image.resize((150,150)))
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


def get_filenames(directory):
	"""
	A convenient function for getting image filenames 
	"""
	for _,_,filenames in os.walk(directory):
		pass
	return filenames


def display_results(model, n=10, img_size=10):
	"""
	Display n predictions using the model
	"""
	if n > 1:
		fig, axs = plt.subplots(n, 3, figsize=(3 * img_size, n * img_size))
		for index, ax in enumerate(axs):
			ax[0].imshow(x_test[index])
			ax[0].axis('off')
			ax[1].imshow(y_test[index])
			ax[1].axis('off')
			y_pred = model.predict(x_test[index:index+1])[0]
			ax[2].imshow(y_pred)
			ax[2].axis('off')
		plt.show()
	else:
		print("n must be at least 2")


def display_error_plots(hist):
	"""
	Plot accuracy and error over epochs
	"""
	if type(hist) == keras.callbacks.History:
		acc = hist.history['acc']
		val_acc = hist.history['val_acc']
		loss = hist.history['loss']
		val_loss = hist.history['val_loss']

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
		raise TypeError("Expected object of type keras.callbacks.History not " + type(hist).__name__)

def save_error_plots(hist):
	"""
	Save error and accuracy plots
	"""
	acc = hist.history['acc']
	val_acc = hist.history['val_acc']
	loss = hist.history['loss']
	val_loss = hist.history['val_loss']

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

def save_results(model):
	"""
	Save image comparisons
	"""
	fig, axs = plt.subplots(10, 3, figsize=(30,100))
	for index, ax in enumerate(axs):
		ax[0].imshow(x_test[index])
		ax[0].axis('off')
		ax[1].imshow(y_test[index])
		ax[1].axis('off')
		y_pred = model.predict(x_test[index:index+1])[0]
		ax[2].imshow(y_pred)
		ax[2].axis('off')
	plt.savefig('results.png')
