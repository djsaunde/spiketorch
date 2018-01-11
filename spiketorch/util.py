import os
import sys
import torch
import numpy as np
import pickle as p
from struct import unpack

data_path = os.path.join('..', 'data')
params_path = os.path.join('..', 'params')
assign_path = os.path.join('..', 'assignments')
results_path = os.path.join('..', 'results')

for path in [ params_path, assign_path, results_path ]:
	if not os.path.isdir(path):
		os.makedirs(path)


def get_labeled_data(filename, train=True):
	'''
	Read input-vector (image) and target class (label, 0-9) and return it as 
	a list of tuples.
	'''
	if os.path.isfile(os.path.join(data_path, '%s.p' % filename)):
		# Get pickled data from disk.
		data = p.load(open(os.path.join(data_path, '%s.p' % filename), 'rb'))
	else:
		# Open the images with gzip in read binary mode.
		if train:
			images = open(os.path.join(data_path, 'train-images-idx3-ubyte'), 'rb')
			labels = open(os.path.join(data_path, 'train-labels-idx1-ubyte'), 'rb')
		else:
			images = open(os.path.join(data_path, 't10k-images-idx3-ubyte'), 'rb')
			labels = open(os.path.join(data_path, 't10k-labels-idx1-ubyte'), 'rb')

		# Get metadata for images.
		images.read(4)
		number_of_images = unpack('>I', images.read(4))[0]
		rows = unpack('>I', images.read(4))[0]
		cols = unpack('>I', images.read(4))[0]

		# Get metadata for labels.
		labels.read(4)
		N = unpack('>I', labels.read(4))[0]

		if number_of_images != N:
			raise Exception('number of labels did not match the number of images')

		# Get the data.
		print('...Loading MNIST data from disk.')
		print('\n')

		X = np.zeros((N, rows, cols), dtype=np.uint8)
		y = np.zeros((N, 1), dtype=np.uint8)

		for i in range(N):
			if i % 1000 == 0:
				print('Progress :', i, '/', N)
			X[i] = [[unpack('>B', images.read(1))[0] for unused_col in \
							range(cols)] for unused_row in range(rows) ]
			y[i] = unpack('>B', labels.read(1))[0]

		print('Progress :', N, '/', N, '\n')

		X = X.reshape([N, 784])
		data = {'X': X, 'y': y }

		p.dump(data, open(os.path.join(data_path, '%s.p' % filename), 'wb'))

	return data


def generate_spike_train(image, intensity, time):
	'''
	Generates Poisson spike trains based on image ink intensity.
	'''
	# Multiply image by desired intensity.
	image = image * intensity

	# Get number of input neurons.
	n_input = image.shape[0]
	
	# Image data preprocessing (divide by 4, invert (for spike rates),
	# multiply by 1000 (conversion from milliseconds to seconds).
	image = (1 / (image / 4)) * 1000
	image[np.isinf(image)] = 0
	
	# Make the spike data.
	spike_times = np.random.poisson(image, [time, n_input])
	spike_times = np.cumsum(spike_times, axis=0)
	spike_times[spike_times >= time] = 0

	# Create spikes matrix from spike times.
	spikes = np.zeros([time, n_input])
	for idx in range(time):
		spikes[spike_times[idx, :], np.arange(n_input)] = 1

	# Temporary fix: The above code forces a spike from
	# every input neuron on the first time step.
	spikes[0, :] = 0

	# Return the input spike occurrence matrix.
	return spikes


def generate_2d_spike_train(image, intensity, time):
	'''
	Generates Poisson spike trains based on image ink intensity.
	'''
	# Multiply image by desired intensity.
	image = image * intensity

	# Get number of input neurons.
	n_input = image.shape[0]
	n_input_sqrt = int(np.sqrt(n_input))
	
	# Image data preprocessing (divide by 4, invert (for spike rates),
	# multiply by 1000 (conversion from milliseconds to seconds).
	image = (1 / (image / 4)) * 1000
	image[np.isinf(image)] = 0
	
	# Make the spike data.
	spike_times = np.random.poisson(image, [time, n_input])
	spike_times = np.cumsum(spike_times, axis=0)
	spike_times[spike_times >= time] = 0

	# Create spikes matrix from spike times.
	spikes = np.zeros([time, n_input])
	for idx in range(time):
		spikes[spike_times[idx, :], np.arange(n_input)] = 1

	# Temporary fix: The above code forces a spike from
	# every input neuron on the first time step.
	spikes[0, :] = 0

	# Return the input spike occurrence matrix.
	return spikes.reshape([time, 1, n_input_sqrt, n_input_sqrt])


def save_params(model_name, params, fname, prefix):
	'''
	Save network params to disk.

	Arguments:
		- params (numpy.ndarray): Array of params to save.
		- fname (str): File name of file to write to.
	'''
	np.save(os.path.join(params_path, model_name, '_'.join([prefix, fname]) + '.npy'), params)


def load_params(model_name, fname, prefix):
	'''
	Load network params from disk.

	Arguments:
		- fname (str): File name of file to read from.
		- prefix (str): Name of the parameters to read from disk.

	Returns:
		- params (numpy.ndarray): Params stored in file `fname`.
	'''
	return np.load(os.path.join(params_path, model_name, '_'.join([prefix, fname]) + '.npy'))


def save_assignments(model_name, assignments, fname):
	'''
	Save network assignments to disk.

	Arguments:
		- assignments (numpy.ndarray): Array of assignments to save.
		- fname (str): File name of file to write to.
	'''
	np.save(os.path.join(assign_path, model_name, '_'.join(['assignments', fname]) + '.npy'), assignments)


def load_assignments(model_name, fname):
	'''
	Save network assignments to disk.

	Arguments:
		- fname (str): File name of file to read from.

	Returns:
		- assignments (numpy.ndarray): Assignments stored in file `fname`.
	'''
	return np.load(os.path.join(assign_path, model_name, '_'.join(['assignments', fname]) + '.npy'))


def get_square_weights(weights, n_input_sqrt, n_neurons_sqrt):
	'''
	Get the weights from the input to excitatory layer and reshape them.
	'''
	square_weights = np.zeros([n_input_sqrt * n_neurons_sqrt, \
								n_input_sqrt * n_neurons_sqrt])

	for n in range(n_neurons_sqrt ** 2):
		filtr = weights[:, n]
		square_weights[(n % n_neurons_sqrt) * n_input_sqrt : \
					((n % n_neurons_sqrt) + 1) * n_input_sqrt, \
					((n // n_neurons_sqrt) * n_input_sqrt) : \
					((n // n_neurons_sqrt) + 1) * n_input_sqrt] = \
						filtr.reshape([n_input_sqrt, n_input_sqrt])
	
	return square_weights