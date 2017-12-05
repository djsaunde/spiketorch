import os
import time
import torch
import timeit
import argparse
import numpy as np
import pickle as p
import matplotlib.pyplot as plt

from struct import unpack
from torchvision import datasets

data_path = os.path.join('..', 'data')

np.set_printoptions(threshold=np.nan, linewidth=100)


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


def classify(spike_activity):
	'''
	Not yet defined.
	'''
	return np.random.choice(10)


def generate_spike_train(image, intensity, time):
	'''
	Generates Poisson spike trains based on image ink intensity.
	'''
	s = []
	image = image / 4
	
	start = timeit.default_timer()

	# Make the spike data. Use a simple Poisson-like spike generator
	# (just for illustrative purposes here. Better spike generators should
	# be used in simulations).
	spike_times = np.random.poisson(image.numpy(), [time, 784])
	spike_times = np.cumsum(spike_times, axis=0)
	spike_times[spike_times >= time] = 0

	# Sparsify spike times into (time, n_input) matrix.
	# spikes = np.zeros([time, 784])
	# spikes[spike_times[spike_times > 0]] = 1

	# Create spikes matrix from spike times.
	spikes = np.zeros([time, 784])
	for idx in range(700):
		spikes[spike_times[idx, :], np.arange(784)] = 1

	# print('It took %.4f seconds to generate one image\'s spike trains' % (timeit.default_timer() - start))

	# Return the input spike occurrence matrix.
	return torch.from_numpy(spikes).float()


class SNN:
	'''
	Replication of the spiking neural network model from "Unsupervised learning of digit
	recognition using spike-timing-dependent plasticity"
	(https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#).
	'''
	def __init__(self, seed=0, n_neurons=100, n_examples=(10000, 10000), dt=0.5, lrs=(1e-4, 1e-2), \
						n_input=784, c_inhib=17.4, sim_times=(350, 150, 350, 150), stdp_times=(20, 20)):
		'''
		Constructs the network based on chosen parameters.

		Arguments:
			- seed: Sets the random number generator sequence.
			- n_neurons: Number of neurons in both excitatory and inhibitory populations.
			- n_examples: Tuple of (n_train, n_test); the number of examples to train and test on.
			- dt: Simulation time step; e.g., 0.5ms.
			- lrs: Tuple of (nu_pre, nu_post); pre- and post-synaptic learning rates.
			- c_inhib: Strength of synapses from inhibitory to excitatory layer.
			- sim_times: Tuple of (train_time, train_rest, test_time, test_rest). Specifies
				how many time steps are used for each training and test examples and rest periods.
			- stdp_times: Tuple of (tc_pre, tc_post); gives STDP window times constants for pre-
				and post-synaptic updates.
		'''
		# Set torch, torch-GPU, and numpy random number generator.
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		
		# Set class attributes.
		self.n_input = n_input
		self.n_input_sqrt = int(np.sqrt(n_input))
		self.n_neurons = n_neurons
		self.n_neurons_sqrt = int(np.sqrt(n_neurons))
		self.n_examples = n_examples
		self.dt = dt
		self.lrs = { 'nu_pre' : lrs[0], 'nu_post' : lrs[1] }
		self.c_inhib = c_inhib
		self.sim_times = { 'train_time' : sim_times[0], 'train_rest' : sim_times[1], \
							'test_time' : sim_times[2], 'test_rest' : sim_times[3] }
		self.stdp_times = { 'tc_pre' : stdp_times[0], 'tc_post' : stdp_times[1] }

		# Population names.
		self.populations = ['Ae', 'Ai']

		# Instantiate weight matrices.
		# Full (random uniform(0.01, 0.3) to start) input to excitatory connectivity.
		self.X_Ae = (torch.rand(784, n_neurons) + 0.01) * 0.3
		# One-to-one connectivity between excitatory and inhibitory neurons.
		self.Ae_Ai = torch.diag(10.4 * torch.ones(n_neurons))
		# All-to-all-but-one connectivity from inhibitory to excitatory layer
		self.Ai_Ae = c_inhib * torch.ones([n_neurons, n_neurons]) - torch.diag(c_inhib * torch.ones(n_neurons))

		# Simulation parameters.
		# Rest (decay towards) voltages.
		self.rest = { 'Ae' : -65.0, 'Ai' : -60.0 }
		# Reset (after spike) voltages.
		self.reset = { 'Ae' : -65.0, 'Ai' : -45.0 }
		# Threshold voltages.
		self.threshold = { 'Ae' : -52.0, 'Ai' : -40.0 }
		# Neuron refractory periods in milliseconds.
		self.refractory = { 'Ae' : 5, 'Ai' : 2 }
		# Adaptive threshold time constant and step increase.
		self.tc_theta = 1e7
		self.theta_plus = 0.05
		# Population-level decay constants.
		self.decay = { 'Ae' : 100, 'Ai' : 10 }
		# Etc.
		self.intensity = 2.0

		# Instantiate neuron state variables.
		# Neuron voltages.
		self.v = { 'Ae' : self.rest['Ae'] * torch.ones(n_neurons), 'Ai' : self.rest['Ai'] * torch.ones(n_neurons) }
		# Spike occurrences.
		self.s = { 'X' : torch.zeros(784), 'Ae' : torch.zeros(n_neurons), 'Ai' : torch.zeros(n_neurons) }
		# Synaptic traces (used for STDP calculations).
		self.a = { 'X' : torch.zeros(784), 'Ae' : torch.zeros(n_neurons) }
		# Adaptive additive threshold parameters (used in excitatory layer).
		self.theta = torch.zeros(n_neurons)


	def run(self, mode, inpt, time):
		'''
		Runs the network on a single input for some time.
		'''
		spikes = { population : torch.zeros([time, self.n_neurons]) for population in self.populations }

		# Run simulation for `time` simulation steps.
		for timestep in range(time):
			# Check for spiking neurons.
			fired = { population : self.v[population] >= self.threshold[population] for population in self.populations }
			for population in self.populations:
				spikes[population][timestep, :] = fired[population]

			# Update neuron voltages.
			for population in self.populations:
				# Integration of input.
				if population == 'Ae':
					self.v[population] = self.v[population] + inpt[timestep, :] @ self.X_Ae - spikes['Ai'][timestep, :] @ self.Ai_Ae
				elif population == 'Ai':
					self.v[population] = self.v[population] + spikes['Ae'][timestep, :] @ self.Ae_Ai

				# Leakiness of integrators.
				self.v[population] = self.v[population] - self.decay[population] * (self.v[population] - self.rest[population])

				# Reset neurons above their threshold voltage.
				self.v[population][self.v[population] > self.threshold[population]] = self.reset[population] 

			# Perform STDP weight update.
			self.X_Ae = self.X_Ae + self.lrs['nu_pre'] * inpt[timestep, :] @ (1 - self.X_Ae) @ self.a['Ae'] - \
										self.lrs['nu_post'] * self.a['X'] @ self.X_Ae @ spikes['Ae'][timestep, :]

		# Return excitatory and inhibitory spikes.
		return spikes

	def get_square_weights(self):
		'''
		Get the weights from the input to excitatory layer and reshape them.
		'''
		square_weights = np.zeros([self.n_input_sqrt * self.n_neurons_sqrt, \
									self.n_input_sqrt * self.n_neurons_sqrt])
		weights = self.X_Ae.numpy()

		for n in range(n_neurons):
			filtr = weights[:, n]
			square_weights[(n % self.n_neurons_sqrt) * self.n_input_sqrt : \
						((n % self.n_neurons_sqrt) + 1) * self.n_input_sqrt, \
						((n // self.n_neurons_sqrt) * self.n_input_sqrt) : \
						((n // self.n_neurons_sqrt) + 1) * self.n_input_sqrt] = \
							filtr.reshape([self.n_input_sqrt, self.n_input_sqrt])
		
		return square_weights.T

if __name__ =='__main__':
	parser = argparse.ArgumentParser(description='LIF simulation toy model implemented with PyTorch.')
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--mode', type=str, default='train')
	parser.add_argument('--n_input', type=int, default=784)
	parser.add_argument('--n_neurons', type=int, default=100)
	parser.add_argument('--n_train', type=int, default=10000)
	parser.add_argument('--n_test', type=int, default=10000)
	parser.add_argument('--dt', type=float, default=0.5)
	parser.add_argument('--nu_pre', type=float, default=1e-4)
	parser.add_argument('--nu_post', type=float, default=1e-2)
	parser.add_argument('--c_inhib', type=float, default=17.4)
	parser.add_argument('--train_time', type=int, default=700)
	parser.add_argument('--train_rest', type=int, default=300)
	parser.add_argument('--test_time', type=int, default=700)
	parser.add_argument('--test_rest', type=int, default=300)
	parser.add_argument('--tc_pre', type=int, default=20)
	parser.add_argument('--tc_post', type=int, default=20)
	parser.add_argument('--gpu', type=str, default='False')
	parser.add_argument('--plot', type=str, default='False')

	# Place parsed arguments in local scope.
	args = parser.parse_args()
	args = vars(args)
	locals().update(args)

	# Print out argument values.
	print('\nOptional argument values:')
	for key, value in args.items():
		print('-', key, ':', value)

	print('\n')

	# Convert string arguments into boolean datatype.
	gpu = gpu == 'True'
	plot = plot == 'True'

	# Initialize the spiking neural network.
	network = SNN(seed, n_neurons, (n_train, n_test), dt, (nu_pre, nu_post), n_input, c_inhib, \
							(train_time, train_rest, test_time, test_rest), (tc_pre, tc_post))

	# Get training, test data from disk.
	train_data = get_labeled_data('train', train=True)
	test_data = get_labeled_data('test', train=False)

	# Convert data into torch Tensors.
	train_X, train_y = torch.from_numpy(train_data['X'][:n_train]), torch.from_numpy(train_data['y'][:n_train])
	test_X, test_y = torch.from_numpy(test_data['X'][:n_test]), torch.from_numpy(test_data['y'][:n_test])

	# Special "zero data" used for network rest period between examples.
	zero_data = torch.zeros(700, n_input).float()

	# Run training phase.
	plt.ion()
	correct = 0
	start = timeit.default_timer()
	for idx, (image, target) in enumerate(zip(train_X, train_y)):
		# Print progress through training data.
		if idx % 10 == 0:
			print('Training progress: (%d / %d) - Elapsed time: %.4f' % (idx, len(train_X), timeit.default_timer() - start))
			start = timeit.default_timer()

		# Run image on network for `train_time` after transforming it into Poisson spike trains.
		inpt = generate_spike_train(image, network.intensity, network.sim_times['train_time'])
		spikes = network.run(mode='train', inpt=inpt, time=network.sim_times['train_time'])

		# Optionally plot the excitatory, inhibitory spiking.
		if plot:
			if idx == 0:
				# Create figure for input image and corresponding spike trains.
				input_figure, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 6))
				im1 = ax1.imshow(image.numpy().reshape(network.n_input_sqrt, network.n_input_sqrt), cmap='binary')
				ax1.set_title('Original MNIST digit (Iteration %d)' % idx)
				im2 = ax2.imshow(inpt.numpy().T, cmap='binary')
				ax2.set_title('Poisson spiking representation')

				plt.tight_layout()

				# Create figure for excitatory and inhibitiory neuron populations.
				spike_figure, [ax3, ax4] = plt.subplots(2, figsize=(10, 5))
				im3 = ax3.imshow(spikes['Ae'].numpy().T, cmap='binary')
				ax3.set_title('Excitatory spikes')
				im4 = ax4.imshow(spikes['Ai'].numpy().T, cmap='binary')
				ax4.set_title('Inhibitory spikes')

				plt.tight_layout()

				# Create figure for input to excitatory weights.
				weights_figure, ax5 = plt.subplots()
				im5 = ax5.imshow(network.get_square_weights(), cmap='hot_r', vmin=0, vmax=1)
				weights_figure.colorbar(im5, ax=ax5)
			else:
				# Reset image data after each iteration.
				im1.set_data(image.numpy().reshape(network.n_input_sqrt, network.n_input_sqrt))
				im2.set_data(inpt.numpy().T)
				im3.set_data(spikes['Ae'].numpy().T)
				im4.set_data(spikes['Ai'].numpy().T)
				im5.set_data(network.get_square_weights())

				# Update title of input digit plot to reflect current iteration.
				ax1.set_title('Original MNIST digit (Iteration %d)' % idx)
			
			plt.pause(1e-8)

		# Classify network output (spikes) based on historical spiking activity.
		prediction = classify(spikes)
		# If correct, increment counter variable.
		if prediction == target.numpy()[0]:
			correct += 1

		# Run zero image on network for `rest_time`.
		network.run(mode='train', inpt=zero_data, time=network.sim_times['train_rest'])

	print('Training accuracy:', correct / n_train)
