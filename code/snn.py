import os
import sys
import time
import torch
import timeit
import argparse
import numpy as np
import pickle as p
import matplotlib.pyplot as plt

from struct import unpack
from torchvision import datasets
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
	return torch.multinomial(torch.arange(0, 10), 1).byte()[0]


def generate_spike_train(image, intensity, time):
	'''
	Generates Poisson spike trains based on image ink intensity.
	'''
	s = []
	image = (image / 4)
	n_input = image.size()[0]
	
	start = timeit.default_timer()

	# Make the spike data. Use a simple Poisson-like spike generator
	# (just for illustrative purposes here. Better spike generators should
	# be used in simulations).
	spike_times = np.random.poisson(image.cpu().numpy(), [time, n_input])
	spike_times = np.cumsum(spike_times, axis=0)
	spike_times[spike_times >= time] = 0

	# Create spikes matrix from spike times.
	spikes = np.zeros([time, n_input])
	for idx in range(time):
		spikes[spike_times[idx, :], np.arange(n_input)] = 1

	# print('It took %.4f seconds to generate one image\'s spike trains' % (timeit.default_timer() - start))

	# Return the input spike occurrence matrix.
	if gpu:
		return torch.from_numpy(spikes).byte().cuda()
	else:
		return torch.from_numpy(spikes).byte()


class SNN:
	'''
	Replication of the spiking neural network model from "Unsupervised learning of digit
	recognition using spike-timing-dependent plasticity"
	(https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#).
	'''
	def __init__(self, seed=0, n_input=784, n_neurons=100, n_examples=(10000, 10000), dt=0.5, lrs=(1e-4, 1e-2), \
						c_inhib=17.4, sim_times=(350, 150, 350, 150), stdp_times=(20, 20), update_interval=100):
		'''
		Constructs the network based on chosen parameters.

		Arguments:
			- seed: Sets the random number generator sequence.
			- n_neurons: Number of neurons in both excitatory and inhibitory populations.
			- n_input: Number of input neurons (corresponds to dimensionality of image).
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

		# Assignments and performance monitoring update interval.
		self.update_interval = update_interval

		# Excitatory neuron assignments.
		self.assignments = -1 * torch.ones(n_neurons)

		# Instantiate weight matrices.+
		self.W = { 'X_Ae' : (torch.rand(n_input, n_neurons) + 0.01) * 0.3, \
					'Ae_Ai' : torch.diag(15.0 * torch.ones(n_neurons)), \
					'Ai_Ae' : c_inhib * torch.ones([n_neurons, n_neurons]) \
							- torch.diag(c_inhib * torch.ones(n_neurons)) }

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
		self.theta_decay = 1 / 1e7
		self.theta_plus = 0.05
		# Population-level decay constants.
		self.v_decay = { 'Ae' : 1 / 100, 'Ai' : 1 / 10 }
		self.a_decay = { 'X' : 1 / 20, 'Ae' : 1 / 20 }
		# Etc.
		self.intensity = 2.0
		self.wmax = 1
		self.norm = 78.0 * self.wmax

		# Instantiate neuron state variables.
		# Neuron voltages.
		self.v = { 'Ae' : self.rest['Ae'] * torch.ones(n_neurons), 'Ai' : self.rest['Ai'] * torch.ones(n_neurons) }
		# Spike occurrences.
		self.s = { 'X' : torch.zeros(n_input), 'Ae' : torch.zeros(n_neurons), 'Ai' : torch.zeros(n_neurons) }
		# Synaptic traces (used for STDP calculations).
		self.a = { 'X' : torch.zeros(n_input), 'Ae' : torch.zeros(n_neurons) }
		# Adaptive additive threshold parameters (used in excitatory layer).
		self.theta = torch.zeros(n_neurons)


	def run(self, mode, inpt, time):
		'''
		Runs the network on a single input for some time.
		'''
		spikes = { pop : torch.zeros([time, self.n_neurons]).byte() for pop in self.populations }
		voltages = { pop : torch.zeros([time, self.n_neurons]) for pop in self.populations }
		traces = { 'X' : torch.zeros([time, self.n_input]), 'Ae' : torch.zeros([time, self.n_neurons]) }

		# Run simulation for `time` simulation steps.
		for timestep in range(time):
			# Check for spiking neurons.
			fired = {}
			for pop in self.populations:
				if pop == 'Ae':
					fired[pop] = self.v[pop] >= self.threshold[pop] + self.theta
					self.theta[fired[pop]] += self.theta_plus
				elif pop == 'Ai':
					fired[pop] = self.v[pop] >= self.threshold[pop]

				spikes[pop][timestep, :] = fired[pop]

			# Recoding synaptic traces.
			for pop in ['X', 'Ae']:
				if pop == 'X':
					self.a[pop][inpt[timestep, :].byte()] = 1.0
				elif pop == 'Ae':
					self.a[pop][spikes[pop][timestep, :].byte()] = 1.0

			# Reset neurons above their threshold voltage.
			for pop in self.populations:
				if pop == 'Ae':
					self.v[pop][self.v[pop] > self.threshold[pop] + self.theta] = self.reset[pop]
				elif pop == 'Ai':
					self.v[pop][self.v[pop] > self.threshold[pop]] = self.reset[pop]

			# Update neuron voltages.
			for pop in self.populations:
				# Integration of input.
				if pop == 'Ae':
					self.v[pop] += inpt[timestep, :].float() @ self.W['X_Ae'] - spikes['Ai'][timestep, :].float() @ self.W['Ai_Ae']
				elif pop == 'Ai':
					self.v[pop] += spikes['Ae'][timestep, :].float() @ self.W['Ae_Ai']

				# Leakiness of integrators.
				self.v[pop] -= self.v_decay[pop] * (self.v[pop] - self.rest[pop])

				# Record voltage history.
				voltages[pop][timestep, :] = self.v[pop]

			# Perform STDP weight update.
			self.W['X_Ae'] = self.W['X_Ae'] - self.lrs['nu_pre'] * (inpt[timestep, :].float().view(self.n_input, 1) \
								* self.a['Ae'].view(1, self.n_neurons)) * (self.wmax - self.W['X_Ae']) + self.lrs['nu_post'] * \
						self.W['X_Ae'] * (self.a['X'].view(self.n_input, 1) * spikes['Ae'][timestep, :].float().view(1, self.n_neurons))

			# Decay synaptic traces.
			for pop in ['X', 'Ae']:
				self.a[pop] -= self.a_decay[pop] * self.a[pop]

				# Record synaptic trace history.
				traces[pop][timestep, :] = self.a[pop]

		# Normalize weights after update.
		self.normalize_weights()

		# Return excitatory and inhibitory spikes.
		return spikes, voltages, traces


	def get_square_weights(self):
		'''
		Get the weights from the input to excitatory layer and reshape them.
		'''
		square_weights = np.zeros([self.n_input_sqrt * self.n_neurons_sqrt, \
									self.n_input_sqrt * self.n_neurons_sqrt])
		
		if gpu:
			weights = self.W['X_Ae'].cpu().numpy()
		else:
			weights = self.W['X_Ae'].numpy()

		for n in range(n_neurons):
			filtr = weights[:, n]
			square_weights[(n % self.n_neurons_sqrt) * self.n_input_sqrt : \
						((n % self.n_neurons_sqrt) + 1) * self.n_input_sqrt, \
						((n // self.n_neurons_sqrt) * self.n_input_sqrt) : \
						((n // self.n_neurons_sqrt) + 1) * self.n_input_sqrt] = \
							filtr.reshape([self.n_input_sqrt, self.n_input_sqrt])
		
		return square_weights


	def normalize_weights(self):
		'''
		Normalize weights on synpases from input to excitatory layer.
		'''
		for idx in range(self.n_neurons):
			self.W['X_Ae'][:, idx] = self.W['X_Ae'][:, idx] * (self.norm / self.W['X_Ae'][:, idx].sum())


	def assign_labels(self, inputs, outputs):
		'''
		Given the excitatory neuron firing history, assign them class labels.
		'''
		print(inputs.size())
		print(outputs.size())

		rates = torch.zeros([self.n_neurons, 10])
		for j in range(10):
			n_inputs = torch.nonzero(inputs == j).size(0)
			if n_inputs > 0:
				rates[:, j] += torch.sum(outputs[inputs == j], axis=0) / n_inputs

		self.assignments = torch.argmax(rates, 1)[1]


if __name__ =='__main__':
	parser = argparse.ArgumentParser(description='LIF simulation toy model implemented with PyTorch.')
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--mode', type=str, default='train')
	parser.add_argument('--n_input', type=int, default=784)
	parser.add_argument('--n_neurons', type=int, default=100)
	parser.add_argument('--n_train', type=int, default=10000)
	parser.add_argument('--n_test', type=int, default=10000)
	parser.add_argument('--update_interval', type=int, default=100)
	parser.add_argument('--dt', type=float, default=0.5)
	parser.add_argument('--nu_pre', type=float, default=1e-4)
	parser.add_argument('--nu_post', type=float, default=1e-2)
	parser.add_argument('--c_inhib', type=float, default=17.4)
	parser.add_argument('--train_time', type=int, default=350)
	parser.add_argument('--train_rest', type=int, default=150)
	parser.add_argument('--test_time', type=int, default=350)
	parser.add_argument('--test_rest', type=int, default=150)
	parser.add_argument('--tc_pre', type=int, default=20)
	parser.add_argument('--tc_post', type=int, default=20)
	parser.add_argument('--gpu', type=str, default='True')
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
	gpu = gpu == 'True' and torch.cuda.is_available()
	plot = plot == 'True'

	if gpu:
		torch.set_default_tensor_type('torch.cuda.FloatTensor')

	# Initialize the spiking neural network.
	network = SNN(seed, n_input, n_neurons, (n_train, n_test), dt, (nu_pre, nu_post), c_inhib, \
				(train_time, train_rest, test_time, test_rest), (tc_pre, tc_post), update_interval)

	# Get training, test data from disk.
	train_data = get_labeled_data('train', train=True)
	test_data = get_labeled_data('test', train=False)

	# Convert data into torch Tensors.
	train_X, train_y = torch.from_numpy(train_data['X'][:n_train]), torch.from_numpy(train_data['y'][:n_train])
	test_X, test_y = torch.from_numpy(test_data['X'][:n_test]), torch.from_numpy(test_data['y'][:n_test])

	if gpu:
		train_X = train_X.cuda()
		train_y = train_y.cuda()
		test_X = test_X.cuda()
		test_y = test_y.cuda()

	# Special "zero data" used for network rest period between examples.
	zero_data = torch.zeros(train_rest, n_input).float()

	# Count spikes from each neuron on each example (between update intervals).
	spike_monitor = torch.zeros([network.update_interval, network.n_neurons])

	# Run training phase.
	plt.ion()
	correct = 0
	start = timeit.default_timer()
	for idx, (image, target) in enumerate(zip(train_X, train_y)):
		# Assign labels to neurons based on network spiking activity.
		if idx > 0 and idx % network.update_interval == 0:
			network.assign_labels(train_y[idx - network.update_interval : idx], spike_monitor)

		# Print progress through training data.
		if idx % 10 == 0:
			print('Training progress: (%d / %d) - Elapsed time: %.4f' % (idx, len(train_X), timeit.default_timer() - start))
			start = timeit.default_timer()

		# Run image on network for `train_time` after transforming it into Poisson spike trains.
		inpt = generate_spike_train(image, network.intensity, network.sim_times['train_time'])
		spikes, voltages, traces = network.run(mode='train', inpt=inpt, time=network.sim_times['train_time'])

		# Classify network output (spikes) based on historical spiking activity.
		prediction = classify(spikes)
		# If correct, increment counter variable.
		if prediction == target[0]:
			correct += 1

		# Run zero image on network for `rest_time`.
		rest_spikes, rest_voltages, rest_traces = network.run(mode='train', inpt=zero_data, time=network.sim_times['train_rest'])

		# Concatenate image and rest network data for plotting purposes.
		spikes = { pop : torch.cat([spikes[pop], rest_spikes[pop]]) for pop in network.populations }
		voltages = { pop : torch.cat([voltages[pop], rest_voltages[pop]]) for pop in network.populations }
		traces = { pop : torch.cat([traces[pop], rest_traces[pop]]) for pop in ['X', 'Ae'] }

		# Add spikes from this iteration to the spike monitor
		spike_monitor[idx % network.update_interval] = torch.sum(spikes['Ae'], 0)

		# Optionally plot the excitatory, inhibitory spiking.
		if plot:
			if idx == 0:
				# Create figure for input image and corresponding spike trains.
				input_figure, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 6))
				im1 = ax1.imshow(image.cpu().numpy().reshape(network.n_input_sqrt, network.n_input_sqrt), cmap='binary')
				ax1.set_title('Original MNIST digit (Iteration %d)' % idx)
				im2 = ax2.imshow(inpt.cpu().numpy().T, cmap='binary')
				ax2.set_title('Poisson spiking representation')

				plt.tight_layout()

				# Create figure for excitatory and inhibitory neuron populations.
				spike_figure, [ax3, ax4] = plt.subplots(2, figsize=(10, 5))
				im3 = ax3.imshow(spikes['Ae'].cpu().numpy().T, cmap='binary')
				ax3.set_title('Excitatory spikes')
				im4 = ax4.imshow(spikes['Ai'].cpu().numpy().T, cmap='binary')
				ax4.set_title('Inhibitory spikes')

				plt.tight_layout()

				# Create figure for input to excitatory weights and excitatory neuron assignments.
				weights_figure, [ax5, ax6] = plt.subplots(1, 2, figsize=(10, 6))
				im5 = ax5.imshow(network.get_square_weights(), cmap='hot_r', vmin=0, vmax=network.wmax)
				ax5.set_title('Input to excitatory weights')
				color = plt.get_cmap('RdBu', 11)
				im6 = ax6.matshow(network.assignments.cpu().numpy().reshape([network.n_neurons_sqrt, \
											network.n_neurons_sqrt]), cmap=color, vmin=-1.5, vmax=9.5)
				ax2.set_title('Neuron labels')

				div5 = make_axes_locatable(ax5)
				div6 = make_axes_locatable(ax6)
				cax5 = div5.append_axes("right", size="5%", pad=0.05)
				cax6 = div6.append_axes("right", size="5%", pad=0.05)

				plt.colorbar(im5, cax=cax5)
				plt.colorbar(im6, cax=cax6, ticks=np.arange(-1, 10))

				plt.tight_layout()

				# Create figure to display neuron voltages over the iteration.
				# voltages_figure, [ax6, ax7] = plt.subplots(2, figsize=(10, 5))
				# ax6.plot(voltages['Ae'].cpu().numpy())
				# ax7.plot(voltages['Ai'].cpu().numpy())

				# plt.tight_layout()

				# # Create for displaying synaptic traces over the iteration.
				# voltages_figure, [ax8, ax9] = plt.subplots(2, figsize=(10, 5))
				# ax8.plot(network.a['X'].cpu().numpy())
				# ax9.plot(network.a['Ae'].cpu().numpy())
			else:
				# Reset image data after each iteration.
				im1.set_data(image.cpu().numpy().reshape(network.n_input_sqrt, network.n_input_sqrt))
				im2.set_data(inpt.cpu().numpy().T)
				im3.set_data(spikes['Ae'].cpu().numpy().T)
				im4.set_data(spikes['Ai'].cpu().numpy().T)
				im5.set_data(network.get_square_weights())
				
				# ax6.clear(); ax7.clear(); # ax8.clear(); ax9.clear()
				# ax6.plot(voltages['Ae'].cpu().numpy())
				# ax7.plot(voltages['Ai'].cpu().numpy())
				# ax8.plot(traces['X'].cpu().numpy())
				# ax9.plot(traces['Ae'].cpu().numpy())

				# Update title of input digit plot to reflect current iteration.
				ax1.set_title('Original MNIST digit (Iteration %d)' % idx)
			
			plt.pause(1e-8)

	print('Training accuracy:', correct / n_train)
