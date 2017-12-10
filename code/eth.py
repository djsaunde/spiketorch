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

np.set_printoptions(threshold=np.nan, linewidth=200)
np.warnings.filterwarnings('ignore')
torch.set_printoptions(threshold=np.nan, linewidth=100, edgeitems=10)


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
	s = []
	n_input = image.size()[0]
	
	start = timeit.default_timer()

	# Image data preprocessing (divide by 4, invert (for spike rates),
	# multiply by 1000 (conversion from milliseconds to seconds).
	image = (1 / (image.cpu().numpy() / 4)) * 1000
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
	if gpu:
		return torch.from_numpy(spikes).byte().cuda()
	else:
		return torch.from_numpy(spikes).byte()


class ETH:
	'''
	Replication of the spiking neural network model from "Unsupervised learning of digit
	recognition using spike-timing-dependent plasticity"
	(https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#).
	'''
	def __init__(self, seed=0, n_input=784, n_neurons=100, n_examples=(10000, 10000), dt=1, lrs=(1e-4, 1e-2), \
				c_inhib=17.4, sim_times=(350, 150, 350, 150), stdp_times=(20, 20), update_interval=100, wmax=1.0):
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
		self.stdp_times = { 'X' : dt / stdp_times[0], 'Ae' : dt / stdp_times[1] }

		# Population names.
		self.populations = ['Ae', 'Ai']
		self.stdp_populations = ['X', 'Ae']

		# Assignments and performance monitoring update interval.
		self.update_interval = update_interval

		# Excitatory neuron assignments.
		self.assignments = -1 * torch.ones(n_neurons)

		# Instantiate weight matrices.+
		self.W = { 'X_Ae' : (torch.rand(n_input, n_neurons) + 0.01) * 0.3, \
					'Ae_Ai' : torch.diag(22.5 * torch.ones(n_neurons)), \
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
		self.theta_plus = 0.1
		# Population-level decay constants.
		self.v_decay = { 'Ae' : 1 / 100, 'Ai' : 1 / 10 }
		# Voting schemes.
		self.voting_schemes = [ 'all' ]
		# Network performances indexed by voting schemes.
		self.performances = { scheme : [] for scheme in self.voting_schemes }
		# Excitatory neuron average rates per category.
		self.rates = torch.zeros([self.n_neurons, 10])
		# Etc.
		self.intensity = 2.0
		self.wmax = wmax
		self.norm = 78.0 * wmax

		# Instantiate neuron state variables.
		# Neuron voltages.
		self.v = { 'Ae' : self.rest['Ae'] * torch.ones(n_neurons), 'Ai' : self.rest['Ai'] * torch.ones(n_neurons) }
		# Spike occurrences.
		self.s = { 'X' : torch.zeros(n_input), 'Ae' : torch.zeros(n_neurons), 'Ai' : torch.zeros(n_neurons) }
		# Synaptic traces (used for STDP calculations).
		self.a = { 'X' : torch.zeros(n_input), 'Ae' : torch.zeros(n_neurons) }
		# Adaptive additive threshold parameters (used in excitatory layer).
		self.theta = torch.zeros(n_neurons)
		# Refractory period counters.
		self.refrac_count = { 'Ae' : torch.zeros(n_neurons), 'Ai' : torch.zeros(n_neurons) }


	def run(self, mode, inpt, time):
		'''
		Runs the network on a single input for some time.

		Arguments:
			- mode (str): Whether we are in test or training mode.
					Affects whether to adaptive network parameters. 
			- inpt (torch.Tensor / torch.cuda.Tensor): Network input, 
					encoded as Poisson spike trains. Has shape (time, 
					self.n_input).
			- time (int): How many simulation time steps to run.

		Returns:
			State variables recorded over the simulation iteration.
		'''
		# Records network state variables for plotting purposes.
		spikes = { pop : torch.zeros([time, self.n_neurons]).byte() for pop in self.populations }
		if plot:
			voltages = { pop : torch.zeros([time, self.n_neurons]) for pop in self.populations }
			traces = { 'X' : torch.zeros([time, self.n_input]), 'Ae' : torch.zeros([time, self.n_neurons]) }

		# Run simulation for `time` simulation steps.
		for timestep in range(time):
			# Get input spikes for this timestep.
			self.s['X'] = inpt[timestep, :]

			# Record voltage history.
			if plot:
				voltages['Ae'][timestep, :] = self.v['Ae']
				voltages['Ai'][timestep, :] = self.v['Ai']

			# Decrement refractory counters.
			self.refrac_count['Ae'][self.refrac_count['Ae'] != 0] -= 1
			self.refrac_count['Ai'][self.refrac_count['Ai'] != 0] -= 1

			# Check for spiking neurons.
			self.s['Ae'] = (self.v['Ae'] >= self.threshold['Ae'] + self.theta) * (self.refrac_count['Ae'] == 0)
			self.s['Ai'] = (self.v['Ai'] >= self.threshold['Ai']) * (self.refrac_count['Ai'] == 0)

			# Reset refractory periods for spiked neurons.
			self.refrac_count['Ae'][self.s['Ae']] = self.refractory['Ae']
			self.refrac_count['Ai'][self.s['Ai']] = self.refractory['Ai']

			# Update adaptive thresholds.
			self.theta[self.s['Ae']] += self.theta_plus

			# Record neuron spiking.
			spikes['Ae'][timestep, :] = self.s['Ae']
			spikes['Ai'][timestep, :] = self.s['Ai']

			# Setting synaptic traces.
			self.a['X'][self.s['X'].byte()] = 1.0
			self.a['Ae'][self.s['Ae'].byte()] = 1.0

			# Reset neurons above their threshold voltage.
			self.v['Ae'][self.s['Ae']] = self.reset['Ae']
			self.v['Ai'][self.s['Ai']] = self.reset['Ai']

			# Integrate input and decay voltages.
			self.v['Ae'] += self.s['X'].float() @ self.W['X_Ae'] - self.s['Ai'].float() @ self.W['Ai_Ae']
			self.v['Ae'] -= self.v_decay['Ae'] * (self.v['Ae'] - self.rest['Ae'])
			self.v['Ai'] += self.s['Ae'].float() @ self.W['Ae_Ai']
			self.v['Ai'] -= self.v_decay['Ai'] * (self.v['Ai'] - self.rest['Ai'])

			if mode == 'train':
				# Perform STDP weight update.
				# Post-synaptic.
				self.W['X_Ae'] += self.lrs['nu_post'] * (self.a['X'].view(self.n_input, 1) * self.s['Ae'].float().view(1, self.n_neurons))
				# Pre-synaptic.
				self.W['X_Ae'] -= self.lrs['nu_pre'] * (self.s['X'].float().view(self.n_input, 1) * self.a['Ae'].view(1, self.n_neurons))

				# Ensure that weights are within [0, self.wmax].
				self.W['X_Ae'] = torch.clamp(self.W['X_Ae'], 0, self.wmax)

				# Decay synaptic traces.
				self.a['X'] -= self.stdp_times['X'] * self.a['X']
				self.a['Ae'] -= self.stdp_times['Ae'] * self.a['Ae']
			
				# Record synaptic trace history.
				if plot:
					traces['X'][timestep, :] = self.a['X']
					traces['Ae'][timestep, :] = self.a['Ae']

				# Decay adaptive thresholds.
				self.theta -= self.theta_decay * self.theta

		# Normalize weights after one iteration.
		self.normalize_weights()

		# Return recorded state variables.
		if plot:
			return spikes, voltages, traces
		else:
			return spikes


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
		self.W['X_Ae'] *= self.norm / self.W['X_Ae'].sum(0).view(1, -1)


	def assign_labels(self, inputs, outputs):
		'''
		Given the excitatory neuron firing history, assign them class labels.
		'''
		# Loop over all target categories.
		for j in range(10):
			# Count the number of inputs having this target.
			n_inputs = torch.nonzero(inputs == j).numel()
			if n_inputs > 0:
				# Get indices of inputs with this category.
				idxs = torch.nonzero((inputs == j).long().view(-1)).view(-1)
				# Calculate average firing rate per neuron, per category.
				self.rates[:, j] = 0.9 * self.rates[:, j] + torch.sum(outputs[idxs], 0) / n_inputs

		# Assignments of neurons are the categories for which they fire the most. 
		self.assignments = torch.max(self.rates, 1)[1]


	def classify(self, spikes):
		'''
		Given the neuron assignments and the network spiking
		activity, make predictions about the data targets.
		'''
		predictions = {}
		for scheme in self.voting_schemes:
			rates = torch.zeros(10)

			if scheme == 'all':
				for idx in range(10):
					n_assigns = torch.nonzero(self.assignments == idx).numel()
					if n_assigns > 0:
						rates[idx] = torch.sum(spikes[torch.nonzero((self.assignments == idx).long().view(-1)).view(-1)]) / n_assigns

			predictions[scheme] = torch.sort(rates, dim=0, descending=True)[1]

		return predictions

		# spikes = torch.sum(spikes, 0)

		# predictions = {}
		# for scheme in self.voting_schemes:
		# 	rates = torch.zeros(10)

		# 	if scheme == 'all':
		# 		for idx in range(10):
		# 			n_assigns = torch.nonzero(self.assignments == idx).numel()
		# 			if n_assigns > 0:
		# 				idxs = torch.nonzero(self.assignments == idx).view(-1)
		# 				rates[idx] = torch.sum(torch.index_select(spikes, 0, idxs)) / n_assigns

		# 	predictions[scheme] = torch.sort(rates, dim=0, descending=True)[1]

		# return predictions



if __name__ =='__main__':
	parser = argparse.ArgumentParser(description='ETH (with LIF neurons) SNN toy model simulation implemented with PyTorch.')
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--mode', type=str, default='train')
	parser.add_argument('--n_input', type=int, default=784)
	parser.add_argument('--n_neurons', type=int, default=100)
	parser.add_argument('--n_train', type=int, default=10000)
	parser.add_argument('--n_test', type=int, default=10000)
	parser.add_argument('--update_interval', type=int, default=100)
	parser.add_argument('--dt', type=float, default=1)
	parser.add_argument('--nu_pre', type=float, default=1e-4)
	parser.add_argument('--nu_post', type=float, default=1e-2)
	parser.add_argument('--c_inhib', type=float, default=17.4)
	parser.add_argument('--train_time', type=int, default=350)
	parser.add_argument('--train_rest', type=int, default=150)
	parser.add_argument('--test_time', type=int, default=350)
	parser.add_argument('--test_rest', type=int, default=150)
	parser.add_argument('--tc_pre', type=int, default=20)
	parser.add_argument('--tc_post', type=int, default=20)
	parser.add_argument('--wmax', type=float, default=1.0)
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

	# Set torch, torch-GPU, and numpy random number generator.
	torch.manual_seed(seed)
	np.random.seed(seed)

	if gpu:
		torch.cuda.manual_seed_all(seed)

	# Initialize the spiking neural network.
	network = ETH(seed, n_input, n_neurons, (n_train, n_test), dt, (nu_pre, nu_post), c_inhib, \
		(train_time, train_rest, test_time, test_rest), (tc_pre, tc_post), update_interval, wmax)

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

	# Keep track of correct classifications for performance monitoring.
	correct = { scheme : 0 for scheme in network.voting_schemes }

	# Run training phase.
	plt.ion()
	start = timeit.default_timer()
	for idx, (image, target) in enumerate(zip(train_X, train_y)):
		if idx > 0 and idx % network.update_interval == 0:
			# Assign labels to neurons based on network spiking activity.
			network.assign_labels(train_y[idx - network.update_interval : idx], spike_monitor)

			print()

			# Assess performance of network on last `update_interval` examples.
			for scheme in network.performances.keys():
				network.performances[scheme].append(correct[scheme] / update_interval)  # Calculate percent correctly classified.
				correct[scheme] = 0  # Reset number of correct examples.
				print(scheme, ':', network.performances[scheme])

			print()

		# Print progress through training data.
		if idx % 10 == 0:
			print('Training progress: (%d / %d) - Elapsed time: %.4f' % (idx, len(train_X), timeit.default_timer() - start))
			start = timeit.default_timer()

		# Run network on image for `train_time` after transforming it into Poisson spike trains.
		inpt = generate_spike_train(image, network.intensity, network.sim_times['train_time'])
		if plot:
			spikes, voltages, traces = network.run(mode='train', inpt=inpt, time=network.sim_times['train_time'])
		else:
			spikes = network.run(mode='train', inpt=inpt, time=network.sim_times['train_time'])

		# Classify network output (spikes) based on historical spiking activity.
		predictions = network.classify(spikes['Ae'])

		# If correct, increment counter variable.
		for scheme in predictions.keys():
			if predictions[scheme][0] == target[0]:
				correct[scheme] += 1

		# Run zero image on network for `rest_time`.
		if plot:
			rest_spikes, rest_voltages, rest_traces = network.run(mode='train', inpt=zero_data, time=network.sim_times['train_rest'])
		else:
			rest_spikes = network.run(mode='train', inpt=zero_data, time=network.sim_times['train_rest'])

		# Concatenate image and rest network data for plotting purposes.
		if plot:
			spikes = { pop : torch.cat([spikes[pop], rest_spikes[pop]]) for pop in network.populations }
			voltages = { pop : torch.cat([voltages[pop], rest_voltages[pop]]) for pop in network.populations }
			traces = { pop : torch.cat([traces[pop], rest_traces[pop]]) for pop in ['X', 'Ae'] }

		# Add spikes from this iteration to the spike monitor
		spike_monitor[idx % network.update_interval] = torch.sum(spikes['Ae'], 0)

		# Optionally plot the excitatory, inhibitory spiking.
		if plot:
			if idx == 0:
				# Create figure for input image and corresponding spike trains.
				input_figure, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=(12, 6))
				im0 = ax0.imshow(image.cpu().numpy().reshape(network.n_input_sqrt, network.n_input_sqrt), cmap='binary')
				ax0.set_title('Original MNIST digit (Iteration %d)' % idx)
				im1 = ax1.imshow(torch.sum(inpt, 0).cpu().numpy().reshape(network.n_input_sqrt, network.n_input_sqrt), cmap='binary')
				ax1.set_title('Sum of spike trains')
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
											network.n_neurons_sqrt]).T, cmap=color, vmin=-1.5, vmax=9.5)
				ax6.set_title('Neuron labels')

				div5 = make_axes_locatable(ax5)
				div6 = make_axes_locatable(ax6)
				cax5 = div5.append_axes("right", size="5%", pad=0.05)
				cax6 = div6.append_axes("right", size="5%", pad=0.05)

				plt.colorbar(im5, cax=cax5)
				plt.colorbar(im6, cax=cax6, ticks=np.arange(-1, 10))

				plt.tight_layout()

				# Create figure to display neuron voltages over the iteration.
				# voltages_figure, [ax7, ax8] = plt.subplots(2, figsize=(10, 5))
				# ax7.plot(voltages['Ae'].cpu().numpy())
				# ax8.plot(voltages['Ai'].cpu().numpy())

				# plt.tight_layout()

				# # Create for displaying synaptic traces over the iteration.
				# voltages_figure, [ax9, ax10] = plt.subplots(2, figsize=(10, 5))
				# ax9.plot(network.a['X'].cpu().numpy())
				# ax10.plot(network.a['Ae'].cpu().numpy())
			else:
				# Reset image data after each iteration.
				im0.set_data(image.cpu().numpy().reshape(network.n_input_sqrt, network.n_input_sqrt))
				im1.set_data(torch.sum(inpt, 0).cpu().numpy().reshape(network.n_input_sqrt, network.n_input_sqrt))
				im2.set_data(inpt.cpu().numpy().T)
				im3.set_data(spikes['Ae'].cpu().numpy().T)
				im4.set_data(spikes['Ai'].cpu().numpy().T)
				im5.set_data(network.get_square_weights())
				im6.set_data(network.assignments.cpu().numpy().reshape([network.n_neurons_sqrt, network.n_neurons_sqrt]).T)
				
				# ax7.clear(); ax8.clear(); # ax9.clear(); ax10.clear()
				# ax7.plot(voltages['Ae'].cpu().numpy())
				# ax8.plot(voltages['Ai'].cpu().numpy())
				# ax9.plot(traces['X'].cpu().numpy())
				# ax10.plot(traces['Ae'].cpu().numpy())

				# Update title of input digit plot to reflect current iteration.
				ax0.set_title('Original MNIST digit (Iteration %d)' % idx)
			
			plt.pause(1e-8)

	for scheme in network.voting_schemes:
		print('Training accuracy for voting scheme %s:' % scheme, correct[scheme] / n_train)
