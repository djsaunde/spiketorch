import os
import sys
import time
import torch
import timeit
import argparse
import numpy as np
import pickle as p
import pandas as pd
import matplotlib.pyplot as plt

from struct import unpack
from torchvision import datasets
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spiketorch.util import *

model_name = 'random_inhibition'

data_path = os.path.join('..', 'data', model_name)
params_path = os.path.join('..', 'params', model_name)
assign_path = os.path.join('..', 'assignments', model_name)
results_path = os.path.join('..', 'results', model_name)

for path in [ params_path, assign_path, results_path ]:
	if not os.path.isdir(path):
		os.makedirs(path)

np.set_printoptions(threshold=np.nan, linewidth=200)
np.warnings.filterwarnings('ignore')
torch.set_printoptions(threshold=np.nan, linewidth=100, edgeitems=10)


class SNN:
	'''
	Replication of the spiking neural network model from "Unsupervised learning of digit
	recognition using spike-timing-dependent plasticity"
	(https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#).
	'''
	def __init__(self, seed=0, mode='train', n_input=784, n_exc=100, n_inh=25, n_examples=(10000, 10000), dt=1, 
													lrs=(1e-4, 1e-2), c_inhib=17.4, sim_times=(350, 150, 350, 150),
										stdp_times=(20, 20), update_interval=100, wmax=1.0, gpu='True'):
		'''
		Constructs the network based on chosen parameters.

		Arguments:
			- seed: Sets the random number generator sequence.
			- n_exc: Number of excitatory neurons.
			- n_inh: Number of inhibitory neurons.
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
		self.n_exc = n_exc
		self.n_exc_sqrt = int(np.sqrt(n_exc))
		self.n_inh = n_inh
		self.n_examples = n_examples
		self.dt = dt
		self.lrs = { 'nu_pre' : lrs[0], 'nu_post' : lrs[1] }
		self.c_inhib = c_inhib
		self.sim_times = { 'train_time' : sim_times[0], 'train_rest' : sim_times[1], \
							'test_time' : sim_times[2], 'test_rest' : sim_times[3] }
		self.stdp_times = { 'X' : dt / stdp_times[0], 'Ae' : dt / stdp_times[1] }
		self.gpu = gpu == 'True' and torch.cuda.is_available()

		torch.manual_seed(seed)

		if self.gpu:
			torch.set_default_tensor_type('torch.cuda.FloatTensor')
			torch.cuda.manual_seed_all(seed)

		# Generic filename for saving out weights and other parameters. 
		self.fname = '_'.join([ str(n_exc), str(n_inh), str(n_examples[0]), str(seed) ])

		# Population names.
		self.populations = ['Ae', 'Ai']
		self.stdp_populations = ['X', 'Ae']

		# Assignments and performance monitoring update interval.
		self.update_interval = update_interval

		# Excitatory neuron assignments.
		if mode == 'train':
			self.assignments = -1 * torch.ones(n_exc)
		elif mode == 'test':
			self.assignments = torch.from_numpy(load_assignments(model_name, self.fname)).cuda()

		# Instantiate weight matrices.
		if mode == 'train':
			self.W = {}
			self.W['X_Ae'] = (torch.rand(n_input, n_exc) + 0.01) * 0.3
			
			neurons = np.random.multinomial(n_exc, [1 / n_inh] * n_inh)
			onehot = np.zeros([neurons.size, neurons.max() + 1])
			onehot[np.arange(neurons.size), neurons] = 1

			if gpu:
				self.W['Ae_Ai'] = 22.5 * torch.from_numpy(onehot).cuda()
			else:
				self.W['Ae_Ai'] = 22.5 * torch.from_numpy(onehot)

			self.W['Ai_Ae'] = c_inhib * torch.bernoulli((4 / n_exc) * torch.ones([n_inh, n_exc]))
		
		elif mode == 'test':
			self.W = { 'X_Ae' : torch.from_numpy(load_params(model_name, self.fname, 'X_Ae')).cuda(),
						'Ae_Ai' : 22.5 * torch.ones([n_exc, n_inh]),
						'Ai_Ae' : c_inhib * torch.bernoulli((4 / n_exc) * torch.ones([n_inh, n_exc])) }

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
		self.rates = torch.zeros([self.n_exc, 10])
		# Etc.
		self.intensity = 1
		self.wmax = wmax
		self.norm = 78.0 * wmax

		# Instantiate neuron state variables.
		# Neuron voltages.
		self.v = { 'Ae' : self.rest['Ae'] * torch.ones(n_exc), 'Ai' : self.rest['Ai'] * torch.ones(n_inh) }
		# Spike occurrences.
		self.s = { 'X' : torch.zeros(n_input), 'Ae' : torch.zeros(n_exc), 'Ai' : torch.zeros(n_inh) }
		# Synaptic traces (used for STDP calculations).
		self.a = { 'X' : torch.zeros(n_input), 'Ae' : torch.zeros(n_exc) }
		
		# Adaptive additive threshold parameters (used in excitatory layer).
		if mode == 'train':
			self.theta = torch.zeros(n_exc)
		elif mode == 'test':
			if gpu:
				self.theta = torch.from_numpy(load_params(model_name, self.fname, 'theta')).cuda()
			else:
				self.theta = torch.from_numpy(load_params(model_name, self.fname, 'theta'))

		# Refractory period counters.
		self.refrac_count = { 'Ae' : torch.zeros(n_exc), 'Ai' : torch.zeros(n_inh) }


	def run(self, mode, inpt, time):
		'''
		Runs the network on a single input for some time.

		Arguments:
			- mode (str): Whether we are in test or training mode.
					Affects whether to adaptive network parameters. 
			- inpt (numpy.ndarray): Network input, encoded as Poisson
					spike trains. Has shape (time, self.n_input).
			- time (int): How many simulation time steps to run.

		Returns:
			State variables recorded over the simulation iteration.
		'''
		# Convert input numpy.ndarray to torch.Tensor
		if self.gpu:
			inpt = torch.from_numpy(inpt).cuda()
		else:
			inpt = torch.from_numpy(inpt)

		# Records network state variables for plotting purposes.
		spikes = { 'Ae' : torch.zeros([time, self.n_exc]).byte(), 'Ai' : torch.zeros([time, self.n_inh]).byte() }

		# Run simulation for `time` simulation steps.
		for timestep in range(time):
			# Get input spikes for this timestep.
			self.s['X'] = inpt[timestep, :]

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

			# Reset neurons above their threshold voltage.
			self.v['Ae'][self.s['Ae']] = self.reset['Ae']
			self.v['Ai'][self.s['Ai']] = self.reset['Ai']

			# Integrate input and decay voltages.
			self.v['Ae'] += self.s['X'].float() @ self.W['X_Ae'] - self.s['Ai'].float() @ self.W['Ai_Ae']
			self.v['Ae'] -= self.v_decay['Ae'] * (self.v['Ae'] - self.rest['Ae'])
			self.v['Ai'] += self.s['Ae'].float() @ self.W['Ae_Ai']
			self.v['Ai'] -= self.v_decay['Ai'] * (self.v['Ai'] - self.rest['Ai'])

			if mode == 'train':
				# Setting synaptic traces.
				self.a['X'][self.s['X'].byte()] = 1.0
				self.a['Ae'][self.s['Ae'].byte()] = 1.0
				
				# Perform STDP weight update.
				# Post-synaptic.
				self.W['X_Ae'] += self.lrs['nu_post'] * (self.a['X'].view(self.n_input, 1) * self.s['Ae'].float().view(1, self.n_exc))
				# Pre-synaptic.
				self.W['X_Ae'] -= self.lrs['nu_pre'] * (self.s['X'].float().view(self.n_input, 1) * self.a['Ae'].view(1, self.n_exc))

				# Ensure that weights are within [0, self.wmax].
				self.W['X_Ae'] = torch.clamp(self.W['X_Ae'], 0, self.wmax)

				# Decay synaptic traces.
				self.a['X'] -= self.stdp_times['X'] * self.a['X']
				self.a['Ae'] -= self.stdp_times['Ae'] * self.a['Ae']
			
				# Decay adaptive thresholds.
				self.theta -= self.theta_decay * self.theta

		# Normalize weights after one iteration.
		self.normalize_weights()

		# Return excitatory spiking activity.
		if self.gpu:
			return { pop : spikes[pop].cpu().numpy() for pop in self.populations }
		else:
			return { pop : spikes[pop].numpy() for pop in self.populations }

	
	def _reset(self):
		'''
		Reset relevant state variables after a single iteration.
		'''
		# Voltages.
		self.v['Ae'][:] = self.rest['Ae']
		self.v['Ai'][:] = self.rest['Ai']

		# Synaptic traces.
		self.a['X'][:] = 0
		self.a['Ae'][:] = 0


	def get_weights(self):
		if self.gpu:
			return self.W['X_Ae'].cpu().numpy()
		else:
			return self.W['X_Ae'].numpy()


	def get_theta(self):
		if self.gpu:
			return self.theta.cpu().numpy()
		else:
			return self.theta.numpy()


	def get_assignments(self):
		if self.gpu:
			return self.assignments.cpu().numpy()
		else:
			return self.assignments.numpy()


	def normalize_weights(self):
		'''
		Normalize weights on synpases from input to excitatory layer.
		'''
		self.W['X_Ae'] *= self.norm / self.W['X_Ae'].sum(0).view(1, -1)


	def assign_labels(self, inputs, outputs):
		'''
		Given the excitatory neuron firing history, assign them class labels.
		'''
		if self.gpu:
			inputs = torch.from_numpy(inputs).cuda()
			outputs = torch.from_numpy(outputs).cuda()
		else:
			inputs = torch.from_numpy(inputs)
			outputs = torch.from_numpy(outputs)

		outputs = outputs.float()

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
		if self.gpu:
			spikes = torch.from_numpy(spikes).cuda()
		else:
			spikes = torch.from_numpy(spikes)

		spikes = spikes.sum(0)

		predictions = {}
		for scheme in self.voting_schemes:
			rates = torch.zeros(10)

			if scheme == 'all':
				for idx in range(10):
					n_assigns = torch.nonzero(self.assignments == idx).numel()
					
					if n_assigns > 0:
						idxs = torch.nonzero((self.assignments == idx).long().view(-1)).view(-1)
						rates[idx] = torch.sum(spikes[idxs]) / n_assigns

			predictions[scheme] = torch.sort(rates, dim=0, descending=True)[1]

		return predictions


if __name__ =='__main__':
	parser = argparse.ArgumentParser(description='SNN toy model simulation implemented with PyTorch.')
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--mode', type=str, default='train')
	parser.add_argument('--n_input', type=int, default=784)
	parser.add_argument('--n_exc', type=int, default=100)
	parser.add_argument('--n_inh', type=int, default=25)
	parser.add_argument('--n_train', type=int, default=10000)
	parser.add_argument('--n_test', type=int, default=10000)
	parser.add_argument('--update_interval', type=int, default=250)
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
	plot = plot == 'True'

	# Set random number generator.
	np.random.seed(seed)

	# Initialize the spiking neural network.
	network = SNN(seed, mode, n_input, n_exc, n_inh, (n_train, n_test), dt, (nu_pre, nu_post), c_inhib, \
		(train_time, train_rest, test_time, test_rest), (tc_pre, tc_post), update_interval, wmax, gpu)

	# Get training, test data from disk.
	if mode == 'train':
		data = get_labeled_data('train', train=True)
	elif mode == 'test':
		data = get_labeled_data('test', train=False)

	# Convert data into torch Tensors.
	if mode == 'train':
		X, y = data['X'], data['y']
	elif mode == 'test':
		X, y = data['X'], data['y']

	# Count spikes from each neuron on each example (between update intervals).
	spike_monitor = np.zeros([network.update_interval, network.n_exc])

	# Keep track of correct classifications for performance monitoring.
	correct = { scheme : 0 for scheme in network.voting_schemes }
	total_correct = { scheme : 0 for scheme in network.voting_schemes }

	if mode == 'train':
		image_time = network.sim_times['train_time']
		rest_time = network.sim_times['train_rest']
	elif mode == 'test':
		image_time = network.sim_times['test_time']
		rest_time = network.sim_times['test_rest']

	# Special "zero data" used for network rest period between examples.
	zero_data = np.zeros([rest_time, n_input])

	# Run network simulation.
	plt.ion()
	best_accuracy = 0
	start = timeit.default_timer()
	iter_start = timeit.default_timer()
	
	if mode == 'train':
		n_samples = n_train
	elif mode == 'test':
		n_samples = n_test

	n_images = X.shape[0]

	for idx in range(n_samples):
		image, target = X[idx % n_images], y[idx % n_images]

		if mode == 'train':
			if idx > 0 and idx % network.update_interval == 0:
				# Assign labels to neurons based on network spiking activity.
				network.assign_labels(y[(idx % n_images) - network.update_interval : idx % n_images], spike_monitor)

				# Assess performance of network on last `update_interval` examples.
				print()
				for scheme in network.performances.keys():
					network.performances[scheme].append(correct[scheme] / update_interval)  # Calculate percent correctly classified.
					correct[scheme] = 0  # Reset number of correct examples.
					print(scheme, ':', network.performances[scheme])

					# Save best accuracy.
					if network.performances[scheme][-1] > best_accuracy:
						best_accuracy = network.performances[scheme][-1]
						save_params(model_name, network.get_weights(), network.fname, 'X_Ae')
						save_params(model_name, network.get_theta(), network.fname, 'theta')
						save_assignments(network.get_assignments(), '.'.join(['_'.join(['assignments', network.fname]), 'npy']))

				print()

		# Print progress through dataset.
		if idx % 10 == 0:
			if mode == 'train':
				print('Training progress: (%d / %d) - Elapsed time: %.4f' % (idx, n_train, timeit.default_timer() - start))
			elif mode == 'test':
				print('Test progress: (%d / %d) - Elapsed time: %.4f' % (idx, n_test, timeit.default_timer() - start))

			start = timeit.default_timer()

		# Encode current input example as Poisson spike trains. 
		inpt = generate_spike_train(image, network.intensity, image_time)

		# Run network on Poisson-encoded image data.
		spikes = network.run(mode=mode, inpt=inpt, time=image_time)

		# Re-run image if there isn't any network activity.
		n_retries = 0
		while np.count_nonzero(spikes['Ae']) < 5 and n_retries < 3:
			network.intensity += 1; n_retries += 1
			inpt = generate_spike_train(image, network.intensity, image_time)
			spikes = network.run(mode=mode, inpt=inpt, time=image_time)

		# Reset input intensity after any retries.
		network.intensity = 1

		# Classify network output (spikes) based on historical spiking activity.
		predictions = network.classify(spikes['Ae'])

		# If correct, increment counter variable.
		for scheme in predictions.keys():
			if predictions[scheme][0] == target[0]:
				correct[scheme] += 1
				total_correct[scheme] += 1

		# Run zero image on network for `rest_time`.
		network._reset()

		# Add spikes from this iteration to the spike monitor
		spike_monitor[idx % network.update_interval] = np.sum(spikes['Ae'], axis=0)

		# Optionally plot the excitatory, inhibitory spiking.
		if plot:
			if idx == 0:
				# Create figure for input image and corresponding spike trains.
				# input_figure, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=(12, 6))
				# im0 = ax0.imshow(image.reshape(network.n_input_sqrt, network.n_input_sqrt), cmap='binary')
				# ax0.set_title('Original MNIST digit (Iteration %d)' % idx)
				# im1 = ax1.imshow(np.sum(inpt, axis=0).reshape(network.n_input_sqrt, network.n_input_sqrt), cmap='binary')
				# ax1.set_title('Sum of spike trains')
				# im2 = ax2.imshow(inpt.T, cmap='binary')
				# ax2.set_title('Poisson spiking representation')

				# plt.tight_layout()

				# Create figure for excitatory and inhibitory neuron populations.
				spike_figure, [ax3, ax4] = plt.subplots(2, figsize=(10, 5))
				im3 = ax3.imshow(spikes['Ae'].T, cmap='binary')
				ax3.set_title('Excitatory spikes')
				im4 = ax4.imshow(spikes['Ai'].T, cmap='binary')
				ax4.set_title('Inhibitory spikes')

				plt.tight_layout()

				# Create figure for input to excitatory weights and excitatory neuron assignments.
				weights_figure, [ax5, ax6] = plt.subplots(1, 2, figsize=(10, 6))
				square_weights = get_square_weights(network.get_weights(), network.n_input_sqrt, network.n_exc_sqrt)

				im5 = ax5.imshow(square_weights, cmap='hot_r', vmin=0, vmax=network.wmax)
				ax5.set_title('Input to excitatory weights')
				
				color = plt.get_cmap('RdBu', 11)
				assignments = network.get_assignments().reshape([network.n_exc_sqrt, network.n_exc_sqrt]).T
				im6 = ax6.matshow(assignments, cmap=color, vmin=-1.5, vmax=9.5)
				ax6.set_title('Neuron labels')

				div5 = make_axes_locatable(ax5)
				div6 = make_axes_locatable(ax6)
				cax5 = div5.append_axes("right", size="5%", pad=0.05)
				cax6 = div6.append_axes("right", size="5%", pad=0.05)

				plt.colorbar(im5, cax=cax5)
				plt.colorbar(im6, cax=cax6, ticks=np.arange(-1, 10))

				plt.tight_layout()

				# # Create figure to display plots of training accuracy over time.
				# if mode == 'train':
				# 	perf_figure, ax11 = plt.subplots()
				# 	for scheme in network.voting_schemes:
				# 		ax11.plot(range(len(network.performances[scheme])), network.performances[scheme], label=scheme)

				# 	ax11.set_xlim([0, n_train / update_interval + 1])
				# 	ax11.set_ylim([0, 1])
				# 	ax11.set_title('Network performance')
				# 	ax11.legend()
			else:
				# Re-draw plotting data after each iteration.
				# im0.set_data(image.reshape(network.n_input_sqrt, network.n_input_sqrt))
				# im1.set_data(np.sum(inpt, axis=0).reshape(network.n_input_sqrt, network.n_input_sqrt))
				# im2.set_data(inpt.T)

				im3.set_data(spikes['Ae'].T)
				im4.set_data(spikes['Ai'].T)

				square_weights = get_square_weights(network.get_weights(), network.n_input_sqrt, network.n_exc_sqrt)
				
				im5.set_data(square_weights)

				assignments = network.get_assignments().reshape([network.n_exc_sqrt, network.n_exc_sqrt]).T
				im6.set_data(assignments)

				# if mode == 'train':
				# 	ax11.clear()
				# 	for scheme in network.voting_schemes:
				# 		ax11.plot(range(len(network.performances[scheme])), network.performances[scheme], label=scheme)

				# 	ax11.set_xlim([0, n_train / update_interval])
				# 	ax11.set_ylim([0, 1])
				# 	ax11.set_title('Network performance')
				# 	ax11.legend()

				# # Update title of input digit plot to reflect current iteration.
				# ax0.set_title('Original MNIST digit (Iteration %d)' % idx)
			
			plt.pause(1e-8)

	results = {}
	for scheme in network.voting_schemes:
		if mode == 'train':
			results[scheme] = 100 * total_correct[scheme] / n_train
			print('Training accuracy for voting scheme %s:' % scheme, results[scheme])
		elif mode == 'test':
			results[scheme] = 100 * total_correct[scheme] / n_test
			print('Test accuracy for voting scheme %s:' % scheme, results[scheme])

	# Save out network parameters and assignments for the test phase.
	if mode == 'train':
		save_params(model_name, network.get_weights(), network.fname, 'X_Ae')
		save_params(model_name, network.get_theta(), network.fname, 'theta')
		save_assignments(model_name, network.get_assignments(), network.fname)

	if mode == 'test':
		results = pd.DataFrame([ [ network.fname ] + list(results.values()) ], \
									columns=[ 'Parameters' ] + list(results.keys()))

		results_fname = '_'.join([str(n_exc), str(n_inh), str(n_train), 'results.csv'])
		if not results_fname in os.listdir(results_path):
			results.to_csv(os.path.join(results_path, results_fname), index=False)
		else:
			all_results = pd.read_csv(os.path.join(results_path, results_fname))
			all_results = pd.concat([all_results, results], ignore_index=True)
			all_results.to_csv(os.path.join(results_path, results_fname), index=False)
