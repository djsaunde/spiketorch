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
from spiketorch.groups import *
from spiketorch.synapses import *
from spiketorch.network import *

data_path = os.path.join('..', '..', 'data')
params_path = os.path.join('..', '..', 'params')
assign_path = os.path.join('..', '..', 'assignments')
results_path = os.path.join('..', '..', 'results')

for path in [ params_path, assign_path, results_path ]:
	if not os.path.isdir(path):
		os.makedirs(path)

np.set_printoptions(threshold=np.nan, linewidth=200)
np.warnings.filterwarnings('ignore')
torch.set_printoptions(threshold=np.nan, linewidth=100, edgeitems=10)


def classify(spikes, voting_schemes, assignments):
	'''
	Given the neuron assignments and the network spiking
	activity, make predictions about the data targets.
	'''
	spikes = spikes.sum(0)

	predictions = {}
	for scheme in voting_schemes:
		rates = torch.zeros(10)

		if scheme == 'all':
			for idx in range(10):
				n_assigns = torch.nonzero(assignments == idx).numel()
				
				if n_assigns > 0:
					idxs = torch.nonzero((assignments == idx).long().view(-1)).view(-1)
					rates[idx] = torch.sum(spikes[idxs]) / n_assigns

		predictions[scheme] = torch.sort(rates, dim=0, descending=True)[1]

	return predictions


def assign_labels(inputs, outputs, rates, assignments):
	'''
	Given the excitatory neuron firing history, assign them class labels.
	'''
	if gpu:
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
			rates[:, j] = 0.9 * rates[:, j] + torch.sum(outputs[idxs], 0) / n_inputs

	# Assignments of neurons are the categories for which they fire the most. 
	assignments = torch.max(self.rates, 1)[1]

	return rates, assignments


parser = argparse.ArgumentParser(description='ETH (with LIF neurons) \
					SNN toy model simulation implemented with PyTorch.')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--n_input', type=int, default=784)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--n_train', type=int, default=10000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--update_interval', type=int, default=250)
parser.add_argument('--dt', type=float, default=1)
parser.add_argument('--nu_pre', type=float, default=1e-4)
parser.add_argument('--nu_post', type=float, default=1e-2)
parser.add_argument('--c_inhib', type=float, default=-17.4)
parser.add_argument('--time', type=int, default=350)
parser.add_argument('--rest', type=int, default=150)
parser.add_argument('--trace_tc', type=int, default=5e-2)
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
plot = plot == 'True'

# Set random number generator.
np.random.seed(seed)

# Record decaying spike traces to use STDP.
if mode == 'train':
	traces = True

# Build filename from command-line arguments.
fname = '_'.join([ str(n_neurons), str(n_train), str(seed) ])

# Initialize the spiking neural network.
network = Network()

# Add neuron populations.
network.add_group(InputGroup(n_input, traces=traces), 'X')
network.add_group(AdaptiveLIFGroup(n_neurons, traces=traces, rest=-65.0, reset=-65.0,
			threshold=-52.0, voltage_decay=1e-2, refractory=5, trace_tc=trace_tc), 'Ae')
network.add_group(LIFGroup(n_neurons, traces=traces, rest=-60.0, reset=-45.0,
		threshold=-40.0, voltage_decay=1e-1, refractory=2, trace_tc=trace_tc), 'Ai')

# Add synaptic connections between populations
network.add_synapses(STDPSynapses(network.groups['X'], network.groups['Ae'],
				wmax=wmax, nu_pre=nu_pre, nu_post=nu_post), name=('X', 'Ae'))
network.add_synapses(Synapses(network.groups['Ae'], network.groups['Ai'], 
					w=torch.diag(22.5 * torch.ones(n_neurons))), name=('Ae', 'Ai'))
network.add_synapses(Synapses(network.groups['Ai'], network.groups['Ae'], w=c_inhib * \
							torch.ones([n_neurons, n_neurons]) - torch.diag(c_inhib \
											* torch.ones(n_neurons))), name=('Ai', 'Ae'))

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
spike_monitor = np.zeros([update_interval, n_neurons])

# Network simulation times.
image_time = time
rest_time = rest

# Voting schemes and neuron label assignments.
voting_schemes = ['all']
assignments = -1 * torch.ones(n_neurons)
rates = torch.zeros(n_neurons, 10)
performances = { scheme : [] for scheme in voting_schemes }

# Keep track of correct classifications for performance monitoring.
correct = { scheme : 0 for scheme in voting_schemes }
total_correct = { scheme : 0 for scheme in voting_schemes }

# Pre-calculated values.
n_input_sqrt = int(np.sqrt(n_input))
n_neurons_sqrt = int(np.sqrt(n_neurons))

# Run network simulation.
plt.ion()
start = timeit.default_timer()
iter_start = timeit.default_timer()

if mode == 'train':
	n_samples = n_train
elif mode == 'test':
	n_samples = n_test

n_images = X.shape[0]

intensity = 1
for idx in range(n_samples):
	image, target = X[idx % n_images], y[idx % n_images]

	if mode == 'train':
		if idx > 0 and idx % update_interval == 0:
			# Assign labels to neurons based on network spiking activity.
			assign_labels(y[(idx % n_images) - update_interval : idx % n_images], spike_monitor, rates, assignments)

			# Assess performance of network on last `update_interval` examples.
			print()
			for scheme in performances.keys():
				performances[scheme].append(correct[scheme] / update_interval)  # Calculate percent correctly classified.
				correct[scheme] = 0  # Reset number of correct examples.
				print(scheme, ':', network.performances[scheme])

				# Save best accuracy.
				if performances[scheme][-1] > best_accuracy:
					best_accuracy = performances[scheme][-1]

					if gpu:
						weights = network.get_weights(('X', 'Ae')).cpu().numpy()
						theta = network.get_theta('Ae').cpu().numpy()
						asgnmts = assignments.cpu().numpy()
					else:
						weights = network.get_weights(('X', 'Ae')).numpy()
						theta = network.get_theta('Ae').numpy()
						asgnmts = assignments.numpy()

					save_params(weights, '.'.join(['_'.join(['X_Ae', fname]), 'npy']))
					save_params(theta, '.'.join(['_'.join(['theta', fname]), 'npy']))
					save_assignments(asgnmts, '.'.join(['_'.join(['assignments', fname]), 'npy']))

			print()

	# Print progress through dataset.
	if idx % 10 == 0:
		if mode == 'train':
			print('Training progress: (%d / %d) - Elapsed time: %.4f' % (idx, n_train, timeit.default_timer() - start))
		elif mode == 'test':
			print('Test progress: (%d / %d) - Elapsed time: %.4f' % (idx, n_test, timeit.default_timer() - start))

		start = timeit.default_timer()

	inpts = {}

	# Encode current input example as Poisson spike trains.
	inpts['X'] = torch.from_numpy(generate_spike_train(image, intensity, image_time))

	# Run network on Poisson-encoded image data.
	spikes = network.run(mode, inpts, image_time)

	# Re-run image if there isn't any network activity.
	n_retries = 0
	while np.count_nonzero(torch.nonzero(spikes['Ae']).size()) < 5 and n_retries < 3:
		intensity += 1; n_retries += 1
		inpts['X'] = torch.from_numpy(generate_spike_train(image, intensity, image_time))
		spikes = network.run(mode=mode, inpts=inpts, time=image_time)

	# Reset input intensity after any retries.
	intensity = 1

	# Classify network output (spikes) based on historical spiking activity.
	predictions = classify(spikes['Ae'], voting_schemes, assignments)

	# If correct, increment counter variable.
	for scheme in predictions.keys():
		if predictions[scheme][0] == target[0]:
			correct[scheme] += 1
			total_correct[scheme] += 1

	# Run zero image on network for `rest_time`.
	network.reset()

	# Add spikes from this iteration to the spike monitor
	spike_monitor[idx % update_interval] = torch.sum(spikes['Ae'], 0).numpy()

	# Optionally plot the excitatory, inhibitory spiking.
	if plot:
		if gpu:
			inpt = inpts['X'].cpu().numpy()
			Ae_spikes = spikes['Ae'].cpu().numpy()
			Ai_spikes = spikes['Ai'].cpu().numpy()
			input_exc_weights = network.synapses[('X', 'Ae')].w.cpu().numpy()
			asgnmts = assignments.cpu().numpy()
			Ae_voltages = network.groups['Ae'].get_voltages().cpu().numpy()
		else:
			inpt = inpts['X'].numpy()
			Ae_spikes = spikes['Ae'].numpy()
			Ai_spikes = spikes['Ai'].numpy()
			input_exc_weights = network.synapses[('X', 'Ae')].w.numpy()
			asgnmts = assignments.numpy()
			Ae_voltages = network.groups['Ae'].get_voltages().numpy()

		if idx == 0:
			# Create figure for input image and corresponding spike trains.
			input_figure, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=(12, 6))
			im0 = ax0.imshow(image.reshape(n_input_sqrt, n_input_sqrt), cmap='binary')
			ax0.set_title('Original MNIST digit (Iteration %d)' % idx)
			im1 = ax1.imshow(np.sum(inpt, axis=0).reshape(n_input_sqrt, n_input_sqrt), cmap='binary')
			ax1.set_title('Sum of spike trains')
			im2 = ax2.imshow(inpt.T, cmap='binary')
			ax2.set_title('Poisson spiking representation')

			plt.tight_layout()

			# Create figure for excitatory and inhibitory neuron populations.
			spike_figure, [ax3, ax4] = plt.subplots(2, figsize=(10, 5))
			im3 = ax3.imshow(Ae_spikes.T, cmap='binary')
			ax3.set_title('Excitatory spikes')
			im4 = ax4.imshow(Ai_spikes.T, cmap='binary')
			ax4.set_title('Inhibitory spikes')

			plt.tight_layout()

			# Create figure for input to excitatory weights and excitatory neuron assignments.
			weights_figure, [ax5, ax6] = plt.subplots(1, 2, figsize=(10, 6))
			square_weights = get_square_weights(input_exc_weights, n_input_sqrt, n_neurons_sqrt)

			im5 = ax5.imshow(square_weights, cmap='hot_r', vmin=0, vmax=wmax)
			ax5.set_title('Input to excitatory weights')
			
			color = plt.get_cmap('RdBu', 11)
			asgnmts = asgnmts.reshape([n_neurons_sqrt, n_neurons_sqrt]).T
			im6 = ax6.matshow(asgnmts, cmap=color, vmin=-1.5, vmax=9.5)
			ax6.set_title('Neuron labels')

			div5 = make_axes_locatable(ax5)
			div6 = make_axes_locatable(ax6)
			cax5 = div5.append_axes("right", size="5%", pad=0.05)
			cax6 = div6.append_axes("right", size="5%", pad=0.05)

			plt.colorbar(im5, cax=cax5)
			plt.colorbar(im6, cax=cax6, ticks=np.arange(-1, 10))

			plt.tight_layout()

			# Create figure to display plots of training accuracy over time.
			if mode == 'train':
				perf_figure, ax11 = plt.subplots()
				for scheme in voting_schemes:
					ax11.plot(range(len(performances[scheme])), performances[scheme], label=scheme)

				ax11.set_xlim([0, n_train / update_interval + 1])
				ax11.set_ylim([0, 1])
				ax11.set_title('Network performance')
				ax11.legend()
		else:
			# Re-draw plotting data after each iteration.
			im0.set_data(image.reshape(n_input_sqrt, n_input_sqrt))
			im1.set_data(np.sum(inpt, axis=0).reshape(n_input_sqrt, n_input_sqrt))
			im2.set_data(inpt.T)
			im3.set_data(Ae_spikes.T)
			im4.set_data(Ai_spikes.T)

			square_weights = get_square_weights(input_exc_weights, n_input_sqrt, n_neurons_sqrt)
			
			im5.set_data(square_weights)

			asgnmts = asgnmts.reshape([n_neurons_sqrt, n_neurons_sqrt]).T
			im6.set_data(asgnmts)

			if mode == 'train':
				ax11.clear()
				for scheme in voting_schemes:
					ax11.plot(range(len(performances[scheme])), performances[scheme], label=scheme)

				ax11.set_xlim([0, n_train / update_interval])
				ax11.set_ylim([0, 1])
				ax11.set_title('Network performance')
				ax11.legend()

			# Update title of input digit plot to reflect current iteration.
			ax0.set_title('Original MNIST digit (Iteration %d)' % idx)
		
		plt.pause(1e-8)

results = {}
for scheme in voting_schemes:
	if mode == 'train':
		results[scheme] = 100 * total_correct[scheme] / n_train
		print('Training accuracy for voting scheme %s:' % scheme, results[scheme])
	elif mode == 'test':
		results[scheme] = 100 * total_correct[scheme] / n_test
		print('Test accuracy for voting scheme %s:' % scheme, results[scheme])

# Save out network parameters and assignments for the test phase.
if mode == 'train':
	save_params(network.get_weights(), '.'.join(['_'.join(['X_Ae', fname]), 'npy']))
	save_params(network.get_theta(), '.'.join(['_'.join(['theta', fname]), 'npy']))
	save_assignments(network.get_assignments(), '.'.join(['_'.join(['assignments', fname]), 'npy']))

if mode == 'test':
	results = pd.DataFrame([ [ fname ] + list(results.values()) ], \
								columns=[ 'Parameters' ] + list(results.keys()))

	results_fname = '_'.join([str(n_neurons), str(n_train), 'results.csv'])
	if not results_fname in os.listdir(results_path):
		results.to_csv(os.path.join(results_path, results_fname), index=False)
	else:
		all_results = pd.read_csv(os.path.join(results_path, results_fname))
		all_results = pd.concat([all_results, results], ignore_index=True)
		all_results.to_csv(os.path.join(results_path, results_fname), index=False)
