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

from .util import *
from .groups import *
from .synapses import *
from .network import *

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


parser = argparse.ArgumentParser(description='ETH (with LIF neurons) SNN toy model simulation implemented with PyTorch.')
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
plot = plot == 'True'

# Set random number generator.
np.random.seed(seed)

# Initialize the spiking neural network.
network = Network()
network.add_group(InputGroup(n_input), 'X')
network.add_group(AdaptiveLIFGroup(n_neurons), 'Ae')
network.add_group(AdaptiveLIFGroup(n_neurons), 'Ai')
network.add_synapses(STDPSynapses(network.groups['X'], network.groups['Ae']), name='X_Ae')
network.add_synapses(Synapses(network.groups['Ae'], network.groups['Ai'], w=torch.diag(22.5 * torch.ones(n_neurons))), name='Ae_Ai')
network.add_synapses(Synapses(network.groups['Ai'], network.groups['Ae'], w=c_inhib * torch.ones([n_neurons, \
											n_neurons]) - torch.diag(c_inhib * torch.ones(n_neurons))), name='Ai_Ae')

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

image_time = time
rest_time = rest

# Special "zero data" used for network rest period between examples.
zero_data = np.zeros([rest_time, n_input])

# Run network simulation.
plt.ion()
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
			input_figure, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=(12, 6))
			im0 = ax0.imshow(image.reshape(network.n_input_sqrt, network.n_input_sqrt), cmap='binary')
			ax0.set_title('Original MNIST digit (Iteration %d)' % idx)
			im1 = ax1.imshow(np.sum(inpt, axis=0).reshape(network.n_input_sqrt, network.n_input_sqrt), cmap='binary')
			ax1.set_title('Sum of spike trains')
			im2 = ax2.imshow(inpt.T, cmap='binary')
			ax2.set_title('Poisson spiking representation')

			plt.tight_layout()

			# Create figure for excitatory and inhibitory neuron populations.
			spike_figure, [ax3, ax4] = plt.subplots(2, figsize=(10, 5))
			im3 = ax3.imshow(spikes['Ae'].T, cmap='binary')
			ax3.set_title('Excitatory spikes')
			im4 = ax4.imshow(spikes['Ai'].T, cmap='binary')
			ax4.set_title('Inhibitory spikes')

			plt.tight_layout()

			# Create figure for input to excitatory weights and excitatory neuron assignments.
			weights_figure, [ax5, ax6] = plt.subplots(1, 2, figsize=(10, 6))
			square_weights = get_square_weights(network.get_weights(), network.n_input_sqrt, network.n_neurons_sqrt)

			im5 = ax5.imshow(square_weights, cmap='hot_r', vmin=0, vmax=network.wmax)
			ax5.set_title('Input to excitatory weights')
			
			color = plt.get_cmap('RdBu', 11)
			assignments = network.get_assignments().reshape([network.n_neurons_sqrt, network.n_neurons_sqrt]).T
			im6 = ax6.matshow(assignments, cmap=color, vmin=-1.5, vmax=9.5)
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
				for scheme in network.voting_schemes:
					ax11.plot(range(len(network.performances[scheme])), network.performances[scheme], label=scheme)

				ax11.set_xlim([0, n_train / update_interval + 1])
				ax11.set_ylim([0, 1])
				ax11.set_title('Network performance')
				ax11.legend()
		else:
			# Re-draw plotting data after each iteration.
			im0.set_data(image.reshape(network.n_input_sqrt, network.n_input_sqrt))
			im1.set_data(np.sum(inpt, axis=0).reshape(network.n_input_sqrt, network.n_input_sqrt))
			im2.set_data(inpt.T)
			im3.set_data(spikes['Ae'].T)
			im4.set_data(spikes['Ai'].T)

			square_weights = get_square_weights(network.get_weights(), network.n_input_sqrt, network.n_neurons_sqrt)
			
			im5.set_data(square_weights)

			assignments = network.get_assignments().reshape([network.n_neurons_sqrt, network.n_neurons_sqrt]).T
			im6.set_data(assignments)

			if mode == 'train':
				ax11.clear()
				for scheme in network.voting_schemes:
					ax11.plot(range(len(network.performances[scheme])), network.performances[scheme], label=scheme)

				ax11.set_xlim([0, n_train / update_interval])
				ax11.set_ylim([0, 1])
				ax11.set_title('Network performance')
				ax11.legend()

			# Update title of input digit plot to reflect current iteration.
			ax0.set_title('Original MNIST digit (Iteration %d)' % idx)
		
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
	save_params(network.get_weights(), '.'.join(['_'.join(['X_Ae', network.fname]), 'npy']))
	save_params(network.get_theta(), '.'.join(['_'.join(['theta', network.fname]), 'npy']))
	save_assignments(network.get_assignments(), '.'.join(['_'.join(['assignments', network.fname]), 'npy']))

if mode == 'test':
	results = pd.DataFrame([ [ network.fname ] + list(results.values()) ], \
								columns=[ 'Parameters' ] + list(results.keys()))

	results_fname = '_'.join([str(n_neurons), str(n_train), 'results.csv'])
	if not results_fname in os.listdir(results_path):
		results.to_csv(os.path.join(results_path, results_fname), index=False)
	else:
		all_results = pd.read_csv(os.path.join(results_path, results_fname))
		all_results = pd.concat([all_results, results], ignore_index=True)
		all_results.to_csv(os.path.join(results_path, results_fname), index=False)
