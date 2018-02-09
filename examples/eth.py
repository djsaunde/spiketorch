import os, sys
import time, timeit
import torch
import logging
import argparse
import numpy as np
import pickle as p
import pandas as pd

from struct import unpack
from datetime import datetime
from torchvision import datasets

sys.path.append(os.path.abspath(os.path.join('..', 'spiketorch')))
sys.path.append(os.path.abspath(os.path.join('..', 'spiketorch', 'network')))

from network import *
from plotting import *
from classification import *

from monitors import Monitor
from synapses import Synapses, STDPSynapses
from datasets import get_MNIST, generate_spike_train
from groups import InputGroup, LIFGroup, AdaptiveLIFGroup

model_name = 'eth'

logs_path = os.path.join('..', 'logs', model_name)
data_path = os.path.join('..', 'data', model_name)
params_path = os.path.join('..', 'params', model_name)
results_path = os.path.join('..', 'results', model_name)
assign_path = os.path.join('..', 'assignments', model_name)
perform_path = os.path.join('..', 'performances', model_name)

for path in [logs_path, params_path, assign_path, results_path, perform_path]:
	if not os.path.isdir(path):
		os.makedirs(path)

np.set_printoptions(threshold=np.nan, linewidth=200)
np.warnings.filterwarnings('ignore')
torch.set_printoptions(threshold=np.nan, linewidth=100, edgeitems=10)


parser = argparse.ArgumentParser(description='ETH (with LIF neurons) \
					SNN toy model simulation implemented with PyTorch.')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--n_input', type=int, default=784)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--n_train', type=int, default=10000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--update_interval', type=int, default=250)
parser.add_argument('--nu_pre', type=float, default=1e-4)
parser.add_argument('--nu_post', type=float, default=1e-2)
parser.add_argument('--c_excite', type=float, default=22.5)
parser.add_argument('--c_inhib', type=float, default=17.5)
parser.add_argument('--time', type=int, default=350)
parser.add_argument('--rest', type=int, default=150)
parser.add_argument('--trace_tc', type=int, default=5e-2)
parser.add_argument('--wmax', type=float, default=1.0)
parser.add_argument('--dt', type=float, default=1)
parser.add_argument('--gpu', type=str, default='False')
parser.add_argument('--plot', type=str, default='False')

# Place parsed arguments in local scope.
args = parser.parse_args()
args = vars(args)
locals().update(args)

# Convert string arguments into boolean datatype.
plot = plot == 'True'
gpu = gpu == 'True'

if gpu:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Set random number generator.
np.random.seed(seed)

# Record decaying spike traces to use STDP.
traces = mode == 'train'

# Build filename from command-line arguments.
fname = '_'.join([ str(n_neurons), str(n_train), str(seed), str(c_inhib), str(c_excite), str(wmax) ])

# Set logging configuration.
logging.basicConfig(format='%(message)s', 
					filename=os.path.join(logs_path, '%s.log' % fname),
					level=logging.DEBUG,
					filemode='w')

# Log argument values.
print('\nOptional argument values:')
for key, value in args.items():
	print('-', key, ':', value)

print('\n')

# Initialize the spiking neural network.
network = Network(dt=dt)

# Add neuron populations.
network.add_group(InputGroup(n_input, traces=traces), 'X')
network.add_group(AdaptiveLIFGroup(n_neurons, traces=traces, rest=-65.0, reset=-65.0,
			threshold=-52.0, voltage_decay=1e-2, refractory=5, trace_tc=trace_tc), 'Ae')
network.add_group(LIFGroup(n_neurons, traces=traces, rest=-60.0, reset=-45.0,
		threshold=-40.0, voltage_decay=1e-1, refractory=2, trace_tc=trace_tc), 'Ai')

# Add synaptic connections between populations
if mode == 'train':
	network.add_synapses(STDPSynapses(network.groups['X'], network.groups['Ae'],
					wmax=wmax, nu_pre=nu_pre, nu_post=nu_post), name=('X', 'Ae'))
elif mode == 'test':
	if gpu:
		network.add_synapses(STDPSynapses(network.groups['X'], network.groups['Ae'],
					w=torch.from_numpy(load_params(model_name, fname, 'X_Ae')).cuda(),
						wmax=wmax, nu_pre=nu_pre, nu_post=nu_post), name=('X', 'Ae'))
	else:
		network.add_synapses(STDPSynapses(network.groups['X'], network.groups['Ae'],
							w=torch.from_numpy(load_params(model_name, fname, 'X_Ae')),
						wmax=wmax, nu_pre=nu_pre, nu_post=nu_post), name=('X', 'Ae'))

network.add_synapses(Synapses(network.groups['Ae'], network.groups['Ai'], 
					w=torch.diag(c_excite * torch.ones(n_neurons))), name=('Ae', 'Ai'))
network.add_synapses(Synapses(network.groups['Ai'], network.groups['Ae'], w=-c_inhib * \
									(torch.ones([n_neurons, n_neurons]) - torch.diag(1 \
											* torch.ones(n_neurons)))), name=('Ai', 'Ae'))

# network.add_monitor(Monitor(obj=network.groups['Ae'], state_vars=['v', 'theta']), name=('Ae', ('v', 'theta')))
# network.add_monitor(Monitor(obj=network.groups['Ai'], state_vars=['v']), name=('Ai', 'v'))

# Get training or test data from disk.
if mode == 'train':
	data = get_MNIST(train=True)
elif mode == 'test':
	data = get_MNIST(train=False)

X, y = data['X'], data['y']

# Count spikes from each neuron on each example (between update intervals).
outputs = torch.zeros([update_interval, n_neurons])

# Network simulation times.
image_time = time
rest_time = rest

# Voting schemes and neuron label assignments.
voting_schemes = ['all']
rates = torch.zeros(n_neurons, 10)
performances = { scheme : [] for scheme in voting_schemes }

if mode == 'train':
	assignments = -1 * torch.ones(n_neurons)
elif mode == 'test':
	if gpu:
		assignments = torch.from_numpy(load_assignments(model_name, fname)).cuda()
	else:
		assignments = torch.from_numpy(load_assignments(model_name, fname))

# Keep track of correct classifications for performance monitoring.
correct = { scheme : 0 for scheme in voting_schemes }
total_correct = { scheme : 0 for scheme in voting_schemes }

# Pre-calculated values.
n_input_sqrt = int(np.sqrt(n_input))
n_neurons_sqrt = int(np.sqrt(n_neurons))

# Run network simulation.
start = timeit.default_timer()

if mode == 'train':
	n_samples = n_train
elif mode == 'test':
	n_samples = n_test

n_images = X.shape[0]
best_accuracy = -np.inf

intensity = 1
for idx in range(n_samples):
	image, target = X[idx % n_images], y[idx % n_images]

	if idx % 10 == 0:
		# Log progress through dataset.
		if mode == 'train':
			logging.info('Training progress: (%d / %d) - Elapsed time: %.4f' % (idx, n_train, timeit.default_timer() - start))
			print('Training progress: (%d / %d) - Elapsed time: %.4f' % (idx, n_train, timeit.default_timer() - start))
		elif mode == 'test':
			logging.info('Test progress: (%d / %d) - Elapsed time: %.4f' % (idx, n_test, timeit.default_timer() - start))
			print('Test progress: (%d / %d) - Elapsed time: %.4f' % (idx, n_test, timeit.default_timer() - start))

	if mode == 'train':
		if idx > 0 and idx % update_interval == 0:
			# Assign labels to neurons based on network spiking activity.
			if gpu:
				inputs = torch.from_numpy(y[(idx % n_images) - update_interval : idx % n_images]).cuda()
			else:
				inputs = torch.from_numpy(y[(idx % n_images) - update_interval : idx % n_images])

			rates, assignments = assign_labels(inputs, outputs, rates, assignments)

			# Assess performance of network on last `update_interval` examples.
			logging.info('\n'); print()
			for scheme in performances.keys():
				performances[scheme].append(correct[scheme] / update_interval)  # Calculate percent correctly classified.
				correct[scheme] = 0  # Reset number of correct examples.
				
				logging.info('%s -> (current) : %.4f | (best) : %.4f | (average) : %.4f' % (scheme,
					performances[scheme][-1], max(performances[scheme]), np.mean(performances[scheme])))
				print('%s -> (current) : %.4f | (best) : %.4f | (average) : %.4f' % (scheme,
					performances[scheme][-1], max(performances[scheme]), np.mean(performances[scheme])))

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

					if gpu:
						save_params(params_path, network.get_weights(('X', 'Ae')).cpu().numpy(), fname, 'X_Ae')
						save_params(params_path, network.get_theta('Ae').cpu().numpy(), fname, 'theta')
						save_assignments(assign_path, assignments.cpu().numpy(), fname)
					else:
						save_params(params_path, network.get_weights(('X', 'Ae')).numpy(), fname, 'X_Ae')
						save_params(params_path, network.get_theta('Ae').numpy(), fname, 'theta')
						save_assignments(assign_path, assignments.numpy(), fname)

			# Save sequence of performance estimates to file.
			p.dump(performances, open(os.path.join(perform_path, fname), 'wb'))

			logging.info('\n'); print()

	inpts = {}

	# Encode current input example as Poisson spike trains.
	inpts['X'] = torch.from_numpy(generate_spike_train(image, intensity * dt, int(image_time / dt)))

	# Run network on Poisson-encoded image data.
	spikes = network.run(mode, inpts, image_time)

	# Re-run image if there isn't any network activity.
	n_retries = 0
	while torch.sum(spikes['Ae']) < 5 and n_retries < 3:
		intensity += 1; n_retries += 1
		inpts['X'] = torch.from_numpy(generate_spike_train(image, intensity * dt, int(image_time / dt)))
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
	outputs[idx % update_interval] = torch.sum(spikes['Ae'], 0)

	# Optionally plot the excitatory, inhibitory spiking.
	if plot:
		if gpu:
			inpt = inpts['X'].cpu().numpy().T
			Ae_spikes = spikes['Ae'].cpu().numpy().T; Ai_spikes = spikes['Ai'].cpu().numpy().T
			input_exc_weights = network.synapses[('X', 'Ae')].w.cpu().numpy()
			square_weights = get_square_weights(input_exc_weights, n_input_sqrt, n_neurons_sqrt)
			asgnmts = assignments.cpu().numpy().reshape([n_neurons_sqrt, n_neurons_sqrt]).T
			
			# exc_voltages = network.monitors[('Ae', ('v', 'theta'))].get('v').cpu().numpy()
			# inh_voltages = network.monitors[('Ai', 'v')].get('v').cpu().numpy(); network.monitors[('Ai', 'v')].reset()
		else:
			inpt = inpts['X'].numpy().T
			Ae_spikes = spikes['Ae'].numpy().T; Ai_spikes = spikes['Ai'].numpy().T
			input_exc_weights = network.synapses[('X', 'Ae')].w.numpy()
			square_weights = get_square_weights(input_exc_weights, n_input_sqrt, n_neurons_sqrt)
			asgnmts = assignments.numpy().reshape([n_neurons_sqrt, n_neurons_sqrt]).T
			
			# exc_voltages = network.monitors[('Ae', ('v', 'theta'))].get('v').numpy()
			# inh_voltages = network.monitors[('Ai', 'v')].get('v').numpy(); network.monitors[('Ai', 'v')].reset()
		
		if idx == 0:
			inpt_ims = plot_input(image.reshape(n_input_sqrt, n_input_sqrt), inpt)
			spike_ims = plot_spikes(Ae_spikes, Ai_spikes)
			weight_ims = plot_weights(square_weights, asgnmts, wmax=wmax)
			performance_ax = plot_performance(performances)

			# voltage_axes = plot_voltages(exc_voltages, inh_voltages)
		else:
			inpt_ims = plot_input(image.reshape(n_input_sqrt, n_input_sqrt), inpt, ims=inpt_ims)
			spike_ims = plot_spikes(Ae_spikes, Ai_spikes, ims=spike_ims)
			weight_ims = plot_weights(square_weights, asgnmts, ims=weight_ims)
			performance_ax = plot_performance(performances, ax=performance_ax)

			# voltage_axes = plot_voltages(exc_voltages, inh_voltages, axes=voltage_axes)
		
		plt.pause(1e-8)


if mode == 'train':
	logging.info('Training progress: (%d / %d) - Elapsed time: %.4f\n' % (n_train, n_train, timeit.default_timer() - start))
elif mode == 'test':
	logging.info('Test progress: (%d / %d) - Elapsed time: %.4f\n' % (n_test, n_test, timeit.default_timer() - start))

results = {}
for scheme in voting_schemes:
	if mode == 'train':
		results[scheme] = 100 * total_correct[scheme] / n_train
		logging.info('Training accuracy for voting scheme "%s": %.4f\n' % (scheme, results[scheme]))
		print('Training accuracy for voting scheme "%s": %.4f\n' % (scheme, results[scheme]))
	elif mode == 'test':
		results[scheme] = 100 * total_correct[scheme] / n_test
		logging.info('Test accuracy for voting scheme "%s": %.4f\n' % (scheme, results[scheme]))
		print('Test accuracy for voting scheme "%s": %.4f\n' % (scheme, results[scheme]))

# Save out network parameters and assignments for the test phase.
if mode == 'train':
	if gpu:
		save_params(params_path, network.get_weights(('X', 'Ae')).cpu().numpy(), fname, 'X_Ae')
		save_params(params_path, network.get_theta('Ae').cpu().numpy(), fname, 'theta')
		save_assignments(assign_path, assignments.cpu().numpy(), fname)
	else:
		save_params(params_path, network.get_weights(('X', 'Ae')).numpy(), fname, 'X_Ae')
		save_params(params_path, network.get_theta('Ae').numpy(), fname, 'theta')
		save_assignments(assign_path, assignments.numpy(), fname)

if mode == 'test':
	results = pd.DataFrame([[datetime.now(), fname] + list(results.values())], columns=['date', 'parameters'] + list(results.keys()))
	results_fname = '_'.join([str(n_neurons), str(n_train), 'results.csv'])
	
	if not results_fname in os.listdir(results_path):
		results.to_csv(os.path.join(results_path, results_fname), index=False)
	else:
		all_results = pd.read_csv(os.path.join(results_path, results_fname))
		all_results = pd.concat([all_results, results], ignore_index=True)
		all_results.to_csv(os.path.join(results_path, results_fname), index=False)
