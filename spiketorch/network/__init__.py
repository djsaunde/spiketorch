import torch
import os, sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join('..', 'spiketorch', 'network')))

import groups, synapses

class Network:
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, dt=1):
		self.dt = dt
		self.groups = {}
		self.synapses = {}
		self.monitors = {}

	def add_group(self, group, name):
		self.groups[name] = group

	def add_synapses(self, synapses, source, target):
		self.synapses[(source, target)] = synapses

	def add_monitor(self, monitor, name):
		self.monitors[name] = monitor

	def get_inputs(self):
		inpts = {}
		for key in self.synapses:
			weights = self.synapses[key].w

			source = self.synapses[key].source
			target = self.synapses[key].target

			if not key[1] in inpts:
				inpts[key[1]] = torch.zeros_like(torch.Tensor(target.n))

			inpts[key[1]] += source.s.float() @ weights

		return inpts

	def get_weights(self, name):
		return self.synapses[name].w

	def get_theta(self, name):
		return self.groups[name].theta

	def run(self, mode, inpts, time):
		'''
		Run network for a single iteration.
		'''
		# Record spikes from each population over the iteration.
		spikes = {}
		for key in self.groups:
			spikes[key] = torch.zeros_like(torch.Tensor(int(time / self.dt), self.groups[key].n))

		for monitor in self.monitors:
			self.monitors[monitor].reset()

		# Get inputs to all neuron groups from their parent neuron groups.
		inpts.update(self.get_inputs())
		
		# Simulate neuron and synapse activity for `time` timesteps.
		for timestep in range(int(time / self.dt)):
			# Update each group in turn.
			for key in self.groups:
				if type(self.groups[key]) == groups.InputGroup:
					self.groups[key].step(inpts[key][timestep, :], mode, self.dt)

					# Record spikes from this population at this timestep.
					spikes[key][timestep, :] = self.groups[key].s
			
			for key in self.groups:
				if type(self.groups[key]) != groups.InputGroup:
					self.groups[key].step(inpts[key], mode, self.dt)

					# Record spikes from this population at this timestep.
					spikes[key][timestep, :] = self.groups[key].s

			# Update synapse weights if we're in training mode.
			if mode == 'train':
				for synapse in self.synapses:
					if type(self.synapses[synapse]) == synapses.STDPSynapses:
						self.synapses[synapse].update()

			# Get inputs to all neuron groups from their parent neuron groups.
			inpts.update(self.get_inputs())

			for monitor in self.monitors:
				self.monitors[monitor].record()

		# Normalize synapse weights if we're in training mode.
		if mode == 'train':
			for synapse in self.synapses:
				if type(self.synapses[synapse]) == synapses.STDPSynapses:
					self.synapses[synapse].normalize()

		return spikes

	def reset(self, attrs=['v', 'x']):
		'''
		Resets certain state variables.
		'''
		for group in self.groups:
			for attr in attrs:
				if hasattr(self.groups[group], attr) and attr in ['v']:
					# Voltages.
					self.groups[group].v[:] = self.groups[group].rest

				if hasattr(self.groups[group], attr) and attr in ['x', 'theta']:
					# Synaptic traces or adaptive thresholds.
					self.groups[group].x[:] = 0


def save_params(params_path, params, fname, prefix):
	'''
	Save network params to disk.

	Arguments:
		- params (numpy.ndarray): Array of params to save.
		- fname (str): File name of file to write to.
	'''
	np.save(os.path.join(params_path, '_'.join([prefix, fname]) + '.npy'), params)


def load_params(params_path, fname, prefix):
	'''
	Load network params from disk.

	Arguments:
		- fname (str): File name of file to read from.
		- prefix (str): Name of the parameters to read from disk.

	Returns:
		- params (numpy.ndarray): Params stored in file `fname`.
	'''
	return np.load(os.path.join(params_path, '_'.join([prefix, fname]) + '.npy'))


def save_assignments(assign_path, assignments, fname):
	'''
	Save network assignments to disk.

	Arguments:
		- assignments (numpy.ndarray): Array of assignments to save.
		- fname (str): File name of file to write to.
	'''
	np.save(os.path.join(assign_path, '_'.join(['assignments', fname]) + '.npy'), assignments)


def load_assignments(assign_path, fname):
	'''
	Save network assignments to disk.

	Arguments:
		- fname (str): File name of file to read from.

	Returns:
		- assignments (numpy.ndarray): Assignments stored in file `fname`.
	'''
	return np.load(os.path.join(assign_path, '_'.join(['assignments', fname]) + '.npy'))


def get_square_weights(weights, n_sqrt):
	square_weights = torch.zeros_like(torch.Tensor(28 * n_sqrt, 28 * n_sqrt))
	for i in range(n_sqrt):
		for j in range(n_sqrt):
			filter_ = weights[:, i * n_sqrt + j].contiguous().view(28, 28)
			square_weights[i * 28 : (i + 1) * 28, (j % n_sqrt) * 28 : ((j % n_sqrt) + 1) * 28] = filter_
	
	return square_weights


def get_conv_weights(weights, kernel_size, stride, n_patches, n_patch_neurons):
	n_patches_sqrt = int(np.sqrt(n_patches))
	n_patch_neurons_sqrt = int(np.sqrt(n_patch_neurons))

	rearranged = np.zeros_like(torch.Tensor([kernel_size * n_patch_neurons, kernel_size * n_patches]))

	for patch in range(n_patches):
		for neuron in range(n_patch_neurons):
			rearranged[kernel_size * neuron : kernel_size * (neuron + 1), kernel_size * patch : kernel_size * (patch + 1)] = weights[patch]

	return rearranged.T


def get_convolution_locations(neuron, n_patch_neurons_sqrt, n_input_sqrt, kernel_size, stride):
	convolution_locations = [0] * (n_input_sqrt ** 2)

	for x in range(kernel_size):
		for y in range(kernel_size):
			convolution_locations[(((neuron % n_patch_neurons_sqrt) * stride + (neuron // \
				n_patch_neurons_sqrt) * n_input_sqrt * stride) + (x * n_input_sqrt) + y)] = 1

	return convolution_locations
