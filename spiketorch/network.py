import torch
import spiketorch


class Network:
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self):
		self.groups = {}
		self.synapses = {}

	def add_group(self, group, name):
		self.groups[name] = group

	def add_synapses(self, synapses, name):
		self.synapses[name] = synapses

	def get_inputs(self):
		inpts = {}
		for key in self.synapses:
			weights = self.synapses[key].w

			source = self.synapses[key].source
			target = self.synapses[key].target

			if not key[1] in inpts:
				inpts[key[1]] = {}

			inpts[key[1]][key[0]] = source.s.float() @ weights

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
			spikes[key] = torch.zeros(time, self.groups[key].n)

		# Get inputs to all neuron groups from their parent neuron groups.
		inpts.update(self.get_inputs())
		
		# Simulate neuron and synapse activity for `time` timesteps.
		for timestep in range(time):
			# Update each group in turn.
			for key in self.groups:
				if type(self.groups[key]) == spiketorch.groups.InputGroup:
					self.groups[key].step(inpts[key][timestep, :], mode)

				# Record spikes from this population at this timestep.
				spikes[key][timestep, :] = self.groups[key].s
			
			for key in self.groups:
				if type(self.groups[key]) != spiketorch.groups.InputGroup:
					self.groups[key].step(inpts[key], mode)

				# Record spikes from this population at this timestep.
				spikes[key][timestep, :] = self.groups[key].s

			# Update synapse weights if we're in training mode.
			if mode == 'train':
				for synapse in self.synapses:
					if type(self.synapses[synapse]) == spiketorch.synapses.STDPSynapses:
						self.synapses[synapse].update()

			# Get inputs to all neuron groups from their parent neuron groups.
			inpts.update(self.get_inputs())

		# Normalize synapse weights if we're in training mode.
		if mode == 'train':
			for synapse in self.synapses:
				if type(self.synapses[synapse]) == spiketorch.synapses.STDPSynapses:
					self.synapses[synapse].normalize()

		return spikes

	def reset(self):
		'''
		Reset relevant state variables after a single iteration.
		'''
		for group in self.groups:
			if hasattr(self.groups[group], 'v'):
				# Voltages.
				self.groups[group].v[:] = self.groups[group].rest

			if hasattr(self.groups[group], 'x'):
				# Synaptic traces.
				self.groups[group].x[:] = 0
