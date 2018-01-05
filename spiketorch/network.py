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
			
			if not target in inpts:
				inpts[key[1]] = {}

			inpts[key[1]][key[0]] = source.s.float() @ weights

		return inpts	


	def run(self, mode, inpts, time):
		'''
		Run network for a single iteration.
		'''
		# Get inputs to all neuron groups from their parent neuron groups.
		inpts.update(self.get_inputs())
		
		for timestep in range(time):
			for key in self.groups:
				if type(self.groups[key]) == spiketorch.groups.InputGroup:
					self.groups[key].step(inpts[key][timestep, :], mode)
				else:
					self.groups[key].step(inpts[key], mode)

			if mode == 'train':
				for synapse in self.synapses:
					if type(self.synapses[synapse]) == STDPSynapses:
						self.synapses[synapses].update()

			# Get inputs to all neuron groups from their parent neuron groups.
			inpts.update(self.get_inputs())

		if mode == 'train':
			for synapses in self.synapses:
				if type(self.synapses[synpase]) == STDPSynapses:
					self.synapses[synapses].normalize()

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
