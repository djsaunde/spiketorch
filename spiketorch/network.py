import torch


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

	def run(mode, inpts, time):
		'''
		Run network for a single iteration.
		'''
		for timestep in range(time):
			for group in self.groups:
				self.groups[group].step()

			for synapses in self.synapses:
				if type(self.synapses[synpase]) == STDPSynapses:
					self.synapes[synapses].update()

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
