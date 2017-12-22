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

	def run(inpts, time):
		for timestep in range(time):
			for group in self.groups:
				self.groups[group].step()

			for synapses in self.synapses:
				if type(self.synapses[synpase]) == STDPSynapses:
					self.synapes[synapses].update()

		for synapses in self.synapses:
			if type(self.synapses[synpase]) == STDPSynapses:
				self.synapses[synapses].normalize()