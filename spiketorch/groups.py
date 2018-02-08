import torch

from abc import ABC, abstractmethod


class Group(ABC):
	'''
	Abstract base class for groups of neurons.
	'''
	def __init__(self):
		super().__init__()

	@abstractmethod
	def step(self, inpts, mode):
		pass


class InputGroup(Group):
	'''
	Group of neurons clamped to input spikes.
	'''
	def __init__(self, n, traces=False, trace_tc=5e-2):
		super().__init__()

		self.n = n  # No. of neurons.
		self.s = torch.zeros(n)  # Spike occurences.
		
		if traces:
			self.x = torch.zeros(n)  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

	def step(self, inpts, mode, dt):
		'''
		On each simulation step, set the spikes of the
		population equal to the inputs.
		'''
		if mode == 'train':
			# Decay spike traces.
			self.x -= dt * self.trace_tc * self.x

		self.s = inpts

		if mode == 'train':
			# Setting synaptic traces.
			self.x[self.s.byte()] = 1.0

	def get_spikes(self):
		return self.s

	def get_voltages(self):
		return self.v

	def get_traces(self):
		return self.traces


class LIFGroup(Group):
	'''
	Group of leaky integrate-and-fire neurons.
	'''
	def __init__(self, n, traces=False, rest=-65.0, reset=-65.0, threshold=-52.0, 
								refractory=5, voltage_decay=1e-2, trace_tc=5e-2):
		
		super().__init__()

		self.n = n  # No. of neurons.
		self.rest = rest  # Rest voltage.
		self.reset = reset  # Post-spike reset voltage.
		self.threshold = threshold  # Spike threshold voltage.
		self.refractory = refractory  # Post-spike refractory period.
		self.voltage_decay = voltage_decay  # Rate of decay of neuron voltage.

		self.v = self.rest * torch.ones(n)  # Neuron voltages.
		self.s = torch.zeros(n)  # Spike occurences.

		if traces:
			self.x = torch.zeros(n)  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

		if self.refractory > 0:
			self.refrac_count = torch.zeros(n)  # Refractory period counters.

	def step(self, inpts, mode, dt):
		if mode == 'train':
			# Decay spike traces.
			self.x -= dt * self.trace_tc * self.x

		if self.refractory > 0:
			# Decrement refractory counters.
			self.refrac_count[self.refrac_count != 0] -= 1 * dt

		# Check for spiking neurons.
		self.s = (self.v >= self.threshold) * (self.refrac_count == 0)

		# Reset refractory periods for spiked neurons.
		self.refrac_count[self.s] = self.refractory

		# Reset neurons above their threshold voltage.
		self.v[self.s] = self.reset

		# Integrate input and decay voltages.
		for key in inpts:
			self.v += inpts[key]

		self.v -= dt * self.voltage_decay * (self.v - self.rest)

		if mode == 'train':
			# Setting synaptic traces.
			self.x[self.s.byte()] = 1.0

	def get_spikes(self):
		return self.s

	def get_voltages(self):
		return self.v

	def get_traces(self):
		return self.traces


class AdaptiveLIFGroup(Group):
	'''
	Group of leaky integrate-and-fire neurons with adaptive thresholds.
	'''
	def __init__(self, n, traces=False, rest=-65.0, reset=-65.0, threshold=-52.0, refractory=5,
							voltage_decay=1e-2, theta_plus=0.05, theta_decay=1e-7, trace_tc=5e-2):
		
		super().__init__()

		self.n = n  # No. of neurons.
		self.rest = rest  # Rest voltage.
		self.reset = reset  # Post-spike reset voltage.
		self.threshold = threshold  # Spike threshold voltage.
		self.refractory = refractory  # Post-spike refractory period.
		self.voltage_decay = voltage_decay  # Rate of decay of neuron voltage.
		self.theta_plus = theta_plus  # Constant mV to raise threshold potential post-firing.
		self.theta_decay = theta_decay  # Rate of decay of adaptive threshold potential.

		self.v = self.rest * torch.ones(n)  # Neuron voltages.
		self.s = torch.zeros(n)  # Spike occurences.
		self.theta = torch.zeros(n)  # Adaptive threshold parameters.

		if traces:
			self.x = torch.zeros(n)  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

		if self.refractory > 0:
			self.refrac_count = torch.zeros(n)  # Refractory period counters.

	def step(self, inpts, mode, dt):
		if mode == 'train':
			# Decay spike traces.
			self.x -= dt * self.trace_tc * self.x

		if self.refractory > 0:
			# Decrement refractory counters.
			self.refrac_count[self.refrac_count != 0] -= 1 * dt

		# Check for spiking neurons.
		self.s = (self.v >= self.threshold + self.theta) * (self.refrac_count == 0)

		if torch.sum(self.s) > 0:
			s = torch.zeros(self.s.size())
			s[torch.multinomial(self.s.float(), 1)] = 1
			self.s = s.byte()

		# Reset refractory periods for spiked neurons.
		self.refrac_count[self.s] = self.refractory

		# Reset neurons above their threshold voltage.
		self.v[self.s] = self.reset

		# Integrate inputs.
		for key in inpts:
			self.v += inpts[key]

		# Decay voltages.
		self.v -= dt * self.voltage_decay * (self.v - self.rest)

		if mode == 'train':
			# Update adaptive thresholds.
			self.theta[self.s] += self.theta_plus
			self.theta -= dt * self.theta_decay * self.theta

			# Setting synaptic traces.
			self.x[self.s.byte()] = 1.0

	def get_spikes(self):
		return self.s

	def get_voltages(self):
		return self.v

	def get_traces(self):
		return self.traces
