import torch
import timeit

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

	def get_spikes(self):
		return self.s

	def get_voltages(self):
		return self.v

	def get_traces(self):
		return self.x


class InputGroup(Group):
	'''
	Group of neurons clamped to input spikes.
	'''
	def __init__(self, n, traces=False, trace_tc=5e-2):
		super().__init__()

		self.n = n  # No. of neurons.
		self.s = torch.zeros_like(torch.Tensor(n))  # Spike occurences.
		
		if traces:
			self.x = torch.zeros_like(torch.Tensor(n))  # Firing traces.
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
			self.x[self.s] = 1.0


class LIFGroup(Group):
	'''
	Group of leaky integrate-and-fire neurons.
	'''
	def __init__(self, n, traces=False, rest=-65.0, reset=-65.0, threshold=-52.0, 
								refractory=5, voltage_decay=1e-2, trace_tc=5e-2):
		
		super().__init__()

		self.n = n  # No. of neurons.
		self.traces = traces  # Whether to record synpatic traces.
		self.rest = rest  # Rest voltage.
		self.reset = reset  # Post-spike reset voltage.
		self.threshold = threshold  # Spike threshold voltage.
		self.refractory = refractory  # Post-spike refractory period.
		self.voltage_decay = voltage_decay  # Rate of decay of neuron voltage.

		self.v = self.rest * torch.ones_like(torch.Tensor(n))  # Neuron voltages.
		self.s = torch.zeros_like(torch.Tensor(n))  # Spike occurences.

		if traces:
			self.x = torch.zeros_like(torch.Tensor(n))  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

		self.refrac_count = torch.zeros_like(torch.Tensor(n))  # Refractory period counters.

	def step(self, inpts, mode, dt):
		# Decay voltages.
		self.v -= dt * self.voltage_decay * (self.v - self.rest)

		if mode == 'train' and self.traces:
			# Decay spike traces.
			self.x -= dt * self.trace_tc * self.x

		# Decrement refractory counters.
		self.refrac_count[self.refrac_count != 0] -= dt

		# Check for spiking neurons.
		self.s = (self.v >= self.threshold) * (self.refrac_count == 0)
		self.refrac_count[self.s] = self.refractory
		self.v[self.s] = self.reset

		# Integrate input and decay voltages.
		self.v += sum([inpts[key] for key in inpts])

		if mode == 'train' and self.traces:
			# Setting synaptic traces.
			self.x[self.s] = 1.0


class AdaptiveLIFGroup(Group):
	'''
	Group of leaky integrate-and-fire neurons with adaptive thresholds.
	'''
	def __init__(self, n, traces=False, rest=-65.0, reset=-65.0, threshold=-52.0, refractory=5,
							voltage_decay=1e-2, theta_plus=0.05, theta_decay=1e-7, trace_tc=5e-2):
		
		super().__init__()

		self.n = n  # No. of neurons.
		self.traces = traces  # Whether to record synpatic traces.
		self.rest = rest  # Rest voltage.
		self.reset = reset  # Post-spike reset voltage.
		self.threshold = threshold  # Spike threshold voltage.
		self.refractory = refractory  # Post-spike refractory period.
		self.voltage_decay = voltage_decay  # Rate of decay of neuron voltage.
		self.theta_plus = theta_plus  # Constant mV to raise threshold potential post-firing.
		self.theta_decay = theta_decay  # Rate of decay of adaptive threshold potential.

		self.v = self.rest * torch.ones_like(torch.Tensor(n))  # Neuron voltages.
		self.s = torch.zeros_like(torch.Tensor(n))  # Spike occurences.
		self.theta = torch.zeros_like(torch.Tensor(n))  # Adaptive threshold parameters.

		if traces:
			self.x = torch.zeros_like(torch.Tensor(n))  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

		self.refrac_count = torch.zeros_like(torch.Tensor(n))  # Refractory period counters.

	def step(self, inpts, mode, dt):
		overall_start = timeit.default_timer()

		# Decay voltages.
		start = timeit.default_timer()
		self.v -= dt * self.voltage_decay * (self.v - self.rest)
		end = timeit.default_timer(); print('Voltage decay:', (end - start) * 1000)

		if mode == 'train':
			# Decay spike traces and adaptive thresholds.
			if self.traces:
				start = timeit.default_timer()
				self.x -= dt * self.trace_tc * self.x
				end = timeit.default_timer(); print('Trace decay', (end - start) * 1000)
			
			start = timeit.default_timer()
			self.theta -= dt * self.theta_decay * self.theta
			end = timeit.default_timer(); print('Theta decay:', (end - start) * 1000)

		# Decrement refractory counters.
		start = timeit.default_timer()
		self.refrac_count -= dt
		end = timeit.default_timer(); print('Refractory decay:', (end - start) * 1000)

		# Check for spiking neurons.
		start = timeit.default_timer()
		self.s = (self.v >= self.threshold + self.theta) & (self.refrac_count <= 0)
		end = timeit.default_timer(); print('Check for spikes:', (end - start) * 1000)
		
		start = timeit.default_timer()
		self.refrac_count[self.s] = dt * self.refractory
		end = timeit.default_timer(); print('Update refractory:', (end - start) * 1000)
		
		start = timeit.default_timer()
		self.v[self.s] = self.reset
		end = timeit.default_timer(); print('Reset voltages:', (end - start) * 1000)

		# Choose only a single neuron to spike (ETH replication).
		if torch.sum(self.s) > 0:
			start = timeit.default_timer()
			s = torch.zeros_like(torch.Tensor(self.s.size()))
			end = timeit.default_timer(); print('Init spike vector:', (end - start) * 1000)
			
			start = timeit.default_timer()
			s[torch.multinomial(self.s.float(), 1)] = 1
			end = timeit.default_timer(); print('Pick a single spike:', (end - start) * 1000)

		# Integrate inputs.
		start = timeit.default_timer()
		self.v += sum([inpts[key] for key in inpts])
		end = timeit.default_timer(); print('Integrate inputs:', (end - start) * 1000)

		if mode == 'train' and self.traces:
			# Update adaptive thresholds, synaptic traces.
			start = timeit.default_timer()
			self.theta[self.s] += self.theta_plus
			end = timeit.default_timer(); print('Update theta:', (end - start) * 1000)
			
			start = timeit.default_timer()
			self.x[self.s] = 1.0
			end = timeit.default_timer(); print('Update traces:', (end - start) * 1000)

		print(); overall_end = timeit.default_timer(); print((overall_end - overall_start) * 1000); print()
