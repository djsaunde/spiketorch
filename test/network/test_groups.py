import torch
import os, sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join('..', '..', 'spiketorch')))
sys.path.append(os.path.abspath(os.path.join('..', '..', 'spiketorch', 'network')))

from network import *

from monitors import Monitor
from synapses import Synapses, STDPSynapses
from groups import InputGroup, LIFGroup, AdaptiveLIFGroup


class TestGroups:
	def test_init(self):
		for group_type in [InputGroup, LIFGroup, AdaptiveLIFGroup]:
			for n in [0, 1, 100, 10000]:
				group = group_type(n)
				
				assert group.n == n
				assert all(group.s == torch.zeros(n))
				assert all(group.get_spikes() == torch.zeros(n))
				
				if group_type in [LIFGroup, AdaptiveLIFGroup]:
					assert all(group.get_voltages() == group.rest * torch.ones(n))

				group = group_type(n, traces=True, trace_tc=1e-5)
				
				assert group.n == n; assert group.trace_tc == 1e-5
				assert all(group.s == torch.zeros(n)); assert all(group.x == torch.zeros(n))
				assert all(group.get_spikes() == torch.zeros(n))
				assert all(group.get_traces() == torch.zeros(n))
				
				if group_type in [LIFGroup, AdaptiveLIFGroup]:
					assert all(group.get_voltages() == group.rest * torch.ones(n))

		for group_type in [LIFGroup, AdaptiveLIFGroup]:
			for n in [0, 1, 100, 10000]:
				group = group_type(n, rest=0.0, reset=-10.0, threshold=10.0, refractory=3, voltage_decay=7e-4)
				
				assert group.rest == 0.0; assert group.reset == -10.0; assert group.threshold == 10.0
				assert group.refractory == 3; assert group.voltage_decay == 7e-4
				assert all(group.get_spikes() == torch.zeros(n))
				assert all(group.get_voltages() == group.rest * torch.ones(n))