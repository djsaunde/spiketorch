import torch
import os, sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join('..', '..', 'spiketorch')))
sys.path.append(os.path.abspath(os.path.join('..', '..', 'spiketorch', 'network')))

from network import *

from monitors import Monitor
from synapses import Synapses, STDPSynapses
from groups import InputGroup, LIFGroup, AdaptiveLIFGroup


class TestNetwork:
	def test_init(self):
		network = Network()
		
		assert network.dt == 1
		assert network.groups == {}
		assert network.synapses == {}
		assert network.monitors == {}
		assert network.get_inputs() == {}

		network = Network(dt=0.25)
		assert network.dt == 0.25

		assert network.dt == 0.25
		assert network.groups == {}
		assert network.synapses == {}
		assert network.monitors == {}
		assert network.get_inputs() == {}

	def test_run(self):
		network = Network()
		spikes = network.run(mode='train', inpts={}, time=0); assert spikes == {}
		spikes = network.run(mode='train', inpts={}, time=1); assert spikes == {}
		spikes = network.run(mode='train', inpts={}, time=100); assert spikes == {}
		spikes = network.run(mode='train', inpts={}, time=10000); assert spikes == {}		

	def test_build(self):
		network = Network(); group = LIFGroup(100); network.add_group(group, name='group')

		assert len(network.groups) == 1
		assert type(network.groups['group']) == LIFGroup
		assert network.groups['group'].n == 100

		synapses = Synapses(group, group); network.add_synapses(synapses, source='group', target='group')

		assert len(network.synapses) == 1
		assert type(network.synapses[('group', 'group')]) == Synapses
		assert network.synapses[('group', 'group')].w.size() == torch.Size([100, 100])