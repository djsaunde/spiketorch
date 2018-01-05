import torch


class Synapses:
	'''
	Specifies constant synapses between two populations of neurons.
	'''
	def __init__(self, source, target, w=None):
		self.source = source
		self.target = target

		if w is None:
			self.w = torch.rand(source.n, target.n)
		else:
			self.w = w


class STDPSynapses:
	'''
	Specifies STDP-adapted synapses between two populations of neurons.
	'''
	def __init__(self, source, target, w=None, nu_pre=1e-4, nu_post=1e-2, wmax=1.0, norm=78.0):
		self.source = source
		self.target = target

		if w is None:
			self.w = torch.rand(source.n, target.n)
		else:
			self.w = w

		self.nu_pre = nu_pre
		self.nu_post = nu_post
		self.wmax = wmax
		self.norm = norm

	def normalize(self):
		'''
		Normalize weights to have average value `self.norm`.
		'''
		self.w *= self.norm / self.w.sum(0).view(1, -1)

	def update(self):
		'''
		Perform STDP weight update.
		'''
		# Post-synaptic.
		self.w += self.nu_post * (self.source.x.view(self.source.n, 1) * self.target.s.float().view(1, self.target.n))
		# Pre-synaptic.
		self.w -= self.nu_pre * (self.source.s.float().view(self.source.n, 1) * self.target.x.view(1, self.target.n))

		# Ensure that weights are within [0, self.wmax].
		self.w = torch.clamp(self.w, 0, self.wmax)
