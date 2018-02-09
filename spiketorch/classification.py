import torch


def classify(spikes, voting_schemes, assignments):
	'''
	Given the neuron assignments and the network spiking
	activity, make predictions about the data targets.
	'''
	spikes = spikes.sum(0)

	predictions = {}
	for scheme in voting_schemes:
		rates = torch.zeros(10)

		if scheme == 'all':
			for idx in range(10):
				n_assigns = torch.nonzero(assignments == idx).numel()
				
				if n_assigns > 0:
					idxs = torch.nonzero((assignments == idx).long().view(-1)).view(-1)
					rates[idx] = torch.sum(spikes[idxs]) / n_assigns

		predictions[scheme] = torch.sort(rates, dim=0, descending=True)[1]

	return predictions


def assign_labels(inputs, outputs, rates, assignments):
	'''
	Given the excitatory neuron firing history, assign them class labels.
	'''
	# Loop over all target categories.
	for j in range(10):
		# Count the number of inputs having this target.
		n_inputs = torch.nonzero(inputs == j).numel()
		if n_inputs > 0:
			# Get indices of inputs with this category.
			idxs = torch.nonzero((inputs == j).long().view(-1)).view(-1)
			# Calculate average firing rate per neuron, per category.
			rates[:, j] = 0.9 * rates[:, j] + torch.sum(outputs[idxs], 0) / n_inputs

	# Assignments of neurons are the categories for which they fire the most. 
	assignments = torch.max(rates, 1)[1]

	return rates, assignments