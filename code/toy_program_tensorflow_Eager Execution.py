import os
import timeit
import argparse
import numpy as np
import pickle as p
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe

from struct import unpack
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util import *

tfe.enable_eager_execution()


class ETH:
  '''
  Replication of the spiking neural network model from "Unsupervised learning of digit
  recognition using spike-timing-dependent plasticity"
  (https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#).
  '''
  def __init__(self, seed=0, n_input=784, n_neurons=100, n_examples=(10000, 10000), dt=1, lrs=(1e-4, 1e-2), \
                c_inhib=17.4, sim_times=(350, 150, 350, 150), stdp_times=(20, 20), update_interval=100, wmax=1.0, gpu='True'):
    '''
    Constructs the network based on chosen parameters.
    Arguments:
      - seed: Sets the random number generator sequence.
      - n_neurons: Number of neurons in both excitatory and inhibitory populations.
      - n_input: Number of input neurons (corresponds to dimensionality of image).
      - n_examples: Tuple of (n_train, n_test); the number of examples to train and test on.
      - dt: Simulation time step; e.g., 0.5ms.
      - lrs: Tuple of (nu_pre, nu_post); pre- and post-synaptic learning rates.
      - c_inhib: Strength of synapses from inhibitory to excitatory layer.
      - sim_times: Tuple of (train_time, train_rest, test_time, test_rest). Specifies
        how many time steps are used for each training and test examples and rest periods.
      - stdp_times: Tuple of (tc_pre, tc_post); gives STDP window times constants for pre-
        and post-synaptic updates.
    '''

    self.device = ''
    if gpu:
      self.device = "/gpu:0"
    else:
      self.device = "/cpu:0"

    print(self.device)


    # Set Tensorflow and numpy random number generator.
    with tf.device(self.device):
      tf.set_random_seed(seed)
      np.random.seed(seed)

      # Set class attributes.
      self.n_input = n_input
      self.n_input_sqrt = int(np.sqrt(n_input))
      self.n_neurons = n_neurons
      self.n_neurons_sqrt = int(np.sqrt(n_neurons))
      self.n_examples = n_examples
      self.dt = dt
      self.lrs = { 'nu_pre' : lrs[0], 'nu_post' : lrs[1] }
      self.c_inhib = c_inhib
      self.sim_times = { 'train_time' : sim_times[0], 'train_rest' : sim_times[1], \
      'test_time' : sim_times[2], 'test_rest' : sim_times[3] }
      self.stdp_times = { 'X' : dt / stdp_times[0], 'Ae' : dt / stdp_times[1] }

      # Generic filename for saving out weights and other parameters.
      self.fname = '_'.join([ str(n_neurons), str(n_train), str(seed) ])

      # Population names.
      self.populations = ['Ae', 'Ai']
      self.stdp_populations = ['X', 'Ae']

      # Assignments and performance monitoring update interval.
      self.update_interval = update_interval

      # Excitatory neuron assignments.
      if mode == 'train':
        self.assignments = -1 * np.ones([1, n_neurons])
      elif mode == 'test':
        self.assignments = load_assignments('.'.join(['_'.join(['assignments', self.fname]), 'npy']))

      # Instantiate weight matrices.+
      self.W = { 'X_Ae' : tf.get_variable('w_in_exc', dtype=tf.float32, initializer =
                          tf.random_uniform([n_input, n_neurons], minval=0.01, maxval=0.3, dtype=tf.float32)), \
                'Ae_Ai' : tf.get_variable('w_exc_inh', dtype=tf.float32, initializer =
                          tf.diag(22.5 * tf.ones([n_neurons]))), \
                'Ai_Ae' : tf.get_variable('w_inh_exc', dtype=tf.float32, initializer =
                          c_inhib * tf.ones([n_neurons, n_neurons]) - tf.diag(c_inhib * tf.ones([n_neurons])))
               }

      # Simulation parameters.
      # Rest (decay towards) voltages.
      self.rest = { 'Ae' : tf.get_variable('rest_Ae', dtype=tf.float32, initializer = tf.ones([1, n_neurons]) * -65.0),
                    'Ai' : tf.get_variable('rest_Ai', dtype=tf.float32, initializer = tf.ones([1, n_neurons]) * -60.0) }
      # Reset (after spike) voltages.
      self.reset = { 'Ae' : tf.get_variable('reset_Ae', dtype=tf.float32, initializer = tf.ones([1, n_neurons]) * -65.0),
                     'Ai' : tf.get_variable('reset_Ai', dtype=tf.float32, initializer = tf.ones([1, n_neurons]) * -45.0) }
      # Threshold voltages.
      self.threshold = { 'Ae' : tf.get_variable('threshold_Ae', dtype=tf.float32, initializer = tf.ones([1, n_neurons]) * -52.0),
                         'Ai' : tf.get_variable('threshold_Ai', dtype=tf.float32, initializer = tf.ones([1, n_neurons]) * -40.0)}
      # Neuron refractory periods in milliseconds.
      self.refractory = { 'Ae' : tf.get_variable('refractory_Ae', dtype=tf.float32, initializer = tf.ones([1, n_neurons]) * 5),
                          'Ai' : tf.get_variable('refractory_Ai', dtype=tf.float32, initializer = tf.ones([1, n_neurons]) * 2) }
      # Adaptive threshold time constant and step increase.
      self.theta_decay = tf.constant(dt / 1e7, dtype=tf.float32)
      self.theta_plus = tf.constant(0.05, dtype=tf.float32)
      # Population-level decay constants.
      self.v_decay = { 'Ae' : tf.constant(dt / 100, dtype=tf.float32) ,
                       'Ai' : tf.constant(dt / 10, dtype=tf.float32) }
      # Voting schemes.
      self.voting_schemes = [ 'all' ]
      # Network performances indexed by voting schemes.
      self.performances = { scheme : [] for scheme in self.voting_schemes }

      self.rates = np.zeros([self.n_neurons, 10])

      # Etc.
      self.intensity = 2.0
      self.wmax = tf.constant(1.0)
      self.norm = 78.0 * self.wmax

      # Instantiate neuron state variables.
      # Neuron voltages.
      self.v = { 'Ae' : tf.get_variable('v_Ae', dtype=tf.float32, initializer = self.rest['Ae'] * tf.ones([1, n_neurons])),
                 'Ai' : tf.get_variable('v_Ai', dtype=tf.float32, initializer = self.rest['Ai'] * tf.ones([1, n_neurons])) }
      # Spike occurrences.
      self.s = { 'X' : tf.get_variable('s_X',  dtype=tf.float32, initializer = tf.zeros([1, n_input])),
                 'Ae': tf.get_variable('s_Ae', dtype=tf.bool, initializer = tf.zeros([1, n_neurons],  dtype=tf.bool)),
                 'Ai': tf.get_variable('s_Ai', dtype=tf.bool, initializer = tf.zeros([1, n_neurons],  dtype=tf.bool)) }
      # Synaptic traces (used for STDP calculations).
      self.a = { 'X' : tf.get_variable('a_X',  dtype=tf.float32, initializer = tf.zeros([1, n_input])),
                 'Ae': tf.get_variable('a_Ae', dtype=tf.float32, initializer = tf.zeros([1, n_neurons])) }
      # Adaptive additive threshold parameters (used in excitatory layer).
      self.theta = tf.get_variable('theta', dtype=tf.float32, initializer = tf.zeros([1, n_neurons]))
      # Refractory period counters.
      self.refrac_count = { 'Ae' : tf.get_variable('refrac_count_Ae', dtype=tf.float32, initializer = tf.zeros([1, n_neurons])),
                            'Ai' : tf.get_variable('refrac_count_Ai', dtype=tf.float32, initializer = tf.zeros([1, n_neurons])) }



  def run(self, mode = 'train', inpt = {}, time = 0 ):
      '''
      Runs the network on a single input for some time.
      Arguments:
          - mode (str): Whether we are in test or training mode.
                  Affects whether to adaptive network parameters.
          - inpt (torch.Tensor / torch.cuda.Tensor): Network input,
                  encoded as Poisson spike trains. Has shape (time,
                  self.n_input).
          - time (int): How many simulation time steps to run.
      Returns:
          State variables recorded over the simulation iteration.
      '''
      # Records network state variables for plotting purposes.
      with tf.device(self.device):
        spikes = { pop : np.zeros([time, self.n_neurons], dtype=np.bool) for pop in self.populations }
        if plot:
            voltages = { pop : np.zeros([time, self.n_neurons], dtype=np.float32) for pop in self.populations }
            traces =   { 'X' : np.zeros([time, self.n_input], dtype=np.float32), 'Ae' : np.zeros([time, self.n_neurons]) }

        # Run simulation for `time` simulation steps.
        for timestep in range(time):
            # Record voltage history.
            if plot:
                voltages['Ae'][timestep, :] = tf.reshape(tf.transpose(self.v['Ae']),[-1])
                voltages['Ai'][timestep, :] = tf.reshape(tf.transpose(self.v['Ai']),[-1])

            # Get input spikes for this timestep.
            self.s['X'] = tf.expand_dims(tf.cast(inpt[timestep, :], dtype=tf.float32 ),0)

            # Decrement refractory counters.
            #self.refrac_count['Ae'][self.refrac_count['Ae'] != 0] -= 1
            loc_temp = tf.not_equal(self.refrac_count['Ae'], 0)
            self.refrac_count['Ae'] = tf.where(loc_temp, self.refrac_count['Ae'] - 1, self.refrac_count['Ae'])
            #self.refrac_count['Ai'][self.refrac_count['Ai'] != 0] -= 1
            loc_temp = tf.not_equal(self.refrac_count['Ai'], 0)
            self.refrac_count['Ai'] = tf.where(loc_temp, self.refrac_count['Ai'] - 1, self.refrac_count['Ai'])

            # Check for spiking neurons.
            #self.s['Ae'] = (self.v['Ae'] >= self.threshold['Ae'] + self.theta) * (self.refrac_count['Ae'] == 0)
            loc_temp = tf.equal(self.refrac_count['Ae'],0)
            thr_temp = tf.greater_equal(self.v['Ae'], self.threshold['Ae'] + self.theta)
            self.s['Ae'] = tf.logical_and(loc_temp, thr_temp)

            #self.s['Ai'] = (self.v['Ai'] >= self.threshold['Ai']) * (self.refrac_count['Ai'] == 0)
            loc_temp = tf.equal(self.refrac_count['Ai'],0)
            thr_temp = tf.greater_equal(self.v['Ai'], self.threshold['Ai'])
            self.s['Ai'] = tf.logical_and(loc_temp, thr_temp)

            # Reset refractory periods for spiked neurons.
            #self.refrac_count['Ae'][self.s['Ae']] = self.refractory['Ae']
            self.refrac_count['Ae'] = tf.where(self.s['Ae'], self.refractory['Ae'], self.refrac_count['Ae'])
            #self.refrac_count['Ai'][self.s['Ai']] = self.refractory['Ai']
            self.refrac_count['Ai'] = tf.where(self.s['Ai'], self.refractory['Ai'], self.refrac_count['Ai'])


            if mode == 'train':
                # Update adaptive thresholds.
                # self.theta[self.s['Ae']] += self.theta_plus
                self.theta = tf.where(self.s['Ae'], tf.add(self.theta, self.theta_plus), self.theta)

            # Record neuron spiking.
            #spikes['Ae'][timestep, :] = self.s['Ae']
            spikes['Ae'][timestep, :] = (tf.reshape(tf.transpose(self.s['Ae']),[-1]))
            #spikes['Ai'][timestep, :] = self.s['Ai']
            spikes['Ai'][timestep, :] = tf.reshape(tf.transpose(self.s['Ai']),[-1])

            # Setting synaptic traces.
            #self.a['X'][self.s['X'].byte()] = 1.0
            self.a['X'] = tf.where(tf.cast(self.s['X'],dtype=tf.bool), tf.ones([1, self.n_input]), self.a['X'])
            #self.a['Ae'][self.s['Ae'].byte()] = 1.0
            self.a['Ae'] = tf.where(self.s['Ae'], tf.ones([1, self.n_neurons]), self.a['Ae'] )

            # Reset neurons above their threshold voltage.
            #self.v['Ae'][self.s['Ae']] = self.reset['Ae']
            self.v['Ae'] = tf.where(self.s['Ae'], self.reset['Ae'], self.v['Ae'])
            #self.v['Ai'][self.s['Ai']] = self.reset['Ai']
            self.v['Ai'] = tf.where(self.s['Ai'], self.reset['Ai'], self.v['Ai'])

            # Integrate input and decay voltages.
            #self.v['Ae'] += self.s['X'].float() @ self.W['X_Ae'] - self.s['Ai'].float() @ self.W['Ai_Ae']
            v_Ae_p = tf.subtract(tf.matmul(self.s['X'], self.W['X_Ae']),
                                 tf.matmul(tf.cast(self.s['Ai'], tf.float32), self.W['Ai_Ae']))
            #self.v['Ae'] -= self.v_decay['Ae'] * (self.v['Ae'] - self.rest['Ae'])
            v_Ae_n = tf.scalar_mul(self.v_decay['Ae'], tf.subtract(self.v['Ae'], self.rest['Ae']))
            self.v['Ae'] = tf.subtract(tf.add(self.v['Ae'], v_Ae_p), v_Ae_n)

            #self.v['Ai'] += self.s['Ae'].float() @ self.W['Ae_Ai']
            v_Ai_p = tf.matmul(tf.cast(self.s['Ae'], tf.float32), self.W['Ae_Ai'])
            #self.v['Ai'] -= self.v_decay['Ai'] * (self.v['Ai'] - self.rest['Ai'])
            v_Ai_n = tf.scalar_mul(self.v_decay['Ai'], tf.subtract(self.v['Ai'], self.rest['Ai']))
            self.v['Ai'] = tf.subtract(tf.add(self.v['Ai'], v_Ai_p),  v_Ai_n)

            if mode == 'train':
                # Perform STDP weight update.
                # Post-synaptic.
                # self.W['X_Ae'] += self.lrs['nu_post'] * (self.a['X'].view(self.n_input, 1) * \
                #                                          self.s['Ae'].float().view(1, self.n_neurons))
                W_X_Ae_p = tf.scalar_mul(self.lrs['nu_post'],
                                         tf.multiply(tf.transpose(self.a['X']),
                                                     tf.cast(self.s['Ae'], tf.float32)))
                # Pre-synaptic.
                # self.W['X_Ae'] -= self.lrs['nu_pre'] * (self.s['X'].float().view(self.n_input, 1) * \
                #                                         self.a['Ae'].view(1, self.n_neurons))
                W_X_Ae_n = tf.scalar_mul(self.lrs['nu_pre'],
                                         tf.multiply(tf.transpose(self.s['X']),self.a['Ae']))
                self.W['X_Ae'] = tf.subtract(tf.add(self.W['X_Ae'], W_X_Ae_p), W_X_Ae_n)

                # Ensure that weights are within [0, self.wmax].
                #self.W['X_Ae'] = torch.clamp(self.W['X_Ae'], 0, self.wmax)
                w_temp = tf.greater(self.W['X_Ae'], self.wmax)
                self.W['X_Ae'] = tf.where(w_temp, tf.ones([self.n_input,self.n_neurons]) * self.wmax, self.W['X_Ae'])
                w_temp = tf.less(self.W['X_Ae'], 0)
                self.W['X_Ae'] = tf.where(w_temp, tf.zeros([self.n_input,self.n_neurons]), self.W['X_Ae'])

                # Decay synaptic traces.
                # self.a['X'] -= self.stdp_times['X'] * self.a['X']
                self.a['X'] = tf.subtract(self.a['X'], tf.multiply(self.stdp_times['X'], self.a['X']))
                # self.a['Ae'] -= self.stdp_times['Ae'] * self.a['Ae']
                self.a['Ae'] = tf.subtract(self.a['Ae'], tf.multiply(self.stdp_times['Ae'], self.a['Ae']))

                # Decay adaptive thresholds.
                # self.theta -= self.theta_decay * self.theta
                self.theta = tf.subtract(self.theta, tf.scalar_mul(self.theta_decay, self.theta))

            # Record synaptic trace history.
            if plot:
                traces['X'][timestep, :] = self.a['X'].numpy()
                traces['Ae'][timestep, :] = self.a['Ae'].numpy()


      # Normalize weights after one iteration.
      self.normalize_weights()

      # Return recorded state variables.
      return {'Ae': spikes['Ae'], 'Ai': spikes['Ai']}


  def get_weights(self):
      return self.W['X_Ae'].numpy()


  def get_theta(self):
        return self.theta.numpy()


  def get_assignments(self):
      return self.assignments


  def normalize_weights(self):
      '''
      Normalize weights on synpases from input to excitatory layer.
      '''
      with tf.device(self.device):
        self.W['X_Ae'] = tf.multiply(self.W['X_Ae'],
                                     tf.scalar_mul(self.norm, (1 / tf.reduce_sum(self.W['X_Ae'], 0))))

  def assign_labels(self, inputs, outputs):
    '''
    Given the excitatory neuron firing history, assign them class labels.
    '''
    # Loop over all target categories.
    for j in range(10):
      # Count the number of inputs having this target.
      n_inputs = np.count_nonzero(tf.equal(inputs, j))
      if n_inputs > 0:
        # Get indices of inputs with this category.
        idxs = np.where(np.equal(inputs, j), 1, 0)
        # Calculate average firing rate per neuron, per category.
        self.rates[:, j] = 0.9 * self.rates[:, j] + np.sum(np.where(idxs, outputs, 0), 0) / n_inputs

    # Assignments of neurons are the categories for which they fire the most.
    self.assignments = np.max(self.rates, 1)



  def classify(self, spikes):
    '''
    Given the neuron assignments and the network spiking
    activity, make predictions about the data targets.
    '''
    spikes = spikes.sum(0)
    predictions = {}
    for scheme in self.voting_schemes:
        rates = np.zeros(10)
        if scheme == 'all':
            for idx in range(10):
                n_assigns = np.count_nonzero(self.assignments == idx)
                if n_assigns > 0:
                    idxs = np.nonzero(self.assignments == idx)
                    rates[idx] = np.sum(spikes[idxs]) / n_assigns
        predictions[scheme] = np.argsort(rates)[::-1]
    return predictions



if __name__ =='__main__':
	parser = argparse.ArgumentParser(description='ETH (with LIF neurons) SNN toy model simulation implemented with PyTorch.')
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--mode', type=str, default='train')
	parser.add_argument('--n_input', type=int, default=784)
	parser.add_argument('--n_neurons', type=int, default=100)
	parser.add_argument('--n_train', type=int, default=10000)
	parser.add_argument('--n_test', type=int, default=10000)
	parser.add_argument('--update_interval', type=int, default=100)
	parser.add_argument('--dt', type=float, default=1)
	parser.add_argument('--nu_pre', type=float, default=1e-4)
	parser.add_argument('--nu_post', type=float, default=1e-2)
	parser.add_argument('--c_inhib', type=float, default=17.4)
	parser.add_argument('--train_time', type=int, default=350)
	parser.add_argument('--train_rest', type=int, default=150)
	parser.add_argument('--test_time', type=int, default=350)
	parser.add_argument('--test_rest', type=int, default=150)
	parser.add_argument('--tc_pre', type=int, default=20)
	parser.add_argument('--tc_post', type=int, default=20)
	parser.add_argument('--wmax', type=float, default=1.0)
	parser.add_argument('--gpu', type=str, default='False')
	parser.add_argument('--plot', type=str, default='True')

	# Place parsed arguments in local scope.
	args = parser.parse_args()
	args = vars(args)
	locals().update(args)

	# Print out argument values.
	print('\nOptional argument values:')
	for key, value in args.items():
		print('-', key, ':', value)

	print('\n')

	# Convert string arguments into boolean datatype.
	plot = plot == 'True'
	gpu = gpu == 'True'

	# Set torch, torch-GPU, and numpy random number generator.
	np.random.seed(seed)

	# Initialize the spiking neural network.
	network = ETH(seed, n_input, n_neurons, (n_train, n_test), dt, (nu_pre, nu_post), c_inhib, \
		(train_time, train_rest, test_time, test_rest), (tc_pre, tc_post), update_interval, wmax, gpu)

	# Get training, test data from disk.
	if mode == 'train':
		data = get_labeled_data('train', train=True)
	elif mode == 'test':
		data = get_labeled_data('test', train=False)

	# Convert data into torch Tensors.
	if mode == 'train':
		X, y = data['X'][:n_train], data['y'][:n_train]
	elif mode == 'test':
		X, y = data['X'][:n_test], data['y'][:n_test]

	# Count spikes from each neuron on each example (between update intervals).
	spike_monitor = np.zeros([network.update_interval, network.n_neurons])

	# Keep track of correct classifications for performance monitoring.
	correct = { scheme : 0 for scheme in network.voting_schemes }

	if mode == 'train':
		image_time = network.sim_times['train_time']
		rest_time = network.sim_times['train_rest']
	elif mode == 'test':
		image_time = network.sim_times['test_time']
		rest_time = network.sim_times['test_rest']

	# Special "zero data" used for network rest period between examples.
	zero_data = np.zeros([rest_time, n_input])

	# Run network simulation.
	plt.ion()
	best_accuracy = 0
	start = timeit.default_timer()
	for idx, (image, target) in enumerate(zip(X, y)):
		if mode == 'train':
			if idx > 0 and idx % network.update_interval == 0:
				# Assign labels to neurons based on network spiking activity.
				network.assign_labels(y[idx - network.update_interval : idx], spike_monitor)

				# Assess performance of network on last `update_interval` examples.
				print()
				for scheme in network.performances.keys():
					network.performances[scheme].append(correct[scheme] / update_interval)  # Calculate percent correctly classified.
					correct[scheme] = 0  # Reset number of correct examples.
					print(scheme, ':', network.performances[scheme])

					# Save best accuracy.
					if network.performances[scheme][-1] > best_accuracy:
						best_accuracy = network.performances[scheme][-1]
						save_params(network.get_weights(), '.'.join(['_'.join(['X_Ae', network.fname]), 'npy']))
						save_params(network.get_theta(), '.'.join(['_'.join(['theta', network.fname]), 'npy']))
						save_assignments(network.get_assignments(), '.'.join(['_'.join(['assignments', network.fname]), 'npy']))

				print()

		# Print progress through dataset.
		if idx % 10 == 0:
			if mode == 'train':
				print('Training progress: (%d / %d) - Elapsed time: %.4f' % (idx, n_train, timeit.default_timer() - start))
			elif mode == 'test':
				print('Test progress: (%d / %d) - Elapsed time: %.4f' % (idx, n_test, timeit.default_timer() - start))

			start = timeit.default_timer()

		# Encode current input example as Poisson spike trains.
		inpt = generate_spike_train(image, network.intensity, image_time)

		# Run network on Poisson-encoded image data.
		spikes = network.run(mode=mode, inpt=inpt, time=image_time)

		# Re-run image if there isn't any network activity.
		while np.count_nonzero(spikes['Ae']) < 5:
			network.intensity += 1
			inpt = generate_spike_train(image, network.intensity, image_time)
			spikes = network.run(mode=mode, inpt=inpt, time=image_time)

		# Reset input intensity after any retries.
		network.intensity = 1

		# Classify network output (spikes) based on historical spiking activity.
		predictions = network.classify(spikes['Ae'])

		# If correct, increment counter variable.
		for scheme in predictions.keys():
			if predictions[scheme][0] == target[0]:
				correct[scheme] += 1

		# Run zero image on network for `rest_time`.
		rest_spikes = network.run(mode=mode, inpt=zero_data, time=rest_time)

		# Concatenate image and rest network data for plotting purposes.
		if plot:
			spikes = { pop : np.concatenate([spikes[pop], rest_spikes[pop]]) for pop in network.populations }

		# Add spikes from this iteration to the spike monitor
		spike_monitor[idx % network.update_interval] = np.sum(spikes['Ae'], axis=0)

		# Optionally plot the excitatory, inhibitory spiking.
		if plot:
			if idx == 0:
#				# Create figure for input image and corresponding spike trains.
#				input_figure, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=(12, 6))
#				im0 = ax0.imshow(image.reshape(network.n_input_sqrt, network.n_input_sqrt), cmap='binary')
#				ax0.set_title('Original MNIST digit (Iteration %d)' % idx)
#				im1 = ax1.imshow(np.sum(inpt, axis=0).reshape(network.n_input_sqrt, network.n_input_sqrt), cmap='binary')
#				ax1.set_title('Sum of spike trains')
#				im2 = ax2.imshow(inpt.T, cmap='binary')
#				ax2.set_title('Poisson spiking representation')
#
#				plt.tight_layout()

				# Create figure for excitatory and inhibitory neuron populations.
				spike_figure, [ax3, ax4] = plt.subplots(2, figsize=(10, 5))
				im3 = ax3.imshow(spikes['Ae'].T, cmap='binary')
				ax3.set_title('Excitatory spikes')
				im4 = ax4.imshow(spikes['Ai'].T, cmap='binary')
				ax4.set_title('Inhibitory spikes')

				plt.tight_layout()

				# Create figure for input to excitatory weights and excitatory neuron assignments.
				weights_figure, [ax5, ax6] = plt.subplots(1, 2, figsize=(10, 6))
				square_weights = get_square_weights(network.get_weights(), network.n_input_sqrt, network.n_neurons_sqrt)

				im5 = ax5.imshow(square_weights, cmap='hot_r', vmin=0, vmax=network.wmax)
				ax5.set_title('Input to excitatory weights')

				color = plt.get_cmap('RdBu', 11)
				assignments = network.get_assignments().reshape([network.n_neurons_sqrt, network.n_neurons_sqrt]).T
				im6 = ax6.matshow(assignments, cmap=color, vmin=-1.5, vmax=9.5)
				ax6.set_title('Neuron labels')

				div5 = make_axes_locatable(ax5)
				div6 = make_axes_locatable(ax6)
				cax5 = div5.append_axes("right", size="5%", pad=0.05)
				cax6 = div6.append_axes("right", size="5%", pad=0.05)

				plt.colorbar(im5, cax=cax5)
				plt.colorbar(im6, cax=cax6, ticks=np.arange(-1, 10))

				plt.tight_layout()

#				# Create figure to display plots of training accuracy over time.
#				if mode == 'train':
#					perf_figure, ax11 = plt.subplots()
#					for scheme in network.voting_schemes:
#						ax11.plot(range(len(network.performances[scheme])), network.performances[scheme], label=scheme)
#
#					ax11.set_xlim([0, n_train / update_interval + 1])
#					ax11.set_ylim([0, 1])
#					ax11.set_title('Network performance')
#					ax11.legend()
			else:
				# Re-draw plotting data after each iteration.
#				im0.set_data(image.reshape(network.n_input_sqrt, network.n_input_sqrt))
#				im1.set_data(np.sum(inpt, axis=0).reshape(network.n_input_sqrt, network.n_input_sqrt))
#				im2.set_data(inpt.T)
				im3.set_data(spikes['Ae'].T)
				im4.set_data(spikes['Ai'].T)

				square_weights = get_square_weights(network.get_weights(), network.n_input_sqrt, network.n_neurons_sqrt)

				im5.set_data(square_weights)

				assignments = network.get_assignments().reshape([network.n_neurons_sqrt, network.n_neurons_sqrt]).T
				im6.set_data(assignments)

#				if mode == 'train':
#					ax11.clear()
#					for scheme in network.voting_schemes:
#						ax11.plot(range(len(network.performances[scheme])), network.performances[scheme], label=scheme)
#
#					ax11.set_xlim([0, n_train / update_interval])
#					ax11.set_ylim([0, 1])
#					ax11.set_title('Network performance')
#					ax11.legend()

				# Update title of input digit plot to reflect current iteration.
#				ax0.set_title('Original MNIST digit (Iteration %d)' % idx)

			plt.pause(1e-8)

	results = {}
	for scheme in network.voting_schemes:
		if mode == 'train':
			results[scheme] = correct[scheme] / n_train
			print('Training accuracy for voting scheme %s:' % scheme, results[scheme])
		elif mode == 'test':
			results[scheme] = correct[scheme] / n_test
			print('Test accuracy for voting scheme %s:' % scheme, results[scheme])

	# Save out network parameters and assignments for the test phase.
	if mode == 'train':
		save_params(network.get_weights(), '.'.join(['_'.join(['X_Ae', network.fname]), 'npy']))
		save_params(network.get_theta(), '.'.join(['_'.join(['theta', network.fname]), 'npy']))
		save_assignments(network.get_assignments(), '.'.join(['_'.join(['assignments', network.fname]), 'npy']))
