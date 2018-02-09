import os
import numpy
import argparse
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.abspath(os.path.join('..', 'spiketorch')))
sys.path.append(os.path.abspath(os.path.join('..', 'spiketorch', 'network')))

from util import *

params_path = os.path.join('..', '..', 'params')


parser = argparse.ArgumentParser()
parser.add_argument('--n_neurons', default=100, type=int)
parser.add_argument('--n_train', default=10000, type=int)
parser.add_argument('--seed', default=0, type=int)

# Place parsed arguments in local scope.
args = parser.parse_args()
args = vars(args)
locals().update(args)

# Print out argument values.
print('\nOptional argument values:')
for key, value in args.items():
	print('-', key, ':', value)

print('\n')

n_input_sqrt = 28
n_neurons_sqrt = int(np.sqrt(n_neurons))

# Generic filename for saving out weights and other parameters. 
fname = '_'.join([ str(n_neurons), str(n_train), str(seed) ])

# Get weight and theta parameters and neuron assignments from disk.
weights = load_params('.'.join(['_'.join(['X_Ae', fname]), 'npy']))
theta = load_params('.'.join(['_'.join(['theta', fname]), 'npy']))
assignments = load_assignments('.'.join(['_'.join(['assignments', fname]), 'npy']))

# Get square weights, theta, and assignments.
weights = get_square_weights(weights, n_input_sqrt, n_neurons_sqrt)
theta = theta.reshape([n_neurons_sqrt, n_neurons_sqrt]).T
assignments = assignments.reshape([n_neurons_sqrt, n_neurons_sqrt]).T

# Assignments colormap.
color = plt.get_cmap('RdBu', 11)

# Plot the weights and theta parameters.
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(20, 10))
im1 = ax1.imshow(weights, cmap='hot_r', vmin=0, vmax=np.max(weights))
im2 = ax2.imshow(theta, cmap='hot_r')
im3 = ax3.imshow(assignments, cmap=color, vmin=-1.5, vmax=9.5)
div1 = make_axes_locatable(ax1)
div2 = make_axes_locatable(ax2)
div3 = make_axes_locatable(ax3)
cax1 = div1.append_axes("right", size="5%", pad=0.05)
cax2 = div2.append_axes("right", size="5%", pad=0.05)
cax3 = div3.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im1, cax=cax1)
plt.colorbar(im2, cax=cax2)
plt.colorbar(im3, cax=cax3, ticks=np.arange(-1, 10))

ax1.set_title('Excitatory neuron filters')
ax2.set_title('Excitatory neuron adaptive thresholds')
ax3.set_title('Neuron digit labels')

plt.tight_layout()
plt.show()
