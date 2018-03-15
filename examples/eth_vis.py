import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.abspath(os.path.join('..', 'spiketorch')))
sys.path.append(os.path.abspath(os.path.join('..', 'spiketorch', 'network')))

from network import load_params, load_assignments, get_square_weights

params_path = os.path.join('..', 'params', 'eth')
assign_path = os.path.join('..', 'assignments', 'eth')

parser = argparse.ArgumentParser()
parser.add_argument('--n_neurons', default=100, type=int)
parser.add_argument('--n_train', default=10000, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--c_excite', type=float, default=22.5)
parser.add_argument('--c_inhib', type=float, default=17.5)
parser.add_argument('--wmax', type=float, default=1.0)

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
fname = '_'.join(map(str, [n_neurons, n_train, seed, c_inhib, c_excite, wmax]))

# Get weight and theta parameters and neuron assignments from disk.
weights = torch.from_numpy(load_params(params_path, fname, 'X_Ae'))
theta = torch.from_numpy(load_params(params_path, fname, 'theta'))
assignments = torch.from_numpy(load_assignments(assign_path, fname))

# Get square weights, theta, and assignments.
weights = get_square_weights(weights, n_neurons_sqrt)
theta = theta.view(n_neurons_sqrt, n_neurons_sqrt)
assignments = assignments.view(n_neurons_sqrt, n_neurons_sqrt)

# Assignments colormap.
color = plt.get_cmap('RdBu', 11)

# # Plot the weights, theta values, and assignments.
# fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(20, 10))
# im1 = ax1.matshow(weights, cmap='hot_r', vmin=0, vmax=torch.max(weights))
# im2 = ax2.matshow(theta, cmap='hot_r')
# im3 = ax3.matshow(assignments, cmap=color, vmin=-1.5, vmax=9.5)
# div1 = make_axes_locatable(ax1)
# div2 = make_axes_locatable(ax2)
# div3 = make_axes_locatable(ax3)
# cax1 = div1.append_axes("right", size="5%", pad=0.05)
# cax2 = div2.append_axes("right", size="5%", pad=0.05)
# cax3 = div3.append_axes("right", size="5%", pad=0.05)

# plt.colorbar(im1, cax=cax1)
# plt.colorbar(im2, cax=cax2)
# plt.colorbar(im3, cax=cax3, ticks=np.arange(-1, 10))

# ax1.set_title('Excitatory neuron filters')
# ax2.set_title('Excitatory neuron adaptive thresholds')
# ax3.set_title('Neuron digit labels')

# plt.tight_layout()
# plt.show()

# Plot the weights and assignments.
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(15, 8))
im1 = ax1.matshow(weights, cmap='hot_r', vmin=0, vmax=wmax)
im2 = ax2.matshow(assignments, cmap=color, vmin=-1.5, vmax=9.5)
div1 = make_axes_locatable(ax1)
div2 = make_axes_locatable(ax2)
cax1 = div1.append_axes("right", size="5%", pad=0.05)
cax2 = div2.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im1, cax=cax1)
plt.colorbar(im2, cax=cax2, ticks=np.arange(-1, 10))

ax1.set_title('Excitatory neuron filters', fontdict={'fontsize' : 16})
ax2.set_title('Neuron digit labels', fontdict={'fontsize' : 16})

ax1.set_xticks(()); ax1.set_yticks(())
ax2.set_xticks(()); ax2.set_yticks(())

plt.tight_layout()
plt.show()