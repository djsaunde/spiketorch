import torch
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.ion()


def plot_input(image, inpt, ims=None, figsize=(12, 6)):
	if not ims:
		_, axes = plt.subplots(1, 2, figsize=figsize)
		ims = axes[0].imshow(image, cmap='binary'), axes[1].imshow(inpt, cmap='binary')
		axes[0].set_title('Current image')
		axes[1].set_title('Poisson spiking representation')
		plt.tight_layout()
	else:
		ims[0].set_data(image)
		ims[1].set_data(inpt)

	return ims


def plot_spikes(exc, inh, ims=None, figsize=(10, 5)):
	if not ims:
		_, axes = plt.subplots(2, figsize=figsize)
		ims = axes[0].imshow(exc, cmap='binary'), axes[1].imshow(inh, cmap='binary')
		axes[0].set_title('Excitatory spikes')
		axes[1].set_title('Inhibitory spikes')
		plt.tight_layout()
	else:
		ims[0].set_data(exc)
		ims[1].set_data(inh)

	return ims


def plot_weights(weights, assignments, wmax=1, ims=None, figsize=(10, 6)):
	if not ims:
		_, axes = plt.subplots(1, 2, figsize=figsize)
		
		color = plt.get_cmap('RdBu', 11)
		ims = axes[0].imshow(weights, cmap='hot_r', vmin=0, vmax=wmax), axes[1].matshow(assignments, cmap=color, vmin=-1.5, vmax=9.5)
		divs = make_axes_locatable(axes[0]), make_axes_locatable(axes[1])
		caxs = divs[0].append_axes("right", size="5%", pad=0.05), divs[1].append_axes("right", size="5%", pad=0.05)

		plt.colorbar(ims[0], cax=caxs[0])
		plt.colorbar(ims[0], cax=caxs[1], ticks=np.arange(-1, 10))
		plt.tight_layout()
	else:
		ims[0].set_data(weights)
		ims[1].set_data(assignments)

	return ims


def plot_performance(performances, ax=None, figsize=(6, 6)):
	if not ax:
		_, ax = plt.subplots(figsize=figsize)
	else:
		ax.clear()

	for scheme in performances:
		ax.plot(range(len(performances[scheme])), [100 * p for p in performances[scheme]], label=scheme)

	ax.set_ylim([0, 100])
	ax.set_title('Estimated classification accuracy')
	ax.set_xlabel('No. of examples')
	ax.set_ylabel('Accuracy')
	ax.legend()

	return ax


def plot_voltages(exc, inh, axes=None, figsize=(8, 8)):
	if axes is None:
		_, axes = plt.subplots(2, 1, figsize=figsize)
		axes[0].set_title('Excitatory voltages')
		axes[1].set_title('Inhibitory voltages')
	
	axes[0].clear(); axes[1].clear()
	axes[0].plot(exc), axes[1].plot(inh)

	return axes