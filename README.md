# SpikeTorch

Python package used for simulating spiking neural networks (SNNs) in [PyTorch](http://pytorch.org/).

At the moment, the focus is on replicating the SNN described in [Unsupervised learning of digit recognition using spike-timing-dependent plasticity](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#) (original code found [here](https://github.com/peter-u-diehl/stdp-mnist), extensions thereof found in my previous project repository [here](https://github.com/djsaunde/stdp-mnist)).

At the moment, we are only interesting in applying SNNs to simple machine learning tasks, but the code can be used for any particular purpose.

## Background

One computational challenge is simulating time-dependent neuronal dynamics. This is typically accounted for by solving ordinary differential equations (ODEs) defining neuronal dynamics. PyTorch does not explicitly support the solution of differential equations (as opposed to [`brian2`](https://github.com/brian-team/brian2), for example), but we can convert the ODEs governing neural dynamics into difference equations and solve them at regular, short intervals (some `dt` on the order of 1 millisecond) as an approximation. Of course, under the hood, packages like `brian2` are doing the same thing.

The concept that the neuron spike ordering and their relative timing encode information is a central hypothesis in neuroscience. [Markram et al. (1997)](http://www.caam.rice.edu/~caam415/lec_gab/g4/markram_etal98.pdf) proposed that synapses between neurons should strengthen or degrade based on this relative timing, and prior to that, [Donald Hebb](https://en.wikipedia.org/wiki/Donald_O._Hebb) proposed the theory of Hebbian learning, often paraphrased as "Neurons that fire together wire together." Markram et al.'s extension of the Hebbian theory came to be known as spike-timing-dependent plasticity (STDP). 

We are interested in applying spiking neural networks to machine learning problems. For now, we use the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/), which, though somewhat antiquated, is simple enough to develop new machine learning techniques on.

## Requirements

All code was developed using Python 3.6.x, and will fail if run with Python 2.x. Use `pip install -r requirements.txt` to download all project dependencies. You may have to consult the [PyTorch webpage](http://pytorch.org/) in order to get the right installation for your machine. 

## Setting things up

To begin, download and unzip the MNIST dataset by running `./data/get_MNIST.sh`. To replicate the SNN from the [above paper](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#), run `python code/eth.py`. There are a number of optional command-line arguments which can be passed in, including `--plot` (displays useful monitoring figures), `--n_neurons [int]` (number of excitatory, inhibitory neurons simulated), `--mode ['train' | 'test']` (sets network operation to the training or testing phase), and more. Run `python code/eth.py --help` for more information on the command-line arguments.

__Note__: This is a work in progress, including the replication script `code/eth.py`.
