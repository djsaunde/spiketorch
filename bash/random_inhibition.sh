#!/bin/bash
#
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/spiketorch/job_reports/eth_%j.out

seed=${1:-0}
n_exc=${2:-100}
n_inh=${3:-25}
n_inh_synapse=${4:-10}
n_train=${5:-60000}

source activate py36
cd ../examples/

python random_inhibition.py --mode train --seed $seed --n_exc $n_exc --n_inh $n_inh --n_inh_synapse $n_inh_synapse --n_train $n_train

python random_inhibition.py --mode test --seed $seed --n_exc $n_exc --n_inh $n_inh --n_inh_synapse $n_inh_synapse --n_train $n_train
