#!/bin/bash
#
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/spiketorch/job_reports/eth_%j.out

seed=${1:-0}
n_neurons=${2:-100}
n_train=${3:-60000}

source activate py36
cd ../examples/

python eth.py --mode train --seed $seed --n_neurons $n_neurons --n_train $n_train

python eth.py --mode test --seed $seed --n_neurons $n_neurons --n_train $n_train
