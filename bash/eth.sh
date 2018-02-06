#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/spiketorch/job_reports/eth_%j.out

seed=${1:-0}
n_neurons=${2:-100}
n_train=${3:-60000}
c_excite=${4:-22.5}
c_inhib=${5:-17.5}
wmax=${6:-1.0}

source activate py36
cd ../examples/

python eth.py --mode train --seed $seed --n_neurons $n_neurons --n_train $n_train --c_excite $c_excite --c_inhib $c_inhib --wmax $wmax

python eth.py --mode test --seed $seed --n_neurons $n_neurons --n_train $n_train --c_excite $c_excite --c_inhib $c_inhib --wmax $wmax
