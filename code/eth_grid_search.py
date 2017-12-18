n_train=60000

for seed in 1 2 3
do
	for n_neurons in 100 225 400 625
	do
		python eth.py --seed $seed --n_neurons $n_neurons --n_train $n_train &
	done
done
