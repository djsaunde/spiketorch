# Dispatch several jobs on the UMass CICS Gypsum cluster.
for seed in 1 2 3 4 5
do
	for n_neurons in 100 225 400 625 900
	do
		for n_train in 60000 120000 180000
		do
			sbatch run_eth.sh $seed $n_neurons $n_train
		done
	done
done
