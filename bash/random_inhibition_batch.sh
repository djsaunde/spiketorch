# Dispatch several jobs on the UMass CICS Gypsum cluster.
for seed in 1 2 3
do
	for n_exc in 400
	do
		for n_inh in 40 100 200 400
		do
			for n_inh_synapse in 25 50 100 200 300 400
			do
				for n_train in 60000
				do
					sbatch random_inhibition.sh $seed $n_exc $n_inh $n_inh_synapse $n_train
				done
			done
		done
	done
done
