# Dispatch several jobs on the UMass CICS Gypsum cluster.
for seed in 1 2 3
do
	for n_neurons in 225 400 625 900
	do
		for n_train in 60000
		do
			for c_inhib in -15.0 -17.5 -20.0
			do
				for c_excite in 20.0 22.5 25.0
				do
					for wmax in 0.5 0.75 1.0 1.25 1.5
					do
						sbatch run.sh $seed $n_neurons $n_train
					done		
				done
			done
		done
	done
done