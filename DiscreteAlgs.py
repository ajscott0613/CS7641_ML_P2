import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import mlrose_hiive
from opt_problems import Knapsack, Queens


# GLOBAL VARIABLES
ALGS = [0, 1, 2, 3]
# PROBS = [0, 1, 2]
PROBS = [1]


# Support Function to Display Results
def display_results(idx, alg_out):

	if idx == 0:
		print("### Running Random Hill Climbing Algorithm ###")
	elif idx == 1:
		print("### Running Simulated Annealing Algorithm ###")
	elif idx == 2:
		print("### Running Genetic Algorithm ###")
	elif idx == 3:
		print("### Running MIMIC Algorithm ###")
	best_state = alg_out[0]
	best_fitness = alg_out[1]
	print("	best state: ", best_state)
	print("	best fitness score: ", best_fitness)
	l_curve = alg_out[2]
	# print(l_curve)
	# fitness = np.zeros(np.shape(l_curve)[0])
	# itr = np.zeros(np.shape(l_curve)[0])
	# for i in range(np.shape(l_curve)[0]):
	# 	fitness[i] = l_curve[i][0]
	# 	itr[i] = l_curve[i][1]
	# plt.plot(itr, fitness)
	# plt.show()

	# print(" ################################################ ")


for prob in PROBS:

	if prob == 0:

		# Run First Discrete Optimization Problem
		print("**** KNAPSACK PROBLEM ****")
		print(" ")
		print(" ")
		print(" ")
		print(" ")

		ks = Knapsack(verbose=True)

		# Initialize custom fitness function object
		fitness_cust = mlrose_hiive.CustomFitness(ks.fitness)

		problem = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)

		# Define decay schedule
		schedule = mlrose_hiive.ExpDecay()

		# Define initial state
		init_state = np.random.randint(2, size=(10,))


		for i in ALGS:

			if i == 0:
				rhc_out = mlrose_hiive.random_hill_climb(problem, max_attempts = 10, max_iters = 10000,
					init_state = init_state, curve=True, random_state = 1)
				display_results(i, rhc_out)

			elif i == 1:
				sa_out = mlrose_hiive.simulated_annealing(problem, schedule = schedule,
		            max_attempts = 10, max_iters = 10000,
		            init_state = init_state, curve=True, random_state = 1)
				display_results(i, sa_out)

			elif i == 2:
				ga_out = mlrose_hiive.genetic_alg(problem, pop_size=8, mutation_prob=0.1, max_attempts=10, 
					max_iters=10000, curve=True, random_state=None)
				display_results(i, ga_out)

			elif i == 3:
				mimic_out = mlrose_hiive.mimic(problem, pop_size=8, keep_pct=0.2, max_attempts=10, max_iters=10000, 
					curve=True, random_state=None)
				display_results(i, mimic_out)


		# Take average over 100 runs for Knapsack problem
		trials = 100
		rhc_all = np.zeros(trials)
		sa_all = np.zeros(trials)
		ga_all = np.zeros(trials)
		mimic_all = np.zeros(trials)

		for i in range(trials):

			ks = Knapsack(verbose=False)

			# Initialize custom fitness function object
			fitness_cust = mlrose_hiive.CustomFitness(ks.fitness)

			problem = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)

			# Define decay schedule
			schedule = mlrose_hiive.ExpDecay()

			# Define initial state
			init_state = np.random.randint(2, size=(10,))



			rhc_out = mlrose_hiive.random_hill_climb(problem, max_attempts = 10, max_iters = 10000,
				init_state = init_state, curve=True, random_state = 1)

			sa_out = mlrose_hiive.simulated_annealing(problem, schedule = schedule,
			    max_attempts = 10, max_iters = 10000,
			    init_state = init_state, curve=True, random_state = 1)

			ga_out = mlrose_hiive.genetic_alg(problem, pop_size=8, mutation_prob=0.1, max_attempts=10, 
				max_iters=10000, curve=True, random_state=None)

			mimic_out = mlrose_hiive.mimic(problem, pop_size=8, keep_pct=0.2, max_attempts=10, max_iters=10000, 
				curve=True, random_state=None)

			rhc_all[i] = rhc_out[1]
			sa_all[i] = sa_out[1]
			ga_all[i] = ga_out[1]
			mimic_all[i] = mimic_out[1]

		rhc_mean = np.mean(rhc_all)
		sa_mean = np.mean(sa_all)
		ga_mean = np.mean(ga_all)
		mimic_mean = np.mean(mimic_all)

		print("Random Hill Climb - Average Fitness over ", trials, " runs:	", rhc_mean)
		print("Simulated Annealing - Average Fitness over ", trials, " runs:	", sa_mean)
		print("Genetic Algorithm - Average Fitness over ", trials, " runs:	", ga_mean)
		print("MIMIC - Average Fitness over ", trials, " runs:	", mimic_mean)



	if prob == 1:
		# Run Second Discrete Optimization Problem
		print("**** QUEENS PROBLEM ****")
		print(" ")
		print(" ")
		print(" ")
		print(" ")

		queens = Queens()

		# Initialize custom fitness function object
		fitness_cust = mlrose_hiive.CustomFitness(queens.fitness)

		problem = mlrose_hiive.DiscreteOpt(length = 8, fitness_fn = fitness_cust, maximize = True, max_val = 8)


		# Define decay schedule
		schedule = mlrose_hiive.ExpDecay()

		# Define initial state
		init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])


		for i in ALGS:

			if i == 0:
				rhc_out = mlrose_hiive.random_hill_climb(problem, max_attempts = 10, max_iters = 10000,
					init_state = init_state, curve=True, random_state = 1)
				display_results(i, rhc_out)
				queens.display_board(rhc_out[0])

			elif i == 1:
				sa_out = mlrose_hiive.simulated_annealing(problem, schedule = schedule,
		            max_attempts = 10, max_iters = 10000,
		            init_state = init_state, curve=True, random_state = 1)
				display_results(i, sa_out)
				queens.display_board(sa_out[0])

			elif i == 2:
				ga_out = mlrose_hiive.genetic_alg(problem, pop_size=8, mutation_prob=0.1, max_attempts=10, 
					max_iters=10000, curve=True, random_state=None)
				display_results(i, ga_out)
				queens.display_board(ga_out[0])

			elif i == 3:
				mimic_out = mlrose_hiive.mimic(problem, pop_size=8, keep_pct=0.2, max_attempts=10, max_iters=10000, 
					curve=True, random_state=None)
				display_results(i, mimic_out)
				queens.display_board(mimic_out[0])



	if prob == 2:
		# Run Second Discrete Optimization Problem
		print("**** 4 Peaks PROBLEM ****")
		print(" ")
		print(" ")
		print(" ")
		print(" ")

		n = 20

		# Initialize custom fitness function object
		fitness = mlrose_hiive.FourPeaks(t_pct=0.15)
		print("Threshold = ", int(n*0.15))

		problem = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)


		# Define decay schedule
		schedule = mlrose_hiive.ExpDecay()

		# Define initial state
		init_state = np.random.randint(2, size=(n,))


		for i in ALGS:

			if i == 0:
				rhc_out = mlrose_hiive.random_hill_climb(problem, max_attempts = 10, max_iters = 10000,
					init_state = init_state, curve=True, random_state = 1)
				display_results(i, rhc_out)

			elif i == 1:
				sa_out = mlrose_hiive.simulated_annealing(problem, schedule = schedule,
		            max_attempts = 10, max_iters = 10000,
		            init_state = init_state, curve=True, random_state = 1)
				display_results(i, sa_out)

			elif i == 2:
				ga_out = mlrose_hiive.genetic_alg(problem, pop_size=8, mutation_prob=0.1, max_attempts=10, 
					max_iters=10000, curve=True, random_state=None)
				display_results(i, ga_out)

			elif i == 3:
				mimic_out = mlrose_hiive.mimic(problem, pop_size=8, keep_pct=0.2, max_attempts=10, max_iters=10000, 
					curve=True, random_state=None)
				display_results(i, mimic_out)