import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import mlrose_hiive
from opt_problems import Knapsack, Queens
import sys, getopt
import time
import utils


# GLOBAL VARIABLES
ALGS = [0, 1, 2, 3]
# PROBS = [0, 1, 2]
PROBS = []
ALGS = []

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


if __name__ == "__main__":

	# if sys.argv[0] == '-p':
	# 	PROBS = [0]

	# print(PROBS)

	# print(sys.argv[1:])
	# if sys.argv[1] == '-p':
	# 	PROBS = [int(sys.argv[2])]

	for i in range(len(sys.argv)):
		if sys.argv[i] == '-p':
			PROBS = [int(sys.argv[i+1])]
		if sys.argv[i] == '-a':
			ALGS = [int(sys.argv[i+1])]
	
	if len(PROBS) == 0:
		PROBS = [0, 1, 2]
	if len(ALGS) == 0:
		ALGS = [0, 1, 2, 3]


	for prob in PROBS:

		if prob == 0:

			# Run First Discrete Optimization Problem
			print("**** KNAPSACK PROBLEM ****")
			print(" ")
			print(" ")
			print(" ")
			print(" ")

			n = 5
			# weight=[10, 5, 2, 8, 15]
			# value=[1, 2, 3, 4, 5]
			# max_w_p=0.6
			# ks = Knapsack(verbose=True, n=4, weight=weight, value=value, max_w_p=max_w_p)
			max_attempts = 100
			max_iters = np.Inf

			# Initialize custom fitness function object
			# fitness_cust = mlrose_hiive.CustomFitness(ks.fitness)


			weights = [2, 10, 3, 6, 18]
			values = [1, 20, 3, 14, 100]
			max_weight_pct = 0.39
			fitness_cust = mlrose_hiive.Knapsack(weights, values, max_weight_pct)
			init_state = np.array([1, 0, 2, 1, 0])

			# problem = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)

			# Define decay schedule
			schedule = mlrose_hiive.ExpDecay()

			# Define initial state
			# init_state = np.random.randint(2, size=(n,))


			# for i in ALGS:

			# 	if i == 0:
			# 		problem1 = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)
			# 		rhc_out = mlrose_hiive.random_hill_climb(problem1, max_attempts=max_attempts, max_iters=max_iters,
			# 			init_state = init_state, curve=True, random_state = 1)
			# 		display_results(i, rhc_out)

			# 	elif i == 1:
			# 		problem2 = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)
			# 		sa_out = mlrose_hiive.simulated_annealing(problem2, schedule = schedule,
			#             max_attempts=max_attempts, max_iters=max_iters,
			#             init_state = init_state, curve=True, random_state = 1)
			# 		display_results(i, sa_out)

			# 	elif i == 2:
			# 		problem3 = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)
			# 		ga_out = mlrose_hiive.genetic_alg(problem3, pop_size=200, mutation_prob=0.1, max_attempts=max_attempts, 
			# 			max_iters=max_iters, curve=True, random_state=None)
			# 		display_results(i, ga_out)

			# 	elif i == 3:
			# 		problem4 = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)
			# 		mimic_out = mlrose_hiive.mimic(problem4, pop_size=200, keep_pct=0.2, max_attempts=max_attempts, 
			# 			max_iters=max_iters, curve=True, random_state=None)
			# 		display_results(i, mimic_out)


			# # Take average over 100 runs for Knapsack problem
			# trials = 25
			# rhc_all = np.zeros(trials)
			# sa_all = np.zeros(trials)
			# ga_all = np.zeros(trials)
			# mimic_all = np.zeros(trials)

			# rhc_time = np.zeros(trials)
			# sa_time = np.zeros(trials)
			# ga_time = np.zeros(trials)
			# mimic_time = np.zeros(trials)

			# for i in range(trials):

			# 	ks = Knapsack(verbose=False)

			# 	# Initialize custom fitness function object
			# 	fitness_cust = mlrose_hiive.CustomFitness(ks.fitness)

			# 	problem = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)

			# 	# Define decay schedule
			# 	schedule = mlrose_hiive.ExpDecay()

			# 	# Define initial state
			# 	init_state = np.random.randint(2, size=(10,))



			# 	problem1 = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)
			# 	t0 = time.process_time()
			# 	rhc_out = mlrose_hiive.random_hill_climb(problem1, max_attempts=max_attempts, max_iters=max_iters,
			# 		init_state = init_state, curve=True, random_state = 1)
			# 	t1 = time.process_time()
			# 	rhc_time[i] = t1 - t0

			# 	problem2 = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)
			# 	t0 = time.process_time()
			# 	sa_out = mlrose_hiive.simulated_annealing(problem2, schedule = schedule,
		 #            max_attempts=max_attempts, max_iters=max_iters,
		 #            init_state = init_state, curve=True, random_state = 1)
			# 	t1 = time.process_time()
			# 	sa_time[i] = t1 - t0

			# 	problem3 = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)
			# 	t0 = time.process_time()
			# 	ga_out = mlrose_hiive.genetic_alg(problem3, pop_size=200, mutation_prob=0.1, max_attempts=max_attempts, 
			# 		max_iters=max_iters, curve=True, random_state=None)
			# 	t1 = time.process_time()
			# 	ga_time[i] = t1 - t0

			# 	problem4 = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)
			# 	t0 = time.process_time()
			# 	mimic_out = mlrose_hiive.mimic(problem4, pop_size=200, keep_pct=0.2, max_attempts=max_attempts, 
			# 		max_iters=max_iters, curve=True, random_state=None)
			# 	t1 = time.process_time()
			# 	mimic_time[i] = t1 - t0

			# 	rhc_all[i] = rhc_out[1]
			# 	sa_all[i] = sa_out[1]
			# 	ga_all[i] = ga_out[1]
			# 	mimic_all[i] = mimic_out[1]

			# rhc_mean = np.mean(rhc_all)
			# sa_mean = np.mean(sa_all)
			# ga_mean = np.mean(ga_all)
			# mimic_mean = np.mean(mimic_all)

			# rhc_mean_time = np.mean(rhc_time)
			# sa_mean_time = np.mean(sa_time)
			# ga_mean_time = np.mean(ga_time)
			# mimic_mean_time = np.mean(mimic_time)

			# print("Random Hill Climb - Average Fitness over ", trials, " runs:	", rhc_mean)
			# print("	Average Process Time: ", np.round(rhc_mean_time, 4))
			# print("Simulated Annealing - Average Fitness over ", trials, " runs:	", sa_mean)
			# print("	Average Process Time: ", np.round(sa_mean_time, 4))
			# print("Genetic Algorithm - Average Fitness over ", trials, " runs:	", ga_mean)
			# print("	Average Process Time: ", np.round(ga_mean_time, 4))
			# print("MIMIC - Average Fitness over ", trials, " runs:	", mimic_mean)
			# print("	Average Process Time: ", np.round(mimic_mean_time, 4))



						# Take average over 100 runs for Knapsack problem
			trials = 5
			rhc_all = np.zeros(trials)
			sa_all = np.zeros(trials)
			ga_all = np.zeros(trials)
			mimic_all = np.zeros(trials)

			rhc_time = np.zeros(trials)
			sa_time = np.zeros(trials)
			ga_time = np.zeros(trials)
			mimic_time = np.zeros(trials)

			for i in ALGS:

				if i == 0:

					for j in range(trials):
						save_file = False
						problem_mim1 = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness_cust, maximize=True, max_val=2)
						t0 = time.process_time()
						rhc_out = mlrose_hiive.random_hill_climb(problem_mim1, max_attempts=1000, max_iters=1000,
							init_state = init_state, curve=True, random_state = 1)
						t1 = time.process_time()
						if j == trials - 1:
							save_file = True
						print(rhc_out[2])
						utils.multi_plot(rhc_out[2], 'knapsack_rhc', 'Knapsack: Random Hill Climb', j, trials, save_file)
						rhc_all[j] = rhc_out[1]
						rhc_time[j] = np.round(t1 - t0, 4)
					print("** FINISHED RHC **")

				elif i == 1:

					for j in range(trials):
						save_file = False
						problem_mim2 = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness_cust, maximize=True, max_val=2)
						t0 = time.process_time()
						sa_out = mlrose_hiive.simulated_annealing(problem_mim2, schedule = schedule,
				            max_attempts=500, max_iters=1000,
				            init_state = init_state, curve=True, random_state = 1)
						t1 = time.process_time()
						if j == trials - 1:
							save_file = True
						utils.multi_plot(sa_out[2], 'knapsack_sa', 'Knapsack: Simulated Annealing', j, trials, save_file)
						sa_all[j] = sa_out[1]
						sa_time[j] = np.round(t1 - t0, 4)
					print("** FINISHED SA **")

				elif i == 2:

					for j in range(trials):
						save_file = False
						problem_mim3 = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness_cust, maximize=True, max_val=2)
						t0 = time.process_time()
						ga_out = mlrose_hiive.genetic_alg(problem_mim3, pop_size=2000, mutation_prob=0.1, max_attempts=500, 
							max_iters=100, curve=True, random_state=None)
						t1 = time.process_time()
						if j == trials - 1:
							save_file = True
						utils.multi_plot(ga_out[2], 'knapsack_ga', 'Knapsack: Genetic Algorithm',j, trials, save_file)
						ga_all[j] = ga_out[1]
						ga_time[j] = np.round(t1 - t0, 4)
						# plt.plot(ga_out[2])
					# pass
					print("** FINISHED GA **")
					# utils.plot_iters_vs_clock(ga_out[2], 'knapsack_ga_time', 'Knapsack: Genetic Algorithm')

				elif i == 3:
					for j in range(trials):
						save_file = False
						problem_mim4 = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness_cust, maximize=True, max_val=2)
						t0 = time.process_time()
						mimic_out = mlrose_hiive.mimic(problem_mim4, pop_size=2000, keep_pct=0.2, max_attempts=2000, max_iters=50, 
							curve=True, random_state=None)
						t1 = time.process_time()
						if j == trials - 1:
							save_file = True
						# utils.multi_plot_fevals(mimic_out[2], 'knapsack_mimic_fevals', 'Knapsack: MIMIC',j, trials, save_file)
						utils.multi_plot(mimic_out[2], 'knapsack_mimic', 'Knapsack: MIMIC',j, trials, save_file)
						mimic_all[j] = mimic_out[1]
						mimic_time[j] = np.round(t1 - t0, 4)
					print("** FINISHED MIMIC **")
					# utils.plot_iters_vs_clock(mimic_out[2], 'knapsack_mimic_time', 'Knapsack: MIMIC')


			rhc_mean = np.mean(rhc_all)
			sa_mean = np.mean(sa_all)
			ga_mean = np.mean(ga_all)
			mimic_mean = np.mean(mimic_all)

			rhc_mean_time = np.mean(rhc_time)
			sa_mean_time = np.mean(sa_time)
			ga_mean_time = np.mean(ga_time)
			mimic_mean_time = np.mean(mimic_time)

			print("Random Hill Climb - Average Fitness over ", trials, " runs:	", rhc_mean)
			print("	Average Process Time: ", np.round(rhc_mean_time, 4))
			print("Simulated Annealing - Average Fitness over ", trials, " runs:	", sa_mean)
			print("	Average Process Time: ", np.round(sa_mean_time, 4))
			print("Genetic Algorithm - Average Fitness over ", trials, " runs:	", ga_mean)
			print("	Average Process Time: ", np.round(ga_mean_time, 4))
			print("MIMIC - Average Fitness over ", trials, " runs:	", mimic_mean)
			print("	Average Process Time: ", np.round(mimic_mean_time, 4))



		if prob == 1:
			# Run Second Discrete Optimization Problem
			print("**** QUEENS PROBLEM ****")
			print(" ")
			print(" ")
			print(" ")
			print(" ")

			print("etete")

			n_queens = 8
			queens = Queens(n_queens=n_queens)
			max_attempts = 1000
			max_iters = np.Inf

			# Initialize custom fitness function object
			fitness_cust = mlrose_hiive.CustomFitness(queens.fitness)

			problem = mlrose_hiive.DiscreteOpt(length=n_queens, fitness_fn=fitness_cust, maximize=True, max_val=n_queens)


			# Define decay schedule
			schedule = mlrose_hiive.ExpDecay()

			# Define initial state
			init_state = np.array(list(range(n_queens)))
			# init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])


			for i in ALGS:

				if i == 0:
					problem_nq1 = mlrose_hiive.DiscreteOpt(length=n_queens, fitness_fn=fitness_cust, maximize=True, max_val=n_queens)
					t0 = time.process_time()
					rhc_out = mlrose_hiive.random_hill_climb(problem_nq1, max_attempts=max_attempts, max_iters=max_iters,
						init_state = init_state, curve=True, random_state = 1)
					t1 = time.process_time()
					display_results(i, rhc_out)
					print("	Process Time: ", np.round(t1 - t0, 4))
					utils.plot_fitness(rhc_out[2], 'NQueens_rhc', 'N-Queens: Random Hill Climb')
					# queens.display_board(rhc_out[0])

				elif i == 1:
					t0 = time.process_time()
					problem_nq2 = mlrose_hiive.DiscreteOpt(length=n_queens, fitness_fn=fitness_cust, maximize=True, max_val=n_queens)
					sa_out = mlrose_hiive.simulated_annealing(problem_nq2, schedule = schedule,
			            max_attempts=max_attempts, max_iters=max_iters,
			            init_state = init_state, curve=True, random_state = 1)
					t1 = time.process_time()
					display_results(i, sa_out)
					print("	Process Time: ", np.round(t1 - t0, 4))
					# utils.plot_fitness(sa_out[2], 'NQueens_sa', 'N-Queens: Simulated Annealing')
					utils.plot_iters_vs_clock(sa_out[2], 'NQueens_sa_time', 'N-Queens: Simulated Annealing')
					# queens.display_board(sa_out[0])

				elif i == 2:
					t0 = time.process_time()
					problem_nq3 = mlrose_hiive.DiscreteOpt(length=n_queens, fitness_fn=fitness_cust, maximize=True, max_val=n_queens)
					ga_out = mlrose_hiive.genetic_alg(problem_nq3, pop_size=1000, mutation_prob=0.1, max_attempts=8, 
						max_iters=max_iters, curve=True, random_state=None)
					t1 = time.process_time()
					display_results(i, ga_out)
					print("	Process Time: ", np.round(t1 - t0, 4))
					# utils.plot_fitness(ga_out[2], 'NQueens_ga', 'N-Queens: Genetic Algorithms')
					utils.plot_iters_vs_clock(ga_out[2], 'NQueens_ga_time', 'N-Queens: Genetic Algorithms')
					# queens.display_board(ga_out[0])

				elif i == 3:
					t0 = time.process_time()
					problem_nq4 = mlrose_hiive.DiscreteOpt(length=n_queens, fitness_fn=fitness_cust, maximize=True, max_val=n_queens)
					mimic_out = mlrose_hiive.mimic(problem_nq4, pop_size=1000, keep_pct=0.2, max_attempts=max_attempts, max_iters=4, 
						curve=True, random_state=None)
					t1 = time.process_time()
					display_results(i, mimic_out)
					print("	Process Time: ", np.round(t1 - t0, 4))
					utils.plot_fitness(mimic_out[2], 'NQueens_mimic', 'N-Queens: MIMIC')
					# queens.display_board(mimic_out[0])



		if prob == 2:
			# Run Second Discrete Optimization Problem
			print("**** 4 Peaks PROBLEM ****")
			print(" ")
			print(" ")
			print(" ")
			print(" ")

			n = 50
			max_attempts = 500
			max_iters = np.Inf

			# Initialize custom fitness function object
			fitness = mlrose_hiive.FourPeaks(t_pct=0.15)
			print("Threshold = ", int(n*0.15))

			problem = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)


			# Define decay schedule
			schedule = mlrose_hiive.ExpDecay()

			# Define initial state
			init_state = np.random.randint(2, size=(n,))


			# for i in ALGS:

			# 	if i == 0:
			# 		problem_mim1 = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)
			# 		t0 = time.process_time()
			# 		rhc_out = mlrose_hiive.random_hill_climb(problem_mim1, max_attempts=1000, max_iters=1000,
			# 			init_state = init_state, curve=True, random_state = 1)
			# 		t1 = time.process_time()
			# 		display_results(i, rhc_out)
			# 		print("	Process Time: ", np.round(t1 - t0, 4))
			# 		utils.plot_fitness(rhc_out[2], '4peaks_rhc', '4 Peaks: Random Hill Climb')

			# 	elif i == 1:
			# 		problem_mim2 = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)
			# 		t0 = time.process_time()
			# 		sa_out = mlrose_hiive.simulated_annealing(problem_mim2, schedule = schedule,
			#             max_attempts=500, max_iters=1000,
			#             init_state = init_state, curve=True, random_state = 1)
			# 		t1 = time.process_time()
			# 		display_results(i, sa_out)
			# 		print("	Process Time: ", np.round(t1 - t0, 4))
			# 		utils.plot_fitness(sa_out[2], '4peaks_sa', '4 Peaks: Simlated Annealing')

			# 	elif i == 2:
			# 		problem_mim3 = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)
			# 		t0 = time.process_time()
			# 		ga_out = mlrose_hiive.genetic_alg(problem_mim3, pop_size=2000, mutation_prob=0.1, max_attempts=500, 
			# 			max_iters=100, curve=True, random_state=None)
			# 		t1 = time.process_time()
			# 		display_results(i, ga_out)
			# 		print("	Process Time: ", np.round(t1 - t0, 4))
			# 		utils.plot_fitness(ga_out[2], '4peaks_ga', '4 Peaks: Genetic Algorithm')
			# 		# pass

			# 	elif i == 3:
			# 		problem_mim4 = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)
			# 		t0 = time.process_time()
			# 		mimic_out = mlrose_hiive.mimic(problem_mim4, pop_size=2000, keep_pct=0.2, max_attempts=2000, max_iters=50, 
			# 			curve=True, random_state=None)
			# 		# mimic_out = mlrose_hiive.MIMICRunner(
			# 		# 	problem=problem_mim4, 
			# 		# 	population_sizes=[2000], 
			# 		# 	keep_percent_list=[0.2], 
			# 		# 	max_attempts=5000, 
			# 		# 	max_iters=5000, 
			# 		# 	use_fast_mimic=True,
			# 		# 	experiment_name="MIMC_exp",
			# 		# 	seed=None,
			# 		# 	iteration_list=2 ** np.arange(15),
			# 		# 	early_stopping=True)
			# 		# mimic_run_stats, mimic_run_curves = mimic_out.run()
			# 		# print(mimic_run_stats)
			# 		# print(mimic_run_curves)
			# 		# print(mimic_run_curves['Fitness'])
			# 		t1 = time.process_time()
			# 		display_results(i, mimic_out)
			# 		print("	Process Time: ", np.round(t1 - t0, 4))
			# 		utils.plot_fitness(mimic_out[2], '4peaks_mimic', '4 Peaks: MIMIC')
			# 		# utils.plot_fitness_runner(mimic_run_curves['Fitness'], '4peaks_mimic', '4 Peaks: MIMIC')
			# 		# mimic_run_stats.to_excel("4peaks_mimic.xlsx")


			# Take average over 100 runs for Knapsack problem
			trials = 5
			rhc_all = np.zeros(trials)
			sa_all = np.zeros(trials)
			ga_all = np.zeros(trials)
			mimic_all = np.zeros(trials)

			rhc_time = np.zeros(trials)
			sa_time = np.zeros(trials)
			ga_time = np.zeros(trials)
			mimic_time = np.zeros(trials)

			for i in ALGS:

				if i == 0:

					for j in range(trials):
						save_file = False
						problem_mim1 = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)
						t0 = time.process_time()
						rhc_out = mlrose_hiive.random_hill_climb(problem_mim1, max_attempts=1000, max_iters=1000,
							init_state = init_state, curve=True, random_state = 1)
						t1 = time.process_time()
						if j == trials - 1:
							save_file = True
						print(rhc_out[2])
						utils.multi_plot(rhc_out[2], '4peaks_rhc', '4 Peaks: Random Hill Climb', j, trials, save_file)
						rhc_all[j] = rhc_out[1]
						rhc_time[j] = np.round(t1 - t0, 4)

				elif i == 1:

					for j in range(trials):
						save_file = False
						problem_mim2 = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)
						t0 = time.process_time()
						sa_out = mlrose_hiive.simulated_annealing(problem_mim2, schedule = schedule,
				            max_attempts=500, max_iters=1000,
				            init_state = init_state, curve=True, random_state = 1)
						t1 = time.process_time()
						if j == trials - 1:
							save_file = True
						utils.multi_plot(sa_out[2], '4peaks_sa', '4 Peaks: Simulated Annealing', j, trials, save_file)
						sa_all[j] = sa_out[1]
						sa_time[j] = np.round(t1 - t0, 4)

				elif i == 2:

					for j in range(trials):
						save_file = False
						problem_mim3 = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)
						t0 = time.process_time()
						ga_out = mlrose_hiive.genetic_alg(problem_mim3, pop_size=2000, mutation_prob=0.1, max_attempts=500, 
							max_iters=100, curve=True, random_state=None)
						t1 = time.process_time()
						if j == trials - 1:
							save_file = True
						utils.multi_plot(ga_out[2], '4peaks_ga', '4 Peaks: Genetic Algorithm',j, trials, save_file)
						ga_all[j] = ga_out[1]
						ga_time[j] = np.round(t1 - t0, 4)
						# plt.plot(ga_out[2])
					# pass

				elif i == 3:
					for j in range(trials):
						save_file = False
						problem_mim4 = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)
						t0 = time.process_time()
						mimic_out = mlrose_hiive.mimic(problem_mim4, pop_size=2000, keep_pct=0.2, max_attempts=2000, max_iters=50, 
							curve=True, random_state=None)
						t1 = time.process_time()
						if j == trials - 1:
							save_file = True
						utils.multi_plot(mimic_out[2], '4peaks_mimic', '4 Peaks: MIMIC',j, trials, save_file)
						mimic_all[j] = mimic_out[1]
						mimic_time[j] = np.round(t1 - t0, 4)


			rhc_mean = np.mean(rhc_all)
			sa_mean = np.mean(sa_all)
			ga_mean = np.mean(ga_all)
			mimic_mean = np.mean(mimic_all)

			rhc_mean_time = np.mean(rhc_time)
			sa_mean_time = np.mean(sa_time)
			ga_mean_time = np.mean(ga_time)
			mimic_mean_time = np.mean(mimic_time)

			print("Random Hill Climb - Average Fitness over ", trials, " runs:	", rhc_mean)
			print("	Average Process Time: ", np.round(rhc_mean_time, 4))
			print("Simulated Annealing - Average Fitness over ", trials, " runs:	", sa_mean)
			print("	Average Process Time: ", np.round(sa_mean_time, 4))
			print("Genetic Algorithm - Average Fitness over ", trials, " runs:	", ga_mean)
			print("	Average Process Time: ", np.round(ga_mean_time, 4))
			print("MIMIC - Average Fitness over ", trials, " runs:	", mimic_mean)
			print("	Average Process Time: ", np.round(mimic_mean_time, 4))