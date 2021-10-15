import numpy as np
import pandas as pd
import random as rd
from random import randint


class Knapsack(object):

	def __init__(self, verbose=False):

		self.verbose = verbose
		self.item_number = np.arange(1,11)
		self.weight = np.random.randint(1, 15, size = 10)
		self.value = np.random.randint(10, 750, size = 10)
		self.knapsack_threshold = 35    #Maximum weight that the bag of thief can hold

		if verbose:
			print('The list is as follows:')
			print('Item No.   Weight   Value')
			for i in range(self.item_number.shape[0]):
			    print('{0}          {1}         {2}\n'.format(self.item_number[i], self.weight[i], self.value[i]))


	def fitness(self, state):
		fitness_val = 0.0
		for i in range(len(state)):
			S1 = np.sum(state * self.value)
			S2 = np.sum(state * self.weight)
			if S2 <= self.knapsack_threshold:
				fitness_val = S1
			else :
				fitness_val = 0
			return int(fitness_val)


class Queens(object):

	def __init__(self):
		pass

	def fitness(self, state):
		# Initialize counter
		fitness_cnt = 0
		# For all pairs of queens
		for i in range(len(state) - 1):
				for j in range(i + 1, len(state)):

						# Check for horizontal, diagonal-up and diagonal-down attacks
						if (state[j] != state[i]) and (state[j] != state[i] + (j - i)) and (state[j] != state[i] - (j - i)):
								# If no attacks, then increment counter
								fitness_cnt += 1

		return fitness_cnt

	def display_board(self, best_state):
		flipped = [0, 0, 0, 0, 0, 0, 0, 0]
		for i in range(8):
			flipped[best_state[i]] = i

		print(flipped)

		horz = " --- --- --- --- --- --- --- ---"
		idx = 0
		for i in range(17):
			if i % 2 == 0:
				print(horz)
			else:
				vert = ["|"," "," "," ","|"," "," "," ","|"," "," "," ","|"," "," "," ","|", \
				" "," "," ","|"," "," "," ","|"," "," "," ","|"," "," "," ","|"]
				vert[4*flipped[idx]+2] = "Q"
				idx += 1
				print(''.join(vert))


# class FourPeaks(object):

# 	def __init__(self, n):

# 		self.n = n
# 		self.threshold = n / 5


# 	def fitness(self, state):
		