import numpy as np
import pandas as pd
import random as rd
from random import randint


class Knapsack(object):

	def __init__(self, verbose=False, n=5, weight=[10, 5, 2, 8, 15], value=[1, 2, 3, 4, 5], max_w_p=0.6):

		self.verbose = verbose
		self.item_number = np.arange(1,n+1)
		# self.weight = np.random.randint(1, 15, size = n)
		# self.value = np.random.randint(10, 750, size = n)
		self.weight = np.array(weight)
		self.value = np.array(value)
		# self.knapsack_threshold = int(0.15*np.sum(self.weight))
		if max_w_p > 1.0:
			self.knapsack_threshold = int(max_w_p)
		else:
			self.knapsack_threshold = int(max_w_p*np.sum(self.weight))



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

	def __init__(self, n_queens=8):

		self.n_queens = n_queens

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
		vert_init = []
		horz = []
		board_len = self.n_queens*4 + 1
		for i in range(board_len):
			if i % 4 == 0:
				vert_init.append("|")
				horz.append(" ")
			else:
				vert_init.append(" ")
				horz.append("-")

		flipped = [0]*self.n_queens
		for i in range(len(flipped)):
			flipped[best_state[i]] = int(i)

		print(flipped)

		horz = ''.join(horz)
		idx = 0
		for i in range(2*self.n_queens + 1):
			if i % 2 == 0:
				print(horz)
			else:
				vert = vert_init.copy()
				vert[4*flipped[idx]+2] = "Q"
				idx += 1
				print(''.join(vert))


# class FourPeaks(object):

# 	def __init__(self, n):

# 		self.n = n
# 		self.threshold = n / 5


# 	def fitness(self, state):
