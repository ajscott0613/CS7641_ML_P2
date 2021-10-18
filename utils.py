from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv


def plot_fitness(l_curve, plt_name, title):

	# print(l_curve)
	fitness = np.zeros(np.shape(l_curve)[0])
	itr = np.zeros(np.shape(l_curve)[0])
	for i in range(np.shape(l_curve)[0]):
		fitness[i] = l_curve[i][0]
		itr[i] = l_curve[i][1]
	plt.plot(itr, fitness)
	# plt.legend(["iterations", "validation_data"])
	plt.title(title)
	plt.xlabel("Iterations")
	plt.ylabel("Fitness Score")
	plt.grid()
	plt.savefig(plt_name + '.png',bbox_inches='tight')
	plt.close()

def plot_fitness_vs_fevals(l_curve, plt_name, title):

	# print(l_curve)
	fitness = np.zeros(np.shape(l_curve)[0])
	itr = np.zeros(np.shape(l_curve)[0])
	for i in range(np.shape(l_curve)[0]):
		fitness[i] = l_curve[i][0]
		itr[i] = l_curve[i][1]
	plt.plot(itr, fitness)
	# plt.legend(["iterations", "validation_data"])
	plt.title(title)
	plt.xlabel("Function Evals")
	plt.ylabel("Fitness Score")
	plt.grid()
	plt.savefig(plt_name + '.png',bbox_inches='tight')
	plt.close()


def plot_iters_vs_clock(l_curve, plt_name, title):

	# print(l_curve)
	fitness = np.zeros(np.shape(l_curve)[0])
	itr = np.zeros(np.shape(l_curve)[0])
	for i in range(np.shape(l_curve)[0]):
		fitness[i] = l_curve[i][0]
		itr[i] = l_curve[i][1]
	plt.plot(itr, fitness)
	# plt.legend(["iterations", "validation_data"])
	plt.title(title)
	plt.xlabel("Iterations")
	plt.ylabel("Clock Time")
	plt.grid()
	plt.savefig(plt_name + '.png',bbox_inches='tight')
	plt.close()

def plot_fitness_runner(l_curve, plt_name, title):

	plt.plot(l_curve)
	# plt.legend(["iterations", "validation_data"])
	plt.title(title)
	plt.xlabel("Iterations")
	plt.ylabel("Fitness Score")
	plt.grid()
	plt.savefig(plt_name + '.png',bbox_inches='tight')
	plt.close()

def multi_plot(l_curve, plt_name, title, itr, trials, save_file):

	fitness = np.zeros(np.shape(l_curve)[0])
	itr = np.zeros(np.shape(l_curve)[0])
	for i in range(np.shape(l_curve)[0]):
		fitness[i] = l_curve[i][0]
		itr[i] = l_curve[i][1]
	plt.plot(itr, fitness)
	print(trials)
	# plt.legend(["iterations", "validation_data"])
	if save_file:
		leg = []
		for i in range(trials):
			leg.append("Trial " + str(i+1))
		plt.legend(leg)
		plt.title(title)
		plt.xlabel("Iterations")
		plt.ylabel("Fitness Score")
		plt.grid()
		plt.savefig(plt_name + '.png',bbox_inches='tight')
		plt.close()


def multi_plot_fevals(l_curve, plt_name, title, itr, trials, save_file):

	fitness = np.zeros(np.shape(l_curve)[0])
	itr = np.zeros(np.shape(l_curve)[0])
	for i in range(np.shape(l_curve)[0]):
		fitness[i] = l_curve[i][0]
		itr[i] = l_curve[i][1]
	plt.plot(itr, fitness)
	print(trials)
	# plt.legend(["iterations", "validation_data"])
	if save_file:
		leg = []
		for i in range(trials):
			leg.append("Trial " + str(i+1))
		plt.legend(leg)
		plt.title(title)
		plt.xlabel("Function Evals")
		plt.ylabel("Fitness Score")
		plt.grid()
		plt.savefig(plt_name + '.png',bbox_inches='tight')
		plt.close()