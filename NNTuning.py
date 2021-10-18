from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import sys, getopt
import mlrose_hiive
import time
import utils
# from metrics import Metrics


# read in data
wine_data = pd.read_csv('winequality-red.csv', sep = ';', header = None)
wine_df = pd.DataFrame(wine_data)
wine_data = np.array(wine_data)
wine_data = np.array(wine_data[1:][:],dtype='float')
y_wine = np.array(wine_data[:,-1], dtype='int')
x_wine = wine_data[:, :-1]

x = preprocessing.normalize(x_wine)
y = y_wine



train_scores_plt = []
test_scores_plt = []

# split into train and validation sets
sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8)
for train_index, test_index in sss.split(x, y):
	x_train = x[train_index]
	y_train = y[train_index]
	x_test = x[test_index]
	y_test = y[test_index]



one_hot = OneHotEncoder()
y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

print("y len: ", len(y))
print(y_train)
print(len(y_train))
print(y_test)
print(len(y_test))


# print(skf)
# model = MLPClassifier(alpha=0.00001, hidden_layer_sizes=(200,200), max_iter=int(itr))
# print(np.unique(y))
# scores  = cross_validate(model, x, y, cv=skf, scoring='accuracy', return_train_score=True)
# print("Best -- Iter Num: ", itr)
# train_scores = scores['train_score']
# test_scores = scores['test_score']
# train_scores_avg = np.mean(train_scores)
# test_scores_avg = np.mean(test_scores)
# train_scores_plt.append(train_scores_avg)
# test_scores_plt.append(test_scores_avg)
# name = "ANN_Wine_" + plts[j]
# title = "ANN (Wine) - Best"
# Metrics().plot_learning_curve_itr(name, title, iters, train_scores_plt, test_scores_plt)

# Nueral Network Hyper-Parameters
hidden_nodes = [30,30]
hidden_nodes = [125,125]
activation = 'relu'
max_iters = 5000
# alpha = 0.000001
alpha = 0.0000001
max_attemtps = 1000

# Initialize neural network object and fit object - Gradient Descent
# print("** Gradient Descent **")
# nn_model1 = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, activation=activation, \
#                                  algorithm = 'gradient_descent', max_iters=max_iters, \
#                                  bias = True, is_classifier = True, learning_rate=alpha, \
#                                  early_stopping = True, max_attempts=max_attemtps, \
#                                  random_state = 3, curve=True)

# print("	Training Model ...")
# t0 = time.process_time()
# nn_model1.fit(x_train, y_train_hot)
# t1 = time.process_time()
# print("	Total Training time:	", np.round(t1 - t0, 4), " seconds")

# y_train_pred = nn_model1.predict(x_train)
# y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
# print("	y_train_accuracy: ", y_train_accuracy)

# y_test_pred = nn_model1.predict(x_test)
# y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
# print("	y_test_accuracy: ", y_test_accuracy)
# l_curve = nn_model1.fitness_curve
# utils.plot_fitness(l_curve, 'NN_GD', 'Gradient Descent')
# # --------------------------------------------------------------------



# # Initialize neural network object and fit object - Random Hill Climbing
# print("** Random Hill Climbing **")
# nn_model2 = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, activation=activation, \
#                                  algorithm = 'random_hill_climb', max_iters=500.0, \
#                                  bias = True, is_classifier = True, learning_rate=alpha, \
#                                  early_stopping = True, max_attempts=max_attemtps, \
#                                  random_state = 3, restarts=20, curve=True)

# print("	Training Model ...")
# t0 = time.process_time()
# nn_model2.fit(x_train, y_train_hot)
# t1 = time.process_time()
# print("	Total Training time:	", np.round(t1 - t0, 4), " seconds")


# y_train_pred = nn_model2.predict(x_train)
# y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
# print("	y_train_accuracy: ", y_train_accuracy)

# y_test_pred = nn_model2.predict(x_test)
# y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
# print("	y_test_accuracy: ", y_test_accuracy)
# l_curve = nn_model2.fitness_curve
# print(l_curve)
# utils.plot_fitness(l_curve, 'NN_SA', 'Random Hill Climb')
# # --------------------------------------------------------------------



# # Initialize neural network object and fit object - Simulated Annealing
# print("** Simulated Annealing **")
# schedule = mlrose_hiive.ExpDecay()
# nn_model3 = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, activation=activation, \
#                                  algorithm = 'simulated_annealing', max_iters=2000, \
#                                  bias = True, is_classifier = True, learning_rate=0.00001, \
#                                  early_stopping = True, max_attempts=max_attemtps, \
#                                  random_state = 3, schedule=schedule, curve=True)

# print("	Training Model ...")
# t0 = time.process_time()
# nn_model3.fit(x_train, y_train_hot)
# t1 = time.process_time()
# print("	Total Training time:	", np.round(t1 - t0, 4), " seconds")


# y_train_pred = nn_model3.predict(x_train)
# y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
# print("	y_train_accuracy: ", y_train_accuracy)

# y_test_pred = nn_model3.predict(x_test)
# y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
# print("	y_test_accuracy: ", y_test_accuracy)
# l_curve = nn_model3.fitness_curve
# print(l_curve)
# utils.plot_fitness(l_curve, 'NN_SA', 'Simulated Annealing')
# # --------------------------------------------------------------------



# Initialize neural network object and fit object - Genetic Algorithms
print("** Genetic Algorithm **")
schedule = mlrose_hiive.ExpDecay()
nn_model4 = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, activation=activation, \
                                 algorithm = 'genetic_alg', max_iters=100, \
                                 bias = True, is_classifier = True, learning_rate=alpha, \
                                 early_stopping = True, max_attempts=max_attemtps, \
                                 random_state = 3, schedule=schedule, pop_size=1000, curve=True)

print("	Training Model ...")
t0 = time.process_time()
nn_model4.fit(x_train, y_train_hot)
t1 = time.process_time()
print("	Total Training time:	", np.round(t1 - t0, 4), " seconds")


y_train_pred = nn_model4.predict(x_train)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
print("	y_train_accuracy: ", y_train_accuracy)

y_test_pred = nn_model4.predict(x_test)
y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
print("	y_test_accuracy: ", y_test_accuracy)
l_curve = nn_model4.fitness_curve
utils.plot_fitness(l_curve, 'NN_GA', 'Genetic Algorithm')
# --------------------------------------------------------------------
