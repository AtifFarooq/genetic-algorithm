"""
This script is used to develop a simple, feed-forward neural
network for the binary classification of ionospheric radar returns
as either 'good' or 'bad'. The dataset can be found at:
https://archive.ics.uci.edu/ml/datasets/ionosphere

The second half of this file shows how optimal hyperparameters
for a feed-forward network can be 'evolved' over time using
an evolutionary approach. A population of n networks is
initialized, and then through successive periods of crossover,
mutation and evolution, the 'best' one is determined. Inspiration
for the code was taken from Will Larson's blog post which can be
found at: https://lethain.com/genetic-algorithms-cool-name-damn-simple/

NOTE: Please ensure that the file 'ionosphere.data' is in the
same directory from where this file is being run.
"""

# import relevant libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
import random
from random import randint

"""
Load data and pre-process the data
"""
# load all the data as a pandas dataframe
data = pd.read_csv('ionosphere.data', header=None)

# shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# determine number of input features
n_features = data.shape[1] - 1

# Assign all data to 2 classes by replacing
# 'g'(good) with 1 and 'b'(bad) with 0
data.at[data[n_features] == 'g', [n_features]] = 1
data.at[data[n_features] == 'b', [n_features]] = 0

# convert dataframe to a matrix
data_matrix = data.as_matrix()

# divide data matrix into input & output matrices
X_array = data_matrix[:, :n_features]
y_array = data_matrix[:, n_features]

# normalize the input (X_array) to the range [-1,1]
scaler = MinMaxScaler((-1,1), copy=True)
X_array = scaler.fit_transform(X_array)


"""
Build a neural network

A simple feed-forward network with one hidden layer.
    input layer: 34 neurons, representing the outputs from the radar signal
    hidden layer: x neurons, using Sigmoid as an activation function (x depends on the initialization)
    output layer: 2 neurons, representing the type of signal (i.e. 'good' or 'bad')

The network will be trained with Stochastic Gradient Descent (SGD) as an
optimiser, that will hold the current state and will update the parameters
based on the computed gradients.

The performance of the network will be evaluated using cross-entropy.
"""

# define the number of inputs, classes, training epochs, and learning rate (hidden_neurons = 5 and num_epochs = 30000 gives good results)
input_neurons = n_features
output_neurons = 2

# The parameters below have been commented out. These were used for assignment1, but are now determined by the genetic algorithm
# hidden_neurons = 10
# learning_rate = 0.01
# num_epochs = 10000


# define a simple feed-forward neural network structure
class FeedForwardNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(FeedForwardNet, self).__init__()
        # define linear hidden layer output
        self.hidden = torch.nn.Linear(n_input, n_hidden, bias=True)
        # define linear output layer output
        self.out = torch.nn.Linear(n_hidden, n_output, bias=True)

    def forward(self, x):
        """
            In the forward function we define the process of performing
            forward pass, that is to accept a Variable of input
            data, x, and return a Variable of output data, y_pred.
        """
        # get hidden layer input
        h_input = self.hidden(x)
        # define activation function for hidden layer
        h_output = F.sigmoid(h_input)
        # get output layer output
        y_pred = self.out(h_output)

        return y_pred


def generate_rand_params():
    ''' generate a list of randomized parameters [learning_rate, num_epochs, hidden_neurons]
        that can later be used to generate a population of randomly initialized networks
    '''
    # initialize a list that contains randomized params
    chromosomes = []

    # return a random learning rate in the range 0.01 .. 1.00
    learning_rate = float(random.randrange(1, 100))/100

    # return a random num of epochs in range 5000..20,000. Always round to nearest 1000
    num_epochs = round(randint(5000, 20000), -3)

    # return a random number of hidden neurons in the range 0..20
    hidden_neurons = randint(1, 20)

    chromosomes.append(learning_rate)
    chromosomes.append(num_epochs)
    chromosomes.append(hidden_neurons)
    return chromosomes

def create_network(hidden_neurons):
    ''' initialize a new network and return it'''
    net = FeedForwardNet(input_neurons, hidden_neurons, output_neurons)
    return net

def train_random_network(hyperparams):
    ''' given a list of hyperparameters, train a network and show the percent corrent
        classification over the k-folds'''

    # extract relevant information from the list of hyperparams that is passed in
    learning_rate = hyperparams[0]
    num_epochs = hyperparams[1]
    hidden_neurons = hyperparams[2]

    # store all percentage correct classification for averaging
    percent_correct = []

    # Initialize stratified k-fold cross validation
    skf = StratifiedKFold(n_splits=2)

    for train_index, test_index in skf.split(X_array, y_array):
        # for each iteration throught the k folds, determine training/testing data
        X_train, X_test = X_array[train_index], X_array[test_index]
        y_train, y_test = y_array[train_index], y_array[test_index]

        # create Tensors to hold inputs and outputs, wrapping them in Pytorch Variables,
        X = Variable(torch.Tensor(X_train).float())
        Y = Variable(torch.Tensor(y_train).long())


        # define a neural network using the customised structure
        #net = FeedForwardNet(input_neurons, hidden_neurons, output_neurons)
        net = create_network(hidden_neurons)

        # define loss function
        loss_func = torch.nn.CrossEntropyLoss()

        # define oan ptimiser
        optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate)

        # train a neural network
        for epoch in range(num_epochs):
            # Perform forward pass: compute predicted y by passing x to the model.
            Y_pred = net(X)

            # Compute loss
            loss = loss_func(Y_pred, Y)

            # print progress
            if epoch % 50 == 0:
                # convert three-column predicted Y values to one column for comparison
                _, predicted = torch.max(Y_pred, 1)

                # calculate and print accuracy
                total = predicted.size(0)
                correct = predicted.data.numpy() == Y.data.numpy()

                print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
                      % (epoch + 1, num_epochs, loss.data[0], 100 * sum(correct)/total))

            # Clear the gradients before running the backward pass.
            net.zero_grad()

            # Perform backward pass
            loss.backward()

            # Calling the step function on an Optimiser makes an update to its
            # parameters
            optimiser.step()


        """
        Evaluating the Results

        To see how well the network performs on different categories, we will
        create a confusion matrix, indicating for every input radar signal
        which class the network guesses.

        """

        confusion = torch.zeros(output_neurons, output_neurons)

        Y_pred = net(X)

        _, predicted = torch.max(Y_pred, 1)

        for i in range(X_train.shape[0]):
            actual_class = Y.data[i]
            predicted_class = predicted.data[i]

            confusion[actual_class][predicted_class] += 1

        print('')
        print('Confusion matrix for training:')
        print(confusion)


        """
        Step 3: Test the neural network

        Pass the testing data during this fold to the network and get its performance
        """

        # create Tensors to hold inputs and outputs, and wrap them in Variables,
        X_test = Variable(torch.Tensor(X_test).float())
        Y_test = Variable(torch.Tensor(y_test).long())

        # test the neural network using testing data
        Y_pred_test = net(X_test)

        # get prediction
        _, predicted_test = torch.max(Y_pred_test, 1)

        # calculate accuracy
        total_test = predicted_test.size(0)
        correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())
        testing_accuracy = 100 * correct_test / total_test
        percent_correct.append(testing_accuracy)
        print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

        """
        Evaluating the Results

        To see how well the network performs on different categories, we will
        create a confusion matrix, indicating for instance of radar data (rows)
        which class 'g' or 'b' the network guesses (column).

        """

        confusion_test = torch.zeros(output_neurons, output_neurons)

        for i in range(X_test.shape[0]):
            actual_class = Y_test.data[i]
            predicted_class = predicted_test.data[i]

            confusion_test[actual_class][predicted_class] += 1

        print('')
        print('Confusion matrix for testing:')
        print(confusion_test)

    fitness = (sum(percent_correct) / len(percent_correct))
    print('The average percentage correct classification is: %.2f %%' % (sum(percent_correct) / len(percent_correct)))
    percent_correct.clear()
    print("The hyperparams were: ", hyperparams)
    return fitness


def create_population(count):
    ''' create a population of networks with random initial parameters
        where count = number of members in the population. The 'population'
        is represented in the form of a list of networks, where each
        network is a list of parameters (chromosomes)
    '''

    population = []
    for i in range(count):
        network = generate_rand_params()
        population.append(network)
    return population


def fitness(population):
    ''' Determine the fitness of each individual in the population. This will be the
        average percent correct classification after k-folds cross validtion. This
        method stores the results in a dictionary
    '''

    score = dict()

    for network in population:
        fitness = train_random_network(network)
        score[str(fitness)] = network
    return score


def generate_fitness_list(score):
    fitness_list = []
    key_list = sorted(score.keys())
    key_list.sort()
    for key in key_list:
        fitness_list.append(key)

    return fitness_list


def generate_parents(fitness_list):
    """ given a list of the fitness values of each individual in the population,
        this function returns a new list of 'parents'. These include the top 20 percent fittest
        individuals, plus some randomly selected lesser performing individuals to
        promote genetic diversity
    """

    parents = []
    # generate the top 20% individuals
    length = len(fitness_list)
    marker = math.floor(0.7 * (length))

    parents = fitness_list[marker:]

    # Add 15% lesser performing individuals
    floor_marker = math.floor(0.4 * length)
    losers = math.floor(0.15 * length)
    parents = parents + fitness_list[floor_marker:floor_marker + losers]

    return parents


def crossover(parents, score):
    """ given a list of parents (fitness values), this function chooses two parents,
        chooses combinations of their parameters at random and
        generates a child
    """
    mother = str(random.choice(parents))
    father = str(random.choice(parents))
    # check against the unlikely case that the input population is very small
    if len(parents) > 3:
        # ensure that both parents are different
        if mother != father:
            # return a child that is a combination of the parents
            mother_params = score[mother]
            father_params = score[father]
            marker = math.floor(0.5 * len(mother_params))
            child = mother_params[:marker] + father_params[marker:]
            return child
        else:
            crossover(parents, score)
    else:
        # return a child that is a combination of the parents
        mother_params = score[mother]
        father_params = score[father]
        marker = math.floor(0.5 * len(mother_params))
        child = mother_params[:marker] + father_params[marker:]
        return child


def breed(parents, score, population):
    """ Given a list of (winning) parent fitness values, a score dictionary and the original
        population, breed as many children from the winners until the population is
        repopulated with new generation. Return this new population.
    """
    new_population = []
    population_num = len(population)

    # extract the network parameters from the parents' fitness value list and append to
    # new_population
    for parent in parents:
        params = score[parent]
        new_population.append(params)

    # generate a new child and add to new_population until its length is 70% of the
    # original (cull the rest for faster convergence)
    while len(new_population) < 0.7 * population_num:
        child = crossover(parents, score)
        if child is not None:
            new_population.append(child)

    return new_population


def mutate(population):
    """ This function takes a population (parameters), chooses a
        'subject' parameter and mutates it
    """
    # generate random parameters
    random_params = generate_rand_params()
    # determine which 'subject' will be mutated
    subject = random.choice(population)
    index = population.index(subject)

    # Choose a parameter in subject to mutate
    limit = len(subject) - 1
    mutation_index = randint(0, limit)
    new_param = random_params[mutation_index]
    # mutate
    subject[mutation_index] = new_param
    population[index] = subject

    return population

# Initialize a new population of n networks
population = create_population(4)


def evolve(population):
    """ method that condenses all functionality and
        carries out one pass of the evolutionary process
    """
    # calculte fitness score of population
    score = fitness(population)
    list = generate_fitness_list(score)
    # generate parents that will breed the next generation
    parents = generate_parents(list)
    new_population = breed(parents, score, population)
    next_generation = mutate(new_population)
    # print("The original population was ", population)
    print("Evolved population ", next_generation)
    return next_generation

# evolve until optimal equilibrium state is reached
while not all(i == population[0] for i in population):
    next_generation = evolve(population)
    population = next_generation
