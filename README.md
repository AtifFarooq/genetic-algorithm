# genetic-algorithm
Hyperparameter Optimization on a Feed-Forward Neural Network using Pruning and Genetic Algorithms

The attached paper and code presents the results of a study in which the ‘Ionosphere Data Set’ from UC Irvine’s Machine
Learning Repository was taken and fed through a multi-layered feed-forward network designed to solve a binary
classification task. A technique that removes neuronal connections on the basis of a metric known as ‘sensitivity’ was
then applied to the data in order to reduce the size of the network. Finally, a genetic algorithm was applied to a
‘population’ of these networks to evolve high-performing hyperparameters. The results showed that both approaches
resulted in networks that resulted in improved classification accuracy. The genetic approach, however, was more
computationally expensive
