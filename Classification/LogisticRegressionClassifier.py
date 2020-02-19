from Classifier import *
import numpy as np
import math
from operator import add


class LogisticRegressionClassifier(Classifier):

    def learn(self, X, y):
        """
        Performs an iterative gradient ascent on the weight values.
        
        Args:
           X: A list of feature arrays where each feature array corresponds to feature values
        of one observation in the training set.
           y: A list of ints where 1s correspond to a positive instance of the class and 0s correspond
        to a negative instance of the class at said 0 or 1s index in featuresList.

    Returns: Nothing
        """

        # learning rate
        self.eta = .001
        # convergence threshold
        self.epsilon = 0.01
        
        # Initial Weight
        initialWeights = 0.0001
        
        self.weights = np.empty(len(X[0]))
        self.weights.fill(initialWeights)
        self.w_0 = initialWeights
        
        
        
        oldAverageChange = 0.000001
        while True:
            sumChange = 0
            for i in range(len(self.weights)):
                sumError = 0
                for j, x in enumerate(X):
                    prob = self.getProbability(self.w_0, self.weights, x)
                    error = y[j] - prob
                    sumError += x[i] * error
                
                change = self.eta * sumError
                self.weights[i] = self.weights[i] + change
                sumChange += abs(change)
            
            currentAverageChange = sumChange / len(self.weights)
            if  abs(currentAverageChange - oldAverageChange ) / oldAverageChange < self.epsilon:
                break
            else:
                oldAverageChange = currentAverageChange
            
        

    def getProbability(self, w_0, weights, x):
        """
        This function calculates the probability that an observation is a member of a certain class given its featurized representation
        
        Args:
            w_0:        pre-determined constant weight
            weights:    calculated weights
            x:          featurized observation
            
        Returns:
             A number denoting the probability that logistic regression classifies the input observation to Y = 1
        """
        
        weighted_sum = 0.0
        for i in range(len(weights)):
            weighted_sum += (weights[i] * x[i])  
        
        return float(1) / (1 + math.exp(-(w_0 + weighted_sum)))  
        
        
    def getLogProbClassAndLogProbNotClass(self, x):
        """
        Args:
            features: A numpy array that corresponds to the feature values for a single observation.

        Returns:
            A tuple containing the log probability that the observation is a member of the class
                and the log probability that the observation is NOT a member of the class
        """

        prob = self.getProbability(self.w_0, self.weights, x)
        one_minus_prob = 1 - prob
        small_threshold = 0.000000001
        if prob == 0:
            prob = small_threshold
        if one_minus_prob == 0:
            one_minus_prob = smal_threshold     
        logProbClass = math.log(prob)
        logProbNotClass = math.log(one_minus_prob)

        return (logProbClass, logProbNotClass)
