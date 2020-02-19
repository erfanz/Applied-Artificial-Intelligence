from Classifier import *
import numpy as np
import math
from operator import add

"""
Your NBClassifier dude...
"""
class NaiveBayesClassifier(Classifier):

    def learn(self, X, y):
        """
        You should set up your various counts to be used in classification here: as detailed in the handout.
        Args: 
            X: A list of feature arrays where each feature array corresponds to feature values
                of one observation in the training set.
            y: A list of ints where 1s correspond to a positive instance of the class and 0s correspond
                to a negative instance of the class at said 0 or 1s index in featuresList.

        Returns: Nothing
        """
        # YOU IMPLEMENT -- CORRECTLY GET THE COUNTS
                  
        self.occurencesOfClass = y.count(1)
        self.occurencesOfNotClass = y.count(0)
        
        print "ham/spam", self.occurencesOfClass, self.occurencesOfNotClass
        
        sumOfClass = np.zeros(len(X[0]), np.int32)
        sumOfNotClass = np.zeros(len(X[0]), np.int32)
        for i, x in enumerate(X):
            if y[i] == 0:
                sumOfNotClass = map(add, sumOfNotClass, x)
            else:
                sumOfClass = map(add, sumOfClass, x)
        
        # print "sums", sumOfClass, sumOfNotClass
                            
        self.totalFeatureCountsForClass = sumOfClass
        self.totalFeatureCountsForNotClass = sumOfNotClass
        
        self.totalOccurencesOfFeatureInClass = np.sum(sumOfClass)
        self.totalOccurencesOfFeatureInNotClass = np.sum(sumOfNotClass)
        
        print "sums", self.totalOccurencesOfFeatureInClass, self.totalOccurencesOfFeatureInNotClass
        
        self.totalFeatureObservations = self.totalOccurencesOfFeatureInClass + self.totalOccurencesOfFeatureInNotClass
        self.epsilon = 1.0 / self.totalFeatureObservations 
        
        self.totalObservations = len(X)

    def getLogProbClassAndLogProbNotClass(self, x):
        """
        You should calculate the log probability of the class/ of not the class using the counts determined
        in learn as detailed in the handout. Don't forget to use epsilon to smooth when a feature in the 
        observation only occurs in only the class or only not the class in the training set! 

        Args: 
            x: a numpy array corresponding to a featurization of a single observation 
            
        Returns: A tuple of (the log probability that the features arg corresponds to a positive 
            instance of the class, and the log probability that the features arg does not correspond
            to a positive instance of the class).
        """        
        # YOU IMPLEMENT -- CORRECTLY GET THE COUNTS
        
        # Smoothing
        if (self.occurencesOfClass == 0) or (self.occurencesOfNotClass == 0):
            p_y = float((self.occurencesOfClass + 1)) / (self.totalObservations + 2)
            p_not_y = float((self.occurencesOfNotClass + 1)) / (self.totalObservations + 2)
        else:
            p_y = float(self.occurencesOfClass) / self.totalObservations
            p_not_y = float(self.occurencesOfNotClass)  / self.totalObservations
        
        sumPosteriorForClass = 0
        sumPosteriorForNotClass = 0
        print "size: ", len(x)
        for i, featureCnt in enumerate(x):
            if (featureCnt > 0):
                if self.totalFeatureCountsForClass[i] == 0:
                    # apply the smoothing
                    sumPosteriorForClass += math.log(self.epsilon)
                else:
                    # no smoothing is needed.
                    sumPosteriorForClass += math.log(float(self.totalFeatureCountsForClass[i]) / self.totalOccurencesOfFeatureInClass)
                
                if self.totalFeatureCountsForNotClass[i] == 0:
                    # apply the smoothing
                    sumPosteriorForNotClass += math.log(self.epsilon)
                else:
                    # no smoothing is needed.
                    sumPosteriorForNotClass += math.log(float(self.totalFeatureCountsForNotClass[i]) / self.totalOccurencesOfFeatureInNotClass)
                
                
                
        # print sumPosteriorForClass, sumPosteriorForNotClass 
            
        logProbClass = math.log(p_y) + sumPosteriorForClass
        logProbNotClass = math.log(p_not_y) + sumPosteriorForNotClass
        

        return (logProbClass, logProbNotClass)







