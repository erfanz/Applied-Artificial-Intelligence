from Classifier import *
import math
import random
import numpy as np

class DecisionTreeClassifier(Classifier): 
#class DecisionTreeClassifier: 


    def learn(self, X, y):
        """
        Constructs a decision tree.

        Args:
           X: A list of feature arrays where each feature array corresponds to feature values
        of one observation in the training set.
           y: A list of ints where 1s correspond to a positive instance of the class and 0s correspond
        to a negative instance of the class at said 0 or 1s index in featuresList.
        """
        DT = TreeNode(X, y, 0)
        DT.makeTree()
        self.DT = DT

    def getLogProbClassAndLogProbNotClass(self, x):
        """Returns log probabilities that a given observation is a positive sample or negative sample"""
        return self.DT.getLogProbClassAndLogProbNotClass(x)

class TreeNode: 

    def __init__(self, X, y, depth):
        self.X = X  # set of featurized observations
        self.y = y  # set of labels associated with the observations 
        self.depth = depth
        self.depthLimit = 3  # limits the depth of your tree for the sake of performance; feel free to adjust
        self.n = len(X)
        self.splitFeature, self.children = None, None  # these attributes should be assigned in splitNode()
        self.entropySplitThreshold = 0.7219 # node splitting threshold for 80%/20% split; feel free to adjust

    def splitNode(self, splitFeature):
        ''' Creates child nodes, splitting the featurized data in the current node on splitFeature. 
        Must set self.splitFeature and self.children to the appropriate values.

        Args: splitFeature, the feature on which this node should split on (this should be the feature you obtain from
            the bestFeature() function)
        Returns: returns True if split is performed, False if not.
        '''
        if len(set(self.y)) < 2: # fewer than 2 labels in this node, so no split is performed (node is a leaf)
            print "fewer than two  labels"
            return False
        
        positiveObservations = []
        positiveLabels = []
        
        negativeObservations = []
        negativeLabels = []
        
        for i, x in enumerate(self.X):
            if x[splitFeature] > 0:
                positiveObservations.append(x)
                positiveLabels.append(self.y[i])
            else:
                negativeObservations.append(x)
                negativeLabels.append(self.y[i])
            
        positiveChild = TreeNode(positiveObservations, positiveLabels, self.depth + 1)
        negativeChild = TreeNode(negativeObservations, negativeLabels, self.depth + 1)
        
        self.children = [negativeChild, positiveChild]
        self.splitFeature = splitFeature
        return True

    def bestFeature(self):
        ''' Identifies and returns the feature that maximizes the information gain.
        You should calculate entropy values for each feature, and then return the feature with highest entropy.
        Consider thresholding on an entropy value -- that is, select a target entropy value, and if no feature 
        has entropy above that value, return None as the bestFeature 

        Returns: the index of the best feature based on entropy
        '''
        
        
        feature_presence = np.zeros(len(self.X[0]))
        feature_absence = np.zeros(len(self.X[0]))
        
        for i, x in enumerate(self.X):
            for j in range(len(x)):
                if (x[j] > 0): 
                    feature_presence[j] += 1
                else:
                    feature_absence[j] += 1
        
        feature_positive_probs = [float(x)/len(self.X) for x in feature_presence]
        feature_negative_probs = [float(x)/len(self.X) for x in feature_absence]
       
        entropies = [0] * len(self.X[0])
        for i in range(len(feature_positive_probs)):
            if (feature_positive_probs[i] == 0 or feature_negative_probs[i] == 0):
                entropies[i] = 0
            else:
                entropies[i] = - (feature_positive_probs[i] * math.log(feature_positive_probs[i], 2))  - (feature_negative_probs[i] * math.log(feature_negative_probs[i], 2))
        
        
            
        bestFeature = entropies.index(max(entropies))
        if entropies[bestFeature] < self.entropySplitThreshold:
            return None
        else:
            return bestFeature
                        
        

    def makeTree(self):
        '''Splits the root node on the best feature (if applicable),
        then recursively calls makeTree() on the children of the root.
        If there is no best feature, you should not perform a split, and this
        node will become a leaf'''
        bestFeature = self.bestFeature()
        if (bestFeature is not None and self.splitNode(bestFeature) == True and self.depth < self.depthLimit):
            self.children[0].makeTree()
            self.children[1].makeTree()
       
            

    def getLogProbClassAndLogProbNotClass(self, x):
        """
        Args:
            x: A numpy array that corresponds to the feature values for a single observation.

        Returns:
            A tuple containing the log probability that the observation is a member of the class
                and the log probability that the observation is NOT a member of the class
        """
        small_threshold = 0.000001
        
        current = self
        while (True):
            if current.splitFeature is None:
                # now we calculate the probability that the new observation is positive or negative
                # To this end, we count the number of positive labels and negative labels in the remaining observations
                probOfClass = float(current.y.count(1)) / len(current.y)
                probOfNotClass = float(current.y.count(0)) / len(current.y)
                if probOfClass == 0:
                    probOfClass = small_threshold
                
                if probOfNotClass == 0:
                    probOfNotClass = probOfClass = small_threshold
                
                logProbNotClass = math.log(probOfNotClass, 2)
                logProbClass = math.log(probOfClass, 2)
                
                    
                return (logProbClass, logProbNotClass)
            else:
                if x[current.splitFeature] > 0:
                    current = current.children[1]
                else:
                    current = current.children[0]