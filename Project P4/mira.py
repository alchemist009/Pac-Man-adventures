# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        score = 0
        w = None
        weight = self.weights
        Cweights = util.Counter()
        for C in Cgrid:
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, "..."
                for i in range(len(trainingData)):
                   # labels = trainingLabels[i]
                    vectors = util.Counter()
                    for l in self.legalLabels:
                        vectors[l] = weight[l] * trainingData[i]
                    maxLabel = vectors.argMax()
                
                    if maxLabel == trainingLabels[i]:
                        continue
                    if maxLabel != trainingLabels[i]:
                        tau = (weight[maxLabel] - weight[trainingLabels[i]]) * trainingData[i]
                        tau = (tau + 1.0)/(trainingData[i]*trainingData[i]*2)
                        tau = min(C,tau)
                        #self.weights[labels] += trainingData[i]
                        #self.weights[l] -= trainingData[i]
                
                    for feature in self.features:
                        weight[trainingLabels[i]][feature] = weight[trainingLabels[i]][feature] + tau * trainingData[i][feature]
                        weight[maxLabel][feature] = weight[maxLabel][feature] - tau * trainingData[i][feature]
            
            valid = self.classify(validationData)
            Cweights[C] = self.weights
            num = 0
            count = util.Counter()
            for v in range(len(validationData)):
                for label in validationLabels:
                    vectors[label] = validationData[v] * (self.weights[l])
                if(validationLabels[v] == maxLabel):
                    num += 1
                    
            count[C] = num
        self.weights = Cweights[count.argMax()]
            
     #   maxcount = count.argMax()
    #    self.weights = Cweights[maxcount]
    

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


