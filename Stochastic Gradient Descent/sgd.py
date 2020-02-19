# This program performs stochastic gradient descent for the logistic regression classifier. The logistic regression will be covered later, so you don't need to understand it. But if you want to know what the code is really doing, read Mitchell's chapter on naive Bayes and logistic regression.

# Your tasks are defined in the edX homework assignment, but are repoduced in the code here. Search YOUR CODE for the sections you should modify.

# You should only modify the train() and goal_test() functions

import itertools
import math
import operator
import sys
import random

# compute log-likelihood
def likelihood(docs, labels, w):
	likely = 0
	for doc, label in itertools.izip(docs, labels):
		tmp = w[-1]
		for k, v in doc.iteritems():
			tmp += w[k] * v
		prediction = 0
		try:
			big = math.exp(tmp)
			prediction = big / (1 + big)
			if prediction < 0.0005: 
				prediction = 0.0005
			elif prediction > 0.9995:
				prediction = 0.9995
		except: 
			prediction = 0.9995
		likely += label * math.log(prediction) + (1 - label) * math.log(1 - prediction)
	return likely

def postprocess(w, map, imap):
	weights = {}
	for word, val in itertools.izip(imap, w[0:-1]):
		weights[word] = val
	return weights

# test 
def test(docs, labels, w):
	correct = 0.
	for doc, label in itertools.izip(docs, labels):
		tmp = w[-1]
		for k, v in doc.iteritems():
			tmp += w[k] * v
		try:
			big = math.exp(tmp)
			prediction = big / (1 + big)
		except: 
			prediction = 1
		if (label == 1 and prediction > 0.5) or (label == 0 and prediction < 0.5):
			correct += 1			
	return correct / len(docs)

# stochastic gredient descent	
def train(docs, labels, length):
	# w is a list of weights
	w = []
	r = random
	
	# initialize weights with random values
	# add w_n (w_0 in the Mitchell chapter)
	for x in xrange(length+1): 
		w.append(r.uniform(-0.5, 0.5)) 

	# YOUR CODE: change eta value so that the algorithm converges in a reasonable time 
	curr_likelihood = -10000 # log-likelihood
	prev_likelihood = curr_likelihood
	iteration = 0
	changes = [1000.0]*10
	# YOUR CODE: set a threshold when to break a while loop
	while not goal_test(changes, iteration):
		iteration += 1
		#YOUR CODE: Modify eta to encode the step size 10/(iteration_number^2)
		#eta = 5 # step size
		eta = 10.0 / (pow(iteration,2))
		for i in xrange(len(docs)):
			# random sampling
			ind = r.randint(0, len(docs)-1) 
			doc = docs[ind]
			label = labels[ind]
			tmp = w[-1];
			for k, v in doc.iteritems():
				tmp += w[k] * v

			# smoothing to avoid float overflow and underflow
			prediction = 0
			try:
				big = math.exp(tmp)
				prediction = big / (1 + big)
				if prediction < 0.0005:
					prediction = 0.0005
				elif prediction > 0.9995:
					prediction = 0.9995
			except: 
				prediction = 0.9995
			val = label - prediction
			
			# stochastic gradient descent update 
			for k, v in doc.iteritems(): 
				w[k] += eta * val * v
			w[-1] += eta * val
			
		print 'iter: ', iteration
		curr_likelihood = likelihood(docs, labels, w)
		print 'likelihood:', '{0:.2f}'.format(curr_likelihood)
		likelihood_change = math.fabs(curr_likelihood - prev_likelihood)
		#YOUR CODE: Here, replace the oldest chnage value with the most recent (stored in likelihood_change)
		changes.pop()
		changes.insert(0, likelihood_change)
		prev_likelihood = curr_likelihood
	return w

# The condition for stopping gradient descent
def goal_test(changes, iteration):
	#YOUR CODE: Modify this function to return true only when iteration is greater than 40 or
	#the 10 most recent changes is less than 25.
	return sum(changes) < 25 or iteration > 40

# read training and test data and preprocess them
# preprocessing step maps strings to ints 
def readfile(train, test):
	doc_train = []
	ftrain = open(train, 'r')
	map = {} # string to int
	imap = [] # int to string
	label_train = []
	for line in ftrain:
		tokens = line.strip().split(' ')
		tmp = {}
		for i in xrange(1, len(tokens)-1):
			if tokens[i] not in map:
				map[tokens[i]] = len(map)
				imap.append(tokens[i])
			ind = map[tokens[i]]
			if ind not in tmp:
				tmp[ind] = 0
			tmp[ind] += 1
		total = 0
		for v in tmp.values():
			total += v
		for k in tmp:
			tmp[k] = tmp[k] * 1. / total
		doc_train.append(tmp)
		if tokens[len(tokens)-1] == 'SPAM':
			label_train.append(1)
		elif tokens[len(tokens)-1] == 'HAM':
			label_train.append(0)
		else:
			print 'error'

	doc_test = []
	label_test = []
	ftest = open(test, 'r')
	for line in ftest:
		tokens = line.strip().split(' ')
		tmp = {}
		for i in xrange(1, len(tokens)-1):
			ind = map[tokens[i]]
			if ind not in tmp:
				tmp[ind] = 0
			tmp[ind] += 1
		total = 0
		for v in tmp.values():
			total += v
		for k in tmp:
			tmp[k] = tmp[k] * 1. / total			
		doc_test.append(tmp)
		if tokens[len(tokens)-1] == 'SPAM':
			label_test.append(1)
		elif tokens[len(tokens)-1] == 'HAM':
			label_test.append(0)
		else:
			print 'error'

	return doc_train, doc_test, label_train, label_test, map, imap


	



def main():
	if len(sys.argv) != 3:
		print 'usage: python sgd.py train test'
		sys.exit()

	doc_train, doc_test, label_train, label_test, map, imap = readfile(sys.argv[1], sys.argv[2])
	min_accuracy = float('inf')
	max_accuracy = float('-inf')
	for i in range(10):
		w = train(doc_train, label_train, len(map))
		accuracy = test(doc_test, label_test, w[0:-1])
		print 'restart number ', i+1, ' test: ', '{0:.2f}'.format(accuracy * 100), '%'
		max_accuracy = max(max_accuracy, accuracy)
		min_accuracy = min(min_accuracy, accuracy)
	print "max: ", max_accuracy
	print "min: ", min_accuracy

main()


