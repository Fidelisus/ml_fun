from constants import *

import random
from math import e
from time import time
from sklearn.model_selection import cross_validate

class Learning():
	def __init__(self, listOfClfs, X, y, cv):
		self.listOfClfs = listOfClfs
		self.X = X
		self.y = y
		self.cv = cv
		self.MSEs = {}
		self.params = {}
		self.times = {}

	def getRandom(self):
		params = {}
		for clf in self.listOfClfs:
			params[clf] = clf.paramRanges
		self.calculateScores(params)
		return self

	def getNeighbor(self, neigborhoodSize):
		neighborhood = self.getNeigborhood(neigborhoodSize)

		l = Learning(self.listOfClfs,self.X, self.y, self.cv)
		l.calculateScores(neighborhood)
		return l

	def calculateScores(self, paramRanges):
		for clf in self.listOfClfs:
			params = {param: random.choice(paramRange) for param, paramRange in paramRanges[clf].items()}
			scores = cross_validate(clf.classifier(**params), self.X, self.y.values.ravel(), cv=self.cv, scoring='neg_mean_squared_error')
			self.MSEs[clf] = -sum(scores['test_score'])/self.cv
			self.params[clf] = params
			self.times[clf] = sum(scores['fit_time'])/self.cv
			#self.times[clf] = sum(scores['score_time'])/cv

	def getNeigborhood(self, neigborhoodSize):
		neighborhood = {}
		for clf, params in self.params.items():
			neighborhood[clf] = {}
			for param, value in params.items():
				paramRange = clf.paramRanges[param]
				i = list(paramRange).index(value)

				start = i-neigborhoodSize if i-neigborhoodSize >= 0 else 0
				end = i+neigborhoodSize if i+neigborhoodSize < len(paramRange) else len(paramRange)-1
				neighborhood[clf][param] = paramRange[start:end+1]
		return neighborhood

	def getBestClf(self):
		return min(self.MSEs, key=self.MSEs.get)

	def getBest(self):
		clf = self.getBestClf()
		return clf.classifier.__name__, self.MSEs[clf], self.params[clf], self.times[clf]

	def __lt__(self, other):
		return min(self.MSEs.values()) < min(other.MSEs.values())

	def __eq__(self, other):
		return min(self.MSEs.values()) == min(other.MSEs.values())

	def __sub__(self, other):
		return min(self.MSEs.values()) - min(other.MSEs.values())


def simulatedAnnealing(listOfClfs, X, y, neigborhoodSize=2, cv=10, T_init=10**7, P:float=0.97, g=lambda T,t: T*0.95, num_lastChanges=10, stopAfter=600, verbose=False):
	"""
		Parameters:
		X: training features of the data
		y: target feature of the data
		neigborhoodSize: number of paramters before and after in list of parameter
			if its set to 1 the neighborhood of 3 in [1,2,3,4,5,6] would be [2,3,4]
			if its set to 2 the neighborhood of 3 in [1,2,3,4,5,6] would be [1,2,3,4,5]
			if its set to 3 the neighborhood of 3 in [1,2,3,4,5,6] would be [1,2,3,4,5,6]
		cv: how many CV iterations
		T_init: initial temperatature for SA
		P: treshold for SA where also bad solutions will be excepted (=1 means greedy)
		g: SA function for changing the temperature after an iteration
		num_lastChanges: number of iterations without changes before the temperature gets reduced
		stopAfter: time in seconds after which no further iterations should be made

		Returns:
		[Learnings], x: List of all solutions as list of learnings, x is the best learning
	"""
	start_time = time()
	learnings = []

	t = 0
	T = T_init
	x = Learning(listOfClfs, X, y, cv).getRandom()
	best = x
	x_new = x
	learnings = [x]
	while (time() < start_time+stopAfter):
		lastChange = 0
		while (lastChange < num_lastChanges) and (time() < start_time+stopAfter):
			if verbose:
				print(lastChange, T, x.MSEs[x.getBestClf()], start_time+stopAfter-time())

			x_new = x.getNeighbor(neigborhoodSize)
			learnings.append(x_new)
			if (x_new < x):
				best = x_new
				x = x_new
				lastChange = 0
			else:
				if (P < e**(-abs(x_new - x) / T)):
					best = x_new
					x = x_new
					lastChange = 0
				else:
					lastChange += 1
			t = t + 1
		T = g(T, t)
	return learnings, best
