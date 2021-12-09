from constants import *

import random
from math import e
from time import time
from sklearn.model_selection import cross_validate

class Learning():
	def __init__(self):
		self.RMSEs = []
		self.params = []
		self.times = []

	def runRandom(self, data, cv=10):
		train, target = data
		for clf in listOfClfs:
			params = {param: random.choice(paramRange) for param, paramRange in clf.params.items()}
			scores = cross_validate(clf.classifier(**params), train, target, cv=10, scoring='neg_mean_squared_error')
			self.RMSEs.append(sum(scores['test_score'])/cv)
			self.params.append(params)
			self.times.append(sum(scores['fit_time'])/cv)
			#self.times.append(sum(scores['score_time'])/cv)
		return self

	def getBest(self):
		i = self.RMSEs.index(min(self.RMSEs))
		return listOfClfs[i].classifier.__name__, self.RMSEs[i], self.params[i], self.times[i]

	def __lt__(self, other):
		return min(self.RMSEs) < min(other.RMSEs)

	def __eq__(self, other):
		return min(self.RMSEs) == min(other.RMSEs)

	def __sub__(self, other):
		return min(self.RMSEs) - min(other.RMSEs)


def simulatedAnnealing(data, cv=10, T_init=10000000, P:float=0.97, g=lambda T,t: T*0.95, num_lastChanges=10, stopAfter=60):
	"""
		Parameters:
		data: tuple of data and target
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
	x = Learning().runRandom(data, cv)
	learnings = [x]
	while (time() < start_time+stopAfter):
		lastChange = 0
		while (lastChange < num_lastChanges) or (time() < start_time+stopAfter):
			print(lastChange)
			x_new = Learning().runRandom(data, cv)
			learnings.append(x_new)
			if (x_new < x):
				x = x_new
				lastChange = 0
			else:
				if (P < e**(-abs(x_new - x) / T)):
					x = x_new
					lastChange = 0
				else:
					lastChange += 1
			t = t + 1
		T = g(T, t)
	return learnings, x















	def __init__(self, g=lambda T,t: T*0.95, abortTemperature=1000000, P:float=0.97, deltaEval:bool=True):
		super().__init__(neighborhoodStructure.__name__+"_aT"+str(abortTemperature)+"_P"+str(P)+"_"+str(deltaEval))
		self.neighborhoodStructure = neighborhoodStructure
		self.g = g
		self.abortTemperature = abortTemperature
		self.P = P
		self.deltaEval = deltaEval
	
	def run(self, initialSolution:CBTSPSolution=None, instance:CBTSPInstance=None):
		if initialSolution == None:
			initialSolution = DeterministicConstructionHeuristic(self.deltaEval).run(instance=instance)
		
		t = 0
		T = abs(initialSolution.instance.M) * initialSolution.instance.n
		self.solutionUpdate(initialSolution)
		while(T > abs(initialSolution.instance.M // self.abortTemperature)):
			nhs = self.neighborhoodStructure(self.solution, self.deltaEval)
			for _ in range(nhs.structureSize//100):
				xN = randomStep(self.neighborhoodStructure(self.solution, self.deltaEval))
				if (xN < self.solution):
					self.solutionUpdate(xN)
				else:
					if (self.P < e**(-abs(xN.cost - self.solution.cost) / T)):
						self.solutionUpdate(xN)
				t = t + 1
			
			T = self.g(T, t)
		return self.solution





	start_predict = time()
	predictions = clf.predict(pp_data_testing[0])
	time_predict = time()-start_predict