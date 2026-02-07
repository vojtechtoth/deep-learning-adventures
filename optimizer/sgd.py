"""Implementation of stochastic gradient descend. Pseudocode was taken 
from SSU course at CTU Faculty of Electrical Engineering.

author: Vojtěch Tóth
year: 2026"""

import numpy as np

class SGD():
	def __init__(self, params, lr=0.01):
		self.params = params
		self.lr = lr

	def step(self):
		for param, grad in self.params():
			param -= self.lr * grad
