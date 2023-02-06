import numpy as np


class LossFunc:
	def MSE(x, y):
		return np.mean(np.square(y-x))

	def RMSE(x, y):	
		return np.sqrt(np.mean(np.square(y-x)))

	def MAE(x, y):
		return np.mean(np.abs(y-x))

	def MAPE(x, y):
		return np.mean(np.abs(y-x)/y)
