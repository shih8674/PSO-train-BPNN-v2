from util.ParticleSwarmOpt import PSO
from util.ActivationFunc import ActFunc
from util.LossFunc import LossFunc
import time
import numpy as np


class Set4RunPSO:
	def __init__(self, NN_structure, iteration, bounded, update_para, data_x, data_y, particle=-3, debug=False):
		self.NN_structure = NN_structure
		self.iteration = iteration
		self.dim = NN_structure[0]*NN_structure[1] + NN_structure[1] + NN_structure[1]*NN_structure[2] + NN_structure[2]
		self.bounded = bounded
		self.update_para = update_para
		
		if particle > 0:
			self.particle = particle     # set particle number by user
		elif particle == 0:
			self.particle = self.dim     # particle number is equal to dim
		else:
			self.particle = -(round(particle*self.dim))   # particle number depends on the multiplier of the dimension	

		self.data_x = data_x
		self.data_y = data_y
		self.debug = debug
		if self.debug:
			print(f'[debug][Set4RunPSO][__init__]dimension: {self.dim}, particle: {self.particle}')


	def NN(self, X, train_x):
		''' set single layer BPNN '''
		in_num = self.NN_structure[0]
		hid_num = self.NN_structure[1]
		out_num = self.NN_structure[2]

		fit_w = X[:in_num*hid_num]
		fit_wbias = X[in_num*hid_num: ((in_num*hid_num)+hid_num)]
		fit_v = X[((in_num*hid_num)+hid_num): (((in_num*hid_num)+hid_num)+hid_num)]
		fit_vbias = X[(((in_num*hid_num)+hid_num)+hid_num): ((((in_num*hid_num)+hid_num)+hid_num)+out_num)]

		data_result = np.zeros(train_x.shape[0])

		for input_count in range(train_x.shape[0]):
			input_x = np.zeros(in_num)
			for input_for_hid_num in range(hid_num):
				if(input_for_hid_num == 0):
					input_x = train_x.iloc[input_count].T
				else:
					input_x = np.hstack([input_x, train_x.iloc[input_count].T])

			hid_temp = fit_w * input_x
			hid_result = np.zeros(hid_num)
			for hid_count in range(hid_num):
				hid_result[hid_count] = ActFunc.relu(np.sum(hid_temp[hid_count*in_num : (hid_count*in_num) + in_num]) + fit_wbias[hid_count])

			output_temp = fit_v * hid_result        
			data_result[input_count] = np.sum(output_temp + fit_vbias)

		return data_result

	def fitfunction(self, X, train_x, train_y):
		''' set fitfunction '''
		output_y = self.NN(X, train_x)

		val_y = np.zeros(train_y.shape)
		val_y[:] = train_y[:]

		fittness = LossFunc.MAPE(output_y.flatten(), val_y)

		return fittness


	def RunPSO(self):
		# start PSO
		wmin = self.update_para['wmin']
		wmax = self.update_para['wmax']
		c1min = self.update_para['c1min']
		c1max = self.update_para['c1max']
		c2min = self.update_para['c2min']
		c2max = self.update_para['c2max']

		time_start = time.time()
		PSO_opt = PSO(self.particle, self.dim, self.bounded, self.data_x, self.data_y, self.fitfunction, self.debug)
		if self.debug:
			print('[debug][Set4RunPSO][RunPSO] Finish PSO parameter setting.')
		X, V, pbest, pbest_fit, gbest_fit, self.gbest = PSO_opt.init_Population()
		if self.debug:
			print('[debug][Set4RunPSO][RunPSO] Finish PSO initialization.')
		fitness = np.zeros(self.iteration)
		for tt in range(self.iteration):
			if tt%10 == 0:
				print(f'[Set4RunPSO][RunPSO] iteration: {tt:05d}')
			w = wmin + (self.iteration-tt)/self.iteration*(wmax-wmin)
			c1 = c1min + (self.iteration-tt)/self.iteration*(c1max-c1min)
			c2 = c2max + (self.iteration-tt)/self.iteration*(c2min-c2max)
			X, V, self.gbest, pbest, pbest_fit, gbest_fit = PSO_opt.iterator(w, c1, c2, X, V, pbest, self.gbest, pbest_fit, gbest_fit)
			fitness[tt] = gbest_fit
			if fitness[tt] < fitness[tt -1]:
				print(f'[Set4RunPSO][RunPSO][gbest update] iteration={tt+1}, loss={fitness[tt]}')
				print('--------------------------------------')
			if fitness[tt] < 0.00000001:
				break
		time_end = time.time()
		if self.debug:
			print(f'[debug][Set4RunPSO][RunPSO] cost time: {(time_end - time_start):.4f}')


	def TestPSO(self, data):
		# calculate test ouput base on trained gbest
		if self.debug:
			print('[debug][Set4RunPSO][TestPSO] calculate test result...')
		return self.NN(self.gbest, data)
