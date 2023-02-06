import time
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util.LoadData import LoadData
from util.Set4RunPSO import Set4RunPSO
np.seterr(divide='ignore', invalid='ignore')


# load data
input_size, train_x, val_test_x, train_y, val_test_y, val_x, test_x, val_y, test_y = LoadData('./data/space_train/train_space_AQI_2.csv')

hid_kernel = 6
out_kernel = 1
NN_structure = [input_size, hid_kernel, out_kernel]
update_para = { 'wmin':0.02 ,
				'wmax':1    , 
				'c1min':0.05,
				'c1max':1.2 ,
				'c2min':0.05,
				'c2max':1.2  }
bounded = 0.85
iteration = 30

PSONN = Set4RunPSO(NN_structure, iteration, bounded, update_para, train_x, train_y, particle=-0.1, debug=True)
gbest = PSONN.RunPSO()
