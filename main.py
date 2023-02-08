import time
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util.LoadData import LoadData
from util.Set4RunPSO import Set4RunPSO
from util.ParameterSetting import DataPath, PSO_para, NN_para
np.seterr(divide='ignore', invalid='ignore')


# load data
input_size, train_x, train_y, val_x, test_x, val_y, test_y = LoadData(DataPath['LoadPath'])

# set NN structure
NN_structure = [input_size, NN_para['hid_kernel'], NN_para['out_kernel']]

# set and run PSO
PSONN = Set4RunPSO(train_x, train_y, NN_structure, PSO_para['iteration'], PSO_para['bounded'], PSO_para['update_para'], PSO_para['particle'], debug=True)
PSONN.RunPSO()

# calculate reuslt base on trained PSO gbest
test_result = PSONN.TestPSO(test_x)
