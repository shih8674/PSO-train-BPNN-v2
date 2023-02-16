import numpy as np
from util.Path_ck import path_ck
from util.LoadData import LoadData
from util.Set4RunPSO import Set4RunPSO
from util.ParameterSetting import DataPath, PSO_para, NN_para, Debug
from util.PlotFigure import Plot_FitY, Plot_Loss
np.seterr(divide='ignore', invalid='ignore')

# check output path
path_ck(['./result/gbest/', 
		 './result/loss_fig/',
		 './result/test_y/',
		 './result/train_y/'])

# load data
input_size, train_x, train_y, val_x, test_x, val_y, test_y = LoadData(DataPath['LoadPath'])

# set NN structure
NN_structure = [input_size, NN_para['hid_kernel'], NN_para['out_kernel']]

# set and run PSO
PSONN = Set4RunPSO(train_x, train_y, 
                   NN_structure, 
                   PSO_para['iteration'], PSO_para['bounded'], PSO_para['update_para'], PSO_para['particle'], 
                   Debug['switch'])
PSONN.RunPSO()

# calculate reuslt base on trained PSO gbest
train_result = PSONN.TestPSO(train_x)
test_result = PSONN.TestPSO(test_x)


Plot_Loss(PSONN.fitness)
Plot_FitY('train', train_result, train_y)
Plot_FitY('test', test_result, test_y)