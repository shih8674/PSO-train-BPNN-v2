from util.LossFunc import LossFunc
from util.ActivationFunc import ActFunc


Debug = {'switch':True
         }

DataPath = {'LoadPath':'./data/space_train/train_space_AQI_2.csv'
		     }

PSO_para = {'update_para':{ 'wmin':0.02 ,
							'wmax':0.8    , 
							'c1min':0.05,
							'c1max':1 ,
							'c2min':0.05,
							'c2max':1  }, 
			'bounded':2, 
			'iteration':20, 
			'particle':-0.05, 
			'LossFunc':LossFunc.MAPE
			}

NN_para = {'hid_kernel':8, 
		   'out_kernel':1, 
		   'ActivationFunc':ActFunc.relu
		   }
