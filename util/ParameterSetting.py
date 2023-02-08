DataPath = {'LoadPath':'./data/space_train/train_space_AQI_2.csv'
		     }

PSO_para = {'update_para':{ 'wmin':0.02 ,
							'wmax':1    , 
							'c1min':0.05,
							'c1max':1.2 ,
							'c2min':0.05,
							'c2max':1.2  }, 
			'bounded':0.85, 
			'iteration':2, 
			'particle':-0.1
			}

NN_para = {'hid_kernel':8, 
		   'out_kernel':1
		   }
