# PSO-train-BP-NN v2
========================
##Code Description
========================

This project optimize the code in 'PSO-train-BP-NN'

This project use PSO (Particle Swarm Optimization) to train the single layer NN structure.

If you only want to see the training process and result, 'main.py' can run directly.

The result of traditional NN set up by keras can find in previous project 'PSO-train-BP-NN'

###code structure:

    util -> ParameterSetting.py => Set all parameter, including data path, PSO parameters, and NN parameters.
								   Detail parameters description show in the file. 
								   User can change the setting in this file.
			LoadData.py => Process the data loading from 'ParameterSetting.py'. 					   
			ActivationFunc.py => The activation for NN. User can add new activation function and set in 'ParameterSetting.py'.
			LossFunc.py => The loss function for PSO. User can add new activation function and set in 'ParameterSetting.py'.
			ParticleSwarmOpt.py => Set up the initialization and iterator for PSO.
								   Without changing PSO update rule(update formulation), this file don't need to be revised.
			Set4RunPSO.py => Set fittness function for PSO (ParticleSwarmOpt.py)
							 Set PSO process for training (RunPSO) and testing (TestPSO).
	
	main.py => call function from util and run PSO-NN based on the parameter set by user.

When the 'main.py' run, the console will represent the number of epoch, the error of val data, the error of test data, and cost time per epoch.
If the debug setting in main.py is ture, more detail messages are shown during training process.


##Data Description
========================

The data is air quality index download from Environmental Protection Administration Executive Yuan, R.O.C. (Taiwan). (https://data.epa.gov.tw/dataset/aqx_p_13)

'space' and 'time' is original data.

'space_train' and 'time_train' is the data after preprocessing.

'AQI_label.py', 'only_6AQI.py', 'space_AQI_label.py',and 'origin_6AQI_label.py' is the preprocessing of data.

The process divided the data into three kinds.

1.Predict the AQI at different times in the same place.

2.Predict the AQI at different places in the same time.

3.Predict AQI from 6 kinds of air quality values.
  
  
