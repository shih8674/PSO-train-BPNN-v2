import numpy as np
import time

class PSO:
    '''
    This class set up the initialization and iterator for PSO, so the user only need to set up own fitfunction.
    Input:
        paticle:int -> number of particles (normal set 2*dim to 3*dim)
        dim:int -> dimension
        bounded:float -> set the search bounded for PSO 
        data_input -> data
        data_ouput -> label
        fitfunction:def -> the fitfunction set by user
        debug:bool -> show debug message
    '''
    def __init__(self, particle, dim, bounded, data_input, data_output, fitfunction, debug=False):
        self.particle = particle
        self.dim = dim
        self.bounded = bounded
        self.fitfunction = fitfunction
        self.data_input = data_input
        self.data_output = data_output
        self.debug = debug
        
    def init_Population(self):
        # initial paticle location and speed
        pbest = np.zeros((self.particle, self.dim))   
        gbest = np.zeros(self.dim)                    
        pbest_fit = np.ones(self.particle)*(10**12)   # fitness of pbest (Set a large value to ensure update loss)     
        gbest_fit = 10**12                            # fitness of gbest (Set a large value to ensure update loss)
        V = np.random.uniform(-self.bounded, self.bounded, (self.particle, self.dim))
        X = np.random.uniform(-self.bounded, self.bounded, (self.particle, self.dim))
        
        for i in range(self.particle):
            loss = self.fitfunction(X[i,:], self.data_input, self.data_output)
            if (loss < pbest_fit[i]):
                pbest_fit[i] = self.fitfunction(X[i,:], self.data_input, self.data_output)         
                pbest[i,:] = X[i,:]
                
            if(pbest_fit[i] < gbest_fit):
                gbest_fit = pbest_fit[i]
                gbest = X[i, :]
                if self.debug:
                    print(f'[debug][ParticleSwarmOpt][init_Population] particle [{i:03d}] update gbest, gbest: {gbest_fit:.6f}')

        if self.debug:
            print('[debug][ParticleSwarmOpt][init_Population] initialization ready')
        return X, V, pbest, pbest_fit, gbest_fit, gbest
    
    def iterator(self, w, c1, c2, X, V, pbest, gbest, pbest_fit, gbest_fit, time_record):
        rand1 = np.random.uniform(0, 1, (self.particle, 1))
        rand2 = np.random.uniform(0, 1, (self.particle, 1))
        # update rule in PSO
        V1 = w*V + c1*rand1*(pbest - X) + c2*rand2*(gbest - X)
        X1 = X + V1
        # ensure parameters are within the set range
        out_range_upper = np.where(X1 > self.bounded)
        out_range_lower = np.where(X1 < -self.bounded)
        X1[out_range_upper] = np.random.uniform(-self.bounded, self.bounded, len(out_range_upper[0]))
        X1[out_range_lower] = np.random.uniform(-self.bounded, self.bounded, len(out_range_lower[0]))
    
        for i in range(self.particle):
            loss = self.fitfunction(X1[i, :], self.data_input, self.data_output)
            if((loss < pbest_fit[i])):    # update pbest and pbest_fit 
                pbest[i,:] = X1[i, :]
                pbest_fit[i] = loss
    
            if(pbest_fit[i] < gbest_fit):    # update gbest and gbest_fit
                gbest = pbest[i, :]
                gbest_fit = pbest_fit[i]
                np.save(f'./result/gbest/parameter_record_{time_record}.npy', gbest)
                if self.debug:
                    print(f'[debug][ParticleSwarmOpt][iterator] particle [{i:03d}] update gbest, gbest: {gbest_fit:.6f}')
                
        if self.debug:
            print(f'[debug][ParticleSwarmOpt][iterator] loss (gbest fit) = {gbest_fit:.6f}')
                
        return  X1, V1, gbest, pbest, pbest_fit, gbest_fit