import matplotlib.pyplot as plt
import numpy as np
import time

def Plot_FitY(category, val_output, val_y):
    new_val_y = np.zeros(val_y.shape[0])
    new_val_y[:] = val_y[:]
    plt.figure(figsize=(8.8,4))
    plt.plot(new_val_y)
    plt.plot(val_output)
    
    plt.title('Predict Result by PSO')
    plt.ylabel('value')
    plt.xlabel('number')
    plt.legend(['val_y', 'val_output'], loc='best')
      
    plt.savefig(f'./result/{category}_y/record_{time.strftime("%m-%d_%H-%M-%S")}.png')


def Plot_Loss(fitness):
    plt.figure(figsize=(8.8,4))
    plt.plot(fitness)    
    plt.title('Loss History')
    plt.ylabel('value')
    plt.xlabel('epoch')
      
    plt.savefig(f'./result/loss_fig/record_{time.strftime("%m-%d_%H-%M-%S")}.png')