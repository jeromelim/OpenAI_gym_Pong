"""
This hold the logic of genetic algorithms.

Reference: 
https://github.com/erilyth/Flappy-Bird-Genetic-Algorithms


"""

import numpy as np

def crossover(current_generation,model_idx1, model_idx2,crossover_prob):
    """
    Crossover two neural network to produce two new networks by 
    swapping the weights randomly


    Attributes:
        current_generation(list): A list of models at current generation
        model_idx1(int): Index of first model
        model_idx2(int): Index of secound model

    """

    weights1 = current_generation[model_idx1].get_weights()
    weights2 = current_generation[model_idx2].get_weights()
    weightsnew1 = weights1
    weightsnew2 = weights2
    
    for swap_layer in [0,2,4]:
        if np.random.uniform(0,1)>crossover_prob:
            
            weightsnew1[swap_layer] = weights2[swap_layer]
            weightsnew2[swap_layer] = weights1[swap_layer]
    return np.asarray([weightsnew1, weightsnew2])



def mutate(weights,mutation_power):
    """
    """
    for layers in [0,2,4]:
        
        mean,std = np.mean(weights[layers]), np.std(weights[layers]) 
        change = np.random.normal(mean,std,weights[layers].shape) # Gaussian noise 
        change = change * mutation_power
                    
        weights[layers] += change
    return weights