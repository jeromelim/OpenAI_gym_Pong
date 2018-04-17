"""
This hold the logic of genetic algorithms.

Reference: 
https://github.com/erilyth/Flappy-Bird-Genetic-Algorithms


"""

import numpy as np

def crossover(current_generation,model_idx1, model_idx2):
    """
    Crossover two neural network to produce two new networks by 
    swapping the weights randomly (layer 0, 2, and 4 ,10, 12; Conv2d, dense ,output)


    Attributes:
        current_generation(list): A list of models at current generation
        model_idx1(int): Index of first model
        model_idx2(int): Index of secound model

    """

    weights1 = current_generation[model_idx1].get_weights()
    weights2 = current_generation[model_idx2].get_weights()
    weightsnew1 = weights1
    weightsnew2 = weights2
    
    for swap_layer in [4,6,8,10]:
        if np.random.uniform(0,1)>0.5:
            
            weightsnew1[swap_layer] = weights2[swap_layer]
            weightsnew2[swap_layer] = weights1[swap_layer]
    return np.asarray([weightsnew1, weightsnew2])



def mutate(weights):
    """Select weights randomly with a 0.15 probability and then change its value with a random number
    """
    for layers in [4,6,8,10]:
        if np.random.uniform(0,1) >=0.98:
            mean,std = np.mean(weights[layers]), np.std(weights[layers]) 
            change = np.random.normal(mean,std,weights[layers].shape) # Twist the weights around it's mean  
            # for idx in range(len(change)):
            #     # if np.random.uniform(0,1) <= 0.95:
            #     change[idx] = 0
            # change = change.reshape(weights[layers].shape)
                    
            weights[layers] += change
    return weights