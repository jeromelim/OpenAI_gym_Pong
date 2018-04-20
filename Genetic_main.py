"""
Train a Pong AI using Genetic algorithms.
"""

import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from genetic_algorithms import crossover, mutate
import skvideo.io
import os

from collections import deque

# init environment
env = gym.make("Pong-v0")

n_observations_per_state = 3

# init variables for genetic algorithms 
num_generations = 1000 # Number of times to evole the population.
population = 30 # Number of networks in each generation.
generation = 0 # Start with first generation
model_to_keep = int(population * 0.2) # Keep top 20% of models
crossover_prob = 0.5
mutation_power = 0.005

# init variables for CNN
currentPool = []
input_dim = 80*80
learning_rate = 1e-6

# Initialize all models

def init_model(poolOfModel,population):
    for _ in range(population):
        """
        Keras 2.1.1; tensorflow as backend.

        Architecture borrowed from: Mnih et al. (2015)



        """
        model = Sequential()
        model.add(Reshape((80,80,1), input_shape=(input_dim,)))
        model.add(Conv2D(32, kernel_size = (8, 8), strides=(4, 4), padding='same', activation='relu', kernel_initializer='he_uniform'))
        model.add(Conv2D(64, kernel_size = (4, 4), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_uniform'))
        model.add(Conv2D(64, kernel_size = (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_uniform'))


        model.add(Flatten())
        model.add(Dense(512, kernel_initializer='he_uniform'))
        
        model.add(Activation('relu'))

        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(lr=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        poolOfModel.append(model)
    return poolOfModel


def preprocess_image(I):
    """ Return array of 80 x 80
    Reference:
    https://github.com/mkturkcan/Keras-Pong/blob/master/keras_pong.py
    """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.reshape([1,len(I.ravel())]).astype(float)

def predict_action(processed_obs,model_num):
    global currentPool
    output_prob = currentPool[model_num].predict(processed_obs, batch_size=1)[0][0]

    return np.random.choice([2,3],1,p=[output_prob,1-output_prob])

def combine_observations_singlechannel(preprocessed_observations, dim_factor=0.5):
    """
    From the book Hands-on Machine Learning with Scikit-Learn and TensorFlow
    """

    dimmed_observations = [obs * dim_factor**index
                           for index, obs in enumerate(reversed(preprocessed_observations))]
    return np.max(np.array(dimmed_observations), axis=0)


def run_episode(env):
    """ Run episode of pong (one game)
    Each episode run, each of networks in population will play 
    the game and get the fitness(final reward when game is finished)


    """

    fitness = [-22 for _ in range(population)] # Worst game score in a game is -21
    
    
    print("Start...")
    for model_num in range(population):
        total_reward = 0
        # Run three games to get average fitness
        for _ in range(3):
            obs = env.reset() #Get the initial pixel output
            preprocessed_observations = deque([], maxlen=n_observations_per_state)

            while True:
    
                preprocessed_observations.append(preprocess_image(obs))
                action = predict_action(combine_observations_singlechannel(preprocessed_observations),model_num)
                obs, reward, done, _ = env.step(action)


                total_reward += reward

                if done:
                    break
        fitness[model_num] = total_reward/3 
        print("Game Over for model ",model_num," with avg. score ",total_reward/3)
    return fitness


def run_game(env,model,generation,render=False,save=False):
    """ Play one pong game given a trained model

    Attributes:
    ----------
    render: if True, render the gameplay
    save: if True save the gameplay in mp4 format
    """
    obs = env.reset()

    preprocessed_observations = deque([], maxlen=n_observations_per_state)
    if save:
        name = "genetic_gameplay/genetic_pong_generation_" + str(generation) +".mp4"
        writer = skvideo.io.FFmpegWriter(name)

    while True:
        

        if render:
            env.render()

        if save:
            writer.writeFrame(env.render(mode='rgb_array'))
            

        preprocessed_observations.append(preprocess_image(obs))
        output_prob = model.predict(combine_observations_singlechannel(preprocessed_observations), batch_size=1)[0][0]
        action = np.random.choice([2,3],1,p=[output_prob,1-output_prob])
        obs, _, done, _ = env.step(action)
        if done:
            break
    
    if save:
        writer.close()

    
def save_pool(best_model,score):
    best_model.save("Current_Model_Pool/model_best_in_generation" + str(generation) +" score_"+str(score) +".h5")
    print("Saved Best model!")


with open("Genetic_generation_score.txt", "w") as text_file:
    text_file.write("{}, {}".format("generation","max_fitness"))
    text_file.write("\n")


def main():
    global currentPool, generation

    print("Init First random population")
    currentPool = init_model(currentPool,population)
    print("Population size: ",len(currentPool))
    
    for _ in range(num_generations+1):
        """ Train models num_generations times 
        """
        
        print("Running Generation: ", generation)
        print("="*70)



        fitness = run_episode(env)
        max_fitness,min_fitness = np.max(fitness),np.min(fitness)
        print("Best model in this generation: ", np.argmax(fitness))
        print(max_fitness,min_fitness)
        best_model = currentPool[np.argmax(fitness)]
        
        if generation %10==0:
            print("Saving gameplay")
            run_game(env,best_model,generation,save=True)

        if max_fitness > -13:
            save_pool(best_model,max_fitness)

        print("Start training")
        print("*"*70)
        sorted_fitness = sorted(fitness,reverse=True)
        # Reference https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
        sorted_models = [x for _, x in sorted(zip(fitness,currentPool.copy()), key=lambda pair: pair[0],reverse=True)]
        Keep_models = list()

        if max_fitness == -21:
            # All models perform worst 
            print("Kill all models and start again")
            Keep_models = init_model(Keep_models,population)


        elif sorted_fitness[1] == -21:
            print("only one model score")
            print("Keep the good one and generate others randomly")
            Keep_models.append(sorted_models[0])
            Keep_models = init_model(Keep_models,population-1)
        else:
            print("more than one models score")
            print("Keep the good ones and generate others")
            Keep_models.extend([x for fitness,x in zip(sorted_fitness,sorted_models) if fitness != -21])
            Keep_models = Keep_models[:model_to_keep] #keep only top 20% (if any)
            


            print("Number of models kept: ", len(Keep_models))
            # Init child_model (Will change the weights later)
            child_models = Keep_models[:2].copy()
            print("Breeding new children")
            while len(Keep_models) < population:
                
                # Higher the fitness score higher chance it is selected 
                idx1 = np.random.choice(list(range(len(Keep_models[:model_to_keep])))) 
                idx2 = idx1

                while idx2 == idx1:
                    idx2 = np.random.choice(list(range(len(Keep_models[:model_to_keep]))))

                new_weights = crossover(Keep_models,idx1, idx2,crossover_prob=crossover_prob)


                # Breed new children
                child_models[0].set_weights(new_weights[0])
                child_models[1].set_weights(new_weights[1])

                for child in child_models:
                    if len(Keep_models) < population:
                        
                        Keep_models.append(child)



        

        print("Mutating weights")

        for i in range(len(Keep_models)):
            new_weights = Keep_models[i].get_weights()
            new_weights = mutate(new_weights,mutation_power)
            Keep_models[i].set_weights(new_weights)


        currentPool = Keep_models

        print("Finished training")


  



        with open("Genetic_generation_score.txt", "a") as text_file:
            text_file.write("{}, {}".format(generation,max_fitness))
            text_file.write("\n")
            
        generation += 1
        print("Finish current generation")
        print("Current best game score: ",max_fitness)
        print("_"*70)

        

    

    return

    


if __name__ == '__main__':
    main()
    
