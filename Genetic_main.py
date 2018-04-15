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
<<<<<<< HEAD
# number_of_inputs = 3 # 2 actions: up, down
action_map= {0:3,1:0,2:2} 
=======
# number_of_inputs = 3 # 3 actions: up, down, stay
action_map= {0:3,1:3} 
>>>>>>> 148edabb7fb00a73182ec694a018b6b77a04e37e
n_observations_per_state = 3

# init variables for genetic algorithms 
num_generations = 1000 # Number of times to evole the population.
population = 30 # Number of networks in each generation.
generation = 0 # Start with first generation
model_to_keep = 6

# init variables for CNN
currentPool = []
input_dim = 80*80
learning_rate = 1e-6

# Initialize all models
for _ in range(population):
    """
    Keras 2.1.1; tensorflow as backend.


    Structure of CNN
    ----------------
    Convolutional Layer: 32 filers of 8 x 8 with stride 4 and applies ReLU activation function
        - output layer (width, height, depth): (20, 20, 32)


    Dense Layer: fully-connected consisted of 32 rectifier units
        - output layer: 32 neurons

    Dropout Layer: 

    Output Layer: fully-connected linear layer with a single output for each valid action, applies softmax activation function
    

    Refernce: https://github.com/mkturkcan/Keras-Pong/blob/master/keras_pong.py

    """
    model = Sequential()
    model.add(Reshape((80,80,1), input_shape=(input_dim,)))
    model.add(Conv2D(32, kernel_size = (3, 3), strides=(4, 4), padding='same', activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(32, kernel_size = (3, 3), strides=(4, 4), padding='same', activation='relu', kernel_initializer='he_uniform'))


    model.add(Flatten())
    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(16, kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='softmax'))
    opt = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    currentPool.append(model)


def preprocess_image(I):
    """ Return array of 80 x 80
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
<<<<<<< HEAD
    output_prob = currentPool[model_num].predict(processed_obs, batch_size=1)[0][0]
    # print(currentPool[model_num].predict(processed_obs, batch_size=1)[0],"+",output_prob)

    # return action_map[output_prob]
    return 2 if output_prob>=0.5 else 3
=======
    output_prob = currentPool[model_num].predict_classes(processed_obs, batch_size=1)[0][0]
    # print(currentPool[model_num].predict(processed_obs, batch_size=1)[0][0])
    

    return 2 if (output_prob >0.5) else 3 (if output_prob <0.5 and output_prob >0) else 1
>>>>>>> 148edabb7fb00a73182ec694a018b6b77a04e37e

def combine_observations_singlechannel(preprocessed_observations, dim_factor=0.7):
    dimmed_observations = [obs * dim_factor**index
                           for index, obs in enumerate(reversed(preprocessed_observations))]
    return np.max(np.array(dimmed_observations), axis=0)


def run_episode(env):
    """ Run single episode of pong (one game)
    Each episode run, each of population model will play 
    the game and get the fitness(final reward when game is finished)

    After each episode of game, we move to new generation of models
    """

    fitness = [-22 for _ in range(population)] # Worst game score in a game is -21
    
    
    print("Start...")
    for model_num in range(population):
        total_reward = 0
        obs = env.reset() #Get the initial pixel output
        preprocessed_observations = deque([], maxlen=n_observations_per_state)

        while True:
   
            preprocessed_observations.append(preprocess_image(obs))
            action = predict_action(combine_observations_singlechannel(preprocessed_observations),model_num) # Predict the next action using CNN
            obs, reward, done, _ = env.step(action)


            total_reward += reward

            if done:
                fitness[model_num] = total_reward

                print("Game Over for model ",model_num," with score ",total_reward)
                break
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
        action = model.predict(combine_observations_singlechannel(preprocessed_observations), batch_size=1)[0][0]
<<<<<<< HEAD
        action = 2 if action >=0.5 else 3 
=======
        action = 2 if action >0.5 else 3 if (action <0.5 and action >0) else 1
>>>>>>> 148edabb7fb00a73182ec694a018b6b77a04e37e
        # action = action_map[action]
        obs, _, done, _ = env.step(action)
        if done:
            break
    
    if save:
        writer.close()

    
def save_pool():
    for model_num in range(population):
        currentPool[model_num].save_weights("Current_Model_Pool/model_new" + str(model_num) + ".keras")
    print("Saved current pool!")


with open("Genetic_generation_score.txt", "w") as text_file:
    text_file.write("{}, {}".format("generation","max_fitness"))
    text_file.write("\n")


def main():
    global currentPool, generation
    
    
    for _ in range(num_generations+1):
        """ Train models num_generations times 
        """
        
        print("Running Generation: ", generation)
        print("="*70)


        save_pool()
        obs = env.reset()
        fitness = run_episode(env)
        max_fitness,min_fitness = np.max(fitness),np.min(fitness)
        print("Best model in this generation: ", np.argmax(fitness))
        print(max_fitness,min_fitness)
        best_model = currentPool[np.argmax(fitness)]

        if generation %10==0:
            print("Saving gameplay")
            run_game(env,best_model,generation,save=True)
            

        print("Start training")
        print("*"*70)


  
        # Reference https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
        sorted_models = [x for _, x in sorted(zip(fitness,currentPool.copy()), key=lambda pair: pair[0],reverse=True)]
        Keep_models = list()

        # Init child_model (Will change the weights later)
        child_models = sorted_models[:2].copy()

        Keep_models.extend(sorted_models[:model_to_keep]) # keep best models
        
        # Randomly keep some models 
        for model in sorted_models[model_to_keep:]:
            if np.random.uniform(0,1) >0.85:
                Keep_models.append(model)

        print("Number of models kept: ", len(Keep_models))


        print("Breeding new children")
        while len(Keep_models) < population:
            
            # Higher the fitness score higher chance it is selected 
            idx1 = np.random.choice(list(range(len(sorted_models[:model_to_keep+3])))) 
            idx2 = idx1

            while idx2 == idx1:
                idx2 = np.random.choice(list(range(len(sorted_models[:model_to_keep+3]))))

            new_weights = crossover(sorted_models,idx1, idx2)


            # Breed new children
            child_models[0].set_weights(new_weights[0])
            child_models[1].set_weights(new_weights[1])

            for child in child_models:
                if len(Keep_models) < population:
                    
                    Keep_models.append(child)

        print("Mutating weights")

        for i in range(len(Keep_models)):
            new_weights = Keep_models[i].get_weights()
            new_weights = mutate(new_weights)
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
    
