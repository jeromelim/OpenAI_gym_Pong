"""
Train a Cartpole AI using Genetic algorithms.
"""

import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from genetic_algorithms_CartPole  import crossover, mutate
import skvideo.io
import os
from keras.models import load_model

from collections import deque

# init environment
env = gym.make("CartPole-v0")
env._max_episode_steps = 600



# init variables for genetic algorithms 
num_generations = 200 # Number of times to evole the population.
population = 50 # Number of networks in each generation.
generation = 0 # Start with first generation
model_to_keep = int(population * 0.2) # Keep top 20% of models
crossover_prob = 0.4
mutation_power = 0.001

# init variables for CNN
currentPool = []
learning_rate = 1e-6

# Initialize all models

def init_model(poolOfModel,population):
    for _ in range(population):
        """
        Keras 2.1.1; tensorflow as backend.

        """
        model = Sequential()

    
        model.add(Dense(24,input_dim=4,activation="relu"))
        
        model.add(Dense(24,activation='relu'))

        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(lr=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        poolOfModel.append(model)
    return poolOfModel



def predict_action(state,model_num):
    global currentPool
    output_prob = currentPool[model_num].predict(state, batch_size=1)[0][0]
    

    return 0 if output_prob<0.5 else 1

def run_episode(env):
    """ Run episode of cartpole (one game)
    Each episode run, each of networks in population will play 
    the game and get the fitness(final reward when game is finished)

    """

    fitness = [0 for _ in range(population)] 
    
    
    
    print("Start...")
    for model_num in range(population):
        total_reward = 0
        # play three times to get average fitness
        for step in range(3):
        
            obs = env.reset() #Get the initial pixel output
            obs = np.reshape(obs, [1, 4])
            env._max_episode_steps = 600

            
            for t in range(600):


                action = predict_action(obs,model_num)
                obs, reward, terminated, truncated, info = env.step(action)
                

                obs = np.reshape(obs, [1, 4])


                total_reward += reward
                
            

                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
        fitness[model_num] = total_reward/3
        print("Game Over for model ",model_num," with avg. score ",total_reward/3)
    return fitness


def run_game(env,model,generation,render=False,save=False):
    """ Play one cartpole game given a trained model

    Attributes:
    ----------
    render: if True, render the gameplay
    save: if True save the gameplay in mp4 format
    """
    obs = env.reset()
    obs = np.reshape(obs, [1, 4])
    env._max_episode_steps = 600


    if save:
        name = "genetic_gameplay/genetic_cartpole_generation_" + str(generation) +".mp4"
        writer = skvideo.io.FFmpegWriter(name)

    for _ in range(600):
        

        if render:
            env.render()

        if save:
            writer.writeFrame(env.render(mode='rgb_array'))
            

        output_prob = model.predict(obs, batch_size=1)[0][0]
        action = 0 if output_prob<0.5 else 1
        obs, _, done, _ = env.step(action)
        obs = np.reshape(obs, [1, 4])
        if done:
            break
    
    if save:
        writer.close()

    
def save_pool(best_model,score):
    best_model.save("Current_Model_Pool/model_best_in_generation_" + str(generation) +" score_"+str(score) +".h5")
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
        
        if generation %1==0:
            print("Saving gameplay")
            run_game(env,best_model,generation,save=True)

        

        if max_fitness >=600:
            save_pool(best_model,max_fitness)
            break
            

        print("Start training")
        print("*"*70)
        sorted_fitness = sorted(fitness,reverse=True)
        # Reference https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
        sorted_models = [x for _, x in sorted(zip(fitness,currentPool.copy()), key=lambda pair: pair[0],reverse=True)]
        Keep_models = list()


        Keep_models.extend([x for fitness,x in zip(sorted_fitness,sorted_models) if fitness != 0])
        Keep_models = Keep_models[:model_to_keep] #keep only top 20% (if any)
        
        # Randomly keep some models 
        Keep_models = init_model(Keep_models,5)

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

            new_weights = crossover(Keep_models,idx1, idx2,crossover_prob)


            # Breed new children
            child_models[0].set_weights(new_weights[0])
            child_models[1].set_weights(new_weights[1])

            for child in child_models:
                if len(Keep_models) < population:
                    
                    Keep_models.append(child)



        

        print("Mutating weights")

        for i in range(5,len(Keep_models)):
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
    
    