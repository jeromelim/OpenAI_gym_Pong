"""
Train a Pong AI using Genetic algorithms.
"""

import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from genetic_algorithms import crossover, mutate
import skvideo.io
import os

# init environment
env = gym.make("Pong-v0")
number_of_inputs = 3 # 3 actions: up, down, stay
action_map = {0:2,1:0,2:3} # predict class 0 will move racket up (action 2), 
                           # class 1 will keep racket stay(action 0) & class 2 will move racket down (action 3)


# init variables for genetic algorithms 
num_generations = 1000 # Number of times to evole the population.
population = 50 # Number of networks in each generation.
# fitness = list() # Fitness scores for evaluate the model
generation = 0 # Start with first generation

# init variables for CNN
currentPool = []
input_dim = 80*80
learning_rate = 0.001

# Initialize all models
for _ in range(population):
    """
    Keras 2.1.1; tensorflow as backend.


    Structure of CNN
    ----------------
    Convolutional Layer: 32 filers of 8 x 8 with stride 4 and applies ReLU activation function
        - output layer (width, height, depth): (20, 20, 32)

    MaxPooling Layer: 2 x 2 filers with stride 2
        - output layer (width, height, depth): (10, 10, 32)
    
    Dense Layer: fully-connected consisted of 32 rectifier units
        - output layer: 32 neurons

    Dropout Layer: 

    Output Layer: fully-connected linear layer with a single output for each valid action, applies softmax activation function
    

    Refernce: https://github.com/mkturkcan/Keras-Pong/blob/master/keras_pong.py

    """
    model = Sequential()
    model.add(Reshape((80,80,1), input_shape=(input_dim,)))
    model.add(Conv2D(32, kernel_size = (8, 8), strides=(4, 4), padding='same', activation='relu', kernel_initializer='he_uniform'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    opt = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    currentPool.append(model)
    # fitness.append(-21) # -21 is the lowest score in game

def preprocess_image(I):
    """ Return array of 80 x 80
    https://github.com/mkturkcan/Keras-Pong/blob/master/keras_pong.py
    """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.reshape([1,len(I.ravel())])

def predict_action(processed_obs,model_num):
    global currentPool
    output_class = currentPool[model_num].predict_classes(processed_obs, batch_size=1)[0]
    return action_map[output_class]

def run_episode(env):
    """ Run single episode of pong (one game)
    Each episode run, each of population model will play 
    the game and get the fitness(final reward when game is finished)

    After each episode of game, we move to new generation of models
    """

    fitness = [-21 for _ in range(population)] # Worst game score in a game is -20
    
    
    print("Start...")
    for model_num in range(population):
        total_reward = 0
        obs = env.reset() #Get the initial pixel output
        prev_obs = None
        while True:
            cur_obs = preprocess_image(obs) # Preprocess the raw pixel to save computation time
            obs_diff = cur_obs - prev_obs if prev_obs is not None else np.zeros(input_dim).reshape([1,input_dim]) # Calculate frame difference as model input 
            prev_obs = cur_obs
            action = predict_action(obs_diff,model_num) # Predict the next action using CNN
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
    prev_obs = None
    if save:
        name = "genetic_gameplay/genetic_pong_generation_" + str(generation) +".mp4"
        writer = skvideo.io.FFmpegWriter(name)

    while True:
        

        if render:
            env.render()

        if save:
            writer.writeFrame(env.render(mode='rgb_array'))
            

    
        cur_obs = preprocess_image(obs) # Preprocess the raw pixel to save computation time
        obs_diff = cur_obs - prev_obs if prev_obs is not None else np.zeros(input_dim).reshape([1,input_dim]) # Calculate frame difference as model input 
        prev_obs = cur_obs
        action = model.predict_classes(obs_diff, batch_size=1)[0]
        action = action_map[action]
        obs, reward, done, _ = env.step(action)
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
        fitness = run_episode(env)
        max_fitness,min_fitness = np.max(fitness),np.min(fitness)
        print("Best model in this generation: ", np.argmax(fitness))
        print(max_fitness,min_fitness)
        best_model = currentPool[np.argmax(fitness)]

        if generation % 100 == 0:
            print("Saving gameplay")
            run_game(env,best_model,generation,save=True)
            


        # Normalize the fitness of each model using minmax normalization 
        for model_num in range(population):
            if (max_fitness-min_fitness) ==0:
                fitness[model_num] = 0.5
            else:
                fitness[model_num] = (fitness[model_num] - min_fitness)/(max_fitness-min_fitness)

        fitness = fitness/ np.sum(fitness)
        
        for model_num in range(int(population/2)):
            parent1 = np.random.uniform(0, 1)
            parent2 = np.random.uniform(0, 1)
            idx1 = -1
            idx2 = -1

            idx1 = np.random.choice(list(range(population)),p=fitness)
            idx2 = np.random.choice(list(range(population)),p=fitness)




            # # Higher the fitness score higher chance it is selected 
            # for idxx in range(population):
            #     if fitness[idxx] >= parent1:
            #         idx1 = idxx
            #         break
            # for idxx in range(population):
            #     if fitness[idxx] >= parent2:
            #         idx2 = idxx
            #         break
            # Crossover weights of two models 
            new_weights1 = crossover(currentPool,idx1, idx2)

            # Mutate the weights randomly
            updated_weights1 = mutate(new_weights1[0])
            updated_weights2 = mutate(new_weights1[1])

            # Update the weights
            currentPool[idx1].set_weights(updated_weights1)
            currentPool[idx2].set_weights(updated_weights2)

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
    
