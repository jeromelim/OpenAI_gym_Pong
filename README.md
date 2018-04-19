# OpenAI_gym_Pong

#### Environment:

In the game of Pong, the policy could take the pixels of the screen and compute the probability of moving the player’s paddle Up, Down, or neither. (https://blog.openai.com/evolution-strategies/)


- Initial input pixel: 210x160x3 [width 210, height 160, and with three color channels R,G,B.]

Steps for processing Image before feeding to CNN (from https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0)

1. Crop the image (we just care about the parts with information we care about).
2. Downsample the image.
3. Convert the image to black and white (color is not particularly important to us).
4. Remove the background.
5. (Maybe not suitiable for CNN)Convert from an 80 x 80 matrix of values to 6400 x 1 matrix (flatten the matrix so it’s easier to use). 
6. Store just the difference between the current frame and the previous frame if we know the previous frame (we only care about what’s changed).


#### Action:
action 0 and 1 are useless, as nothing happens to the racket.

action 2 & 4 makes the racket go up, and action 3 & 5 makes the racket go down.

## DQN

### How to run the programs:

#### Environment Requirement
-   Tensorflow:  1.2.1
-   Python:         2.7

#### Deep Q-Learning (DQN)

Our DQN agent can be ran from the ./dqn directory. You can run it by using the command:

```bash
python main.py --env_name=Pong-v0 --is_train=True --display=True
```


This will run the program on the Pong environment with Training Mode and Rendering
turned on.

### Acknowledgement:

The code for our DQN approach is the existing code from devisers.
The original repository can be found [here](https://github.com/yashbhutwala/pong-ai/tree/master/dqn)


