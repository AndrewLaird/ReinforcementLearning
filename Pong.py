import gym
import numpy as np
#simple NN layers
from keras.layers import Input,Dense,Flatten
from keras.models import Model
#convolutional NN layers
from keras.layers import MaxPooling2D,Conv2D

def initial_samples(num_steps,env):
    env.reset()
    samples = []
    game = []
    for episode in range(num_steps):
        score = 0 
        #play the game 
        while (True):
            #env.render()
            action = env.action_space.sample()
            screen, reward, done, info = env.step(action)
            samples.append([screen,action])
            if(reward >0):
                score +=1
            #when the game is over, record the score and the moves
            if(done):
                samples.append([game,score])
                print(score)
                env.reset()
                break
    return samples


class PolicyNetwork:

    def create_nn(self):
        #Nerual Network should take in the 
        # pixels on the screen (usually 210*160*3)
        # and turn them into up or down

        #Im going to use a CNN to first pool the layers 
        #and then compress them down to the decision
        #using keras sequential for simplicity

        model = Input((210,160,3))
        #pool the input down to 210,160,1
        model = MaxPooling2D(1)(model)
        #convolute over the image a couple times
        model = Conv2D(16,(4,4))(model)

        model = MaxPooling2D(1)

    def create_nn_simple(self):
        #flatten it down to 210*160*3 by 1 and then feed it
        # through a simple 2 layer network for up or down
        inputs= Input((210,160,3))
        flatten = Flatten()(inputs)
        dense1 = Dense(256)(flatten)
        dense2 = Dense(32)(dense1)
        outputs = Dense(1)(dense2)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam",loss=
        return model

    def __init__(self):
        self.model = self.create_nn_simple()

    def train(self,samples):
        #transform the samples into numpy arrays
        rewards = samples[:,1]
        screens = samples[:,0]
        self.model.train(screens,rewards)
        


if __name__ == "__main__":
    environment = "Pong-v0"
    env = gym.make(environment)
    samples = initial_samples(10,env)
    samples = np.array(samples)

    Network = PolicyNetwork()
    Network.train(samples)
