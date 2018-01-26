import gym
import random
from collections import deque
from keras.models import Sequential
import keras.models
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.initializers import normal
from keras import optimizers
from keras import backend as b
from keras import losses
import numpy as np
import copy as cp
import os
import gym_ple
import pygame
from pygame import event
import cv2

#create local directory weights/FlappyBird
#for saving weights - otherwise error

#dependencies required:
#gym, gym-ple, Pygame LE, Keras(Thenao/TensorFlow), numpy, opencv2, python 2.7/3.5, h5py

def _huber_loss(x, y):
    e = x - y
    return b.mean(b.sqrt(1 + b.square(e)) - 1, axis=-1)

class DQN_Agent:

    def __init__(self, env, epos,action_size,state_size,action_idle=0, action_idle_multiply=1,control=False, dim=1, frames_input=1, second_size=None, epos_snap=750, epos_explore_jump=150, explore_jump=0.55, log_path=None, model_path=None, weights_path=None, ddqn=False, past_epos=0, expl_rate=1, expl_decay=0.999, expl_min=0.05, discount=0.95, lr=0.0001, memory=5000, batch=50, beta=None, init_reward=0):

        #if log_path is None then no logs are stored
        self.numb=0
        self.log_path=log_path
        self.log_handler=None

        #if model_path is not None then model is loaded from this path
        self.model_path=model_path
        #if weights path is not None then weights are loaded from this path
        self.weights_path=weights_path

        #if its true then focusing pygame window lets you control game
        self.control = control
        self.env = gym.make(env)


        self.past_epos = past_epos
        self.epos = epos
        self.epos_snap = epos_snap
        self.sum_for_average = 0

        self.exploration_rate = expl_rate
        self.exploration_decay = expl_decay
        self.exploration_min = expl_min
        self.epos_explore_jump = epos_explore_jump
        self.explore_jump = explore_jump
        self.discount_rate = discount
        self.learning_rate = lr
        self.frames_input=frames_input
        self.state_size = state_size
        self.action_size = action_size
        #action idle is action that should be picked the most, changes most slightly or when no idea
        #action idle is inactive when action_idle_multiply=1, multiply shouldnt be less than 1
        self.action_idle=action_idle
        self.action_idle_multiply=action_idle_multiply

        self.dim = dim
        #Second size must be used when dim is >1
        self.second_size = second_size
        self.memory = deque(maxlen=memory)
        self.batch_size = batch

        self.err_max = 1
        #if beta is None then Model dynamics prediction explore is not used
        self.prediction_model = None
        self.beta = beta

        #if ddqn is true then Double q learning is used
        self.ddqn = ddqn
        self.target_model=None
        self.model=None

        self.rew_bonus=0
        self.best=0
        self.total_avg = 0
        self.over_zero = 0
        self.total_rew = 0
        self.init_rew = init_reward

        self.previous4input=None
        self.previous3input=None
        self.previous2input=None
        self.previousinput=None
        self.observation=None
        self.previous_observation=None

        self.var_file()
        if(self.log_path is not None):
            self.log_file()
        self.build_model()
        if(self.beta is not None):
            self.build_prediction_model()



    def convert_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image2 = image[0:400, 44:244]
        image2 = cv2.resize(image2, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        return image2


    def var_file(self):
        self.numb=0
        if not os.path.isfile("var.txt"):
            fo = open("var.txt", "w")
            fo.write("0")
            fo.close()
        else:
            fo = open("var.txt", "r")
            self.numb = int(next(fo))
            fo.close()

        self.numb += 1
        fo = open("var.txt", "w")
        fo.write(str(self.numb))
        fo.close()


    def log_file(self):
        self.log_handler = open(self.log_path, "a")




    def build_model(self):

        if(self.model_path is not None):
            self.model = keras.models.load_model(self.model_path)
        else:
            self.model = Sequential()

            self.model.add(Convolution2D(32,(10,10), strides=(5,5), data_format='channels_first', padding='same', activation='relu',  input_shape=(self.frames_input,self.state_size,self.second_size)))
            self.model.add(Convolution2D(64,(4,4),strides=(2,2), data_format='channels_first', padding='same', activation='relu'))
            self.model.add(Convolution2D(32,(2,2),strides=(1,1), data_format='channels_first', padding='same', activation='relu'))
            self.model.add(Flatten())

            self.model.add(Dense(400, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            self.model.compile(optimizer=optimizers.Adam(self.learning_rate), loss='mse')


        if(self.weights_path is not None):
            self.model.load_weights(self.weights_path)

        if(self.ddqn):
            self.target_model=cp.deepcopy(self.model)
            self.target_model.set_weights(self.model.get_weights())


    def build_prediction_model(self):

        self.prediction_model = Sequential()
        self.prediction_model.add(Dense(24, activation='relu', input_dim=self.state_size + 1))
        self.prediction_model.add(Dense(18, activation='relu'))
        self.prediction_model.add(Dense(self.state_size, activation='linear'))
        self.prediction_model.compile(optimizer=optimizers.Adam(lr=self.learning_rate), loss='mse')



    def build_deq(self, a, b, c, d):
        x = deque(maxlen=self.frames_input)
        x.append(a)
        x.append(b)
        x.append(c)
        x.append(d)
        return x

    def build_array(self, deq):
        a=deq.popleft()
        temp = np.append(a, deq.popleft(),axis=0)

        while len(deq):
            temp=np.append(temp,deq.popleft(), axis=0)
        return temp

    def reshape_input(self, observation):
        if (self.dim == 1):
            observation = np.reshape(observation, [1, self.frames_input, self.state_size])
        if (self.dim == 2):
            observation = np.reshape(observation, [1, self.frames_input, self.state_size, self.second_size])
        return observation

    def sample_one_fail(self,batch, i_episode):

        if np.random.rand() < (i_episode+self.past_epos)*(1.5e-7)*self.batch_size:
            batch.pop()
            i=0
            while self.memory[i][2] != -self.init_rew:
                i+=1
            batch.append(self.memory[i])
        return batch



    def replay(self, i_episode):
        if(self.ddqn):
            self.target_model.set_weights(self.model.get_weights())

        for i in range(self.batch_size):
            if (self.exploration_rate > self.exploration_min):
                self.exploration_rate *= self.exploration_decay

        if len(self.memory) >= self.batch_size * 50:

            batch = random.sample(self.memory, self.batch_size)

            #exchange first sample with one latest fail with probability scalling up with time and batch_size
            batch = self.sample_one_fail(batch, i_episode)


            for state, action, reward, next_state, done, prev_state, prev2_state, prev3_state in batch:

                '''
                ob = []
                for val in st[0]:
                    ob.append(val)

                ob.append(action)
                ob = np.reshape(ob, [1, state_size + 1])

                explr_pred = expl.predict(ob)[0]

                err = np.linalg.norm(next_state - explr_pred)
                if err > err_max:
                    err_max = err

                err_norm = err / err_max
                rew_bonus = err_norm * beta/(1+past_epos/100+i_episode/100)
                #rew += rew_bonus
                expl.fit(ob, next_state, verbose =0, epochs=1)
                '''

                target = reward
                if not done:
                   #same as in decide action build
                    temp = self.reshape_input(self.build_array(self.build_deq(prev2_state,prev_state,state,next_state)))

                    a = self.model.predict(temp)[0]
                    if (self.ddqn):
                        b = self.target_model.predict(temp)[0]
                        target = reward + self.discount_rate * b[np.argmax(a)]
                    else:
                        target = reward+self.discount_rate*np.amax(a)

                temp=self.reshape_input(self.build_array(self.build_deq(prev3_state,prev2_state,prev_state,state)))

                target_f = self.model.predict(temp)

                target_f[0][action] = target

                self.model.fit(temp, target_f, epochs=1, verbose=0)




    def epos_snapshot(self, i_episode):
        if (i_episode % self.epos_explore_jump == 0):
            self.exploration_rate += self.explore_jump
        if (i_episode % self.epos_snap == 0):
            self.model.save_weights("weights/FlappyBird/" + str(self.numb) + "_dqn_ep_" + str(i_episode),
                                    overwrite=True)
            self.model.save("weights/FlappyBird/" + str(self.numb) + "_dqn_model", overwrite=True,
                            include_optimizer=True)
            meta = open("weights/FlappyBird/" + str(self.numb) + "_metadata", "a")

            meta.write("Episode: {}, Best: {}, Average: {}, Overzero: {} \n".format(i_episode, self.best,
                                                                                    self.total_avg / self.epos_snap,
                                                                                    self.over_zero / self.epos_snap))
            meta.close()
            self.best = 0
            self.total_avg = 0
            self.over_zero = 0

    def reshape_observation(self, observation):
        if(self.dim == 1):
            observation = np.reshape(observation, [1, self.state_size])
        if(self.dim == 2):
            observation = np.reshape(observation, [1, self.state_size, self.second_size])
        return observation



    def preprocess_observation(self, observation):
        # use a preprocess functions
        observation=self.convert_image(observation)

        #appropriate reshape
        observation=self.reshape_observation(observation)
        return observation

    def prepare_states(self, observation):
        # prepare state
        self.observation = self.preprocess_observation(observation)

        # prepare previous state
        self.previous_observation = self.observation

    def prepare_inputs(self,inp):
        self.previousinput = inp
        self.previous2input = self.previousinput
        self.previous3input = self.previous2input
        self.previous4input = self.previous3input



    def decide_action(self, inp):

        if(self.control):

            if pygame.key.get_focused():
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                       return 0
                return 1




        # if not control then random
        if (np.random.rand() < self.exploration_rate):
            action = np.random.randint(self.action_size+self.action_idle_multiply-1)
            if action > self.action_size-1:
                return self.action_idle
            else:
                return action

        # if not random then decide
        # prediction is taken from reshaped array builded up from deq of last 3 inputs and current one
        prediction = self.model.predict(self.reshape_input(self.build_array(self.build_deq(self.previous3input, self.previous2input, self.previousinput, inp))))[0]
        return np.argmax(prediction)

    def slide_inputs(self, inp):
        self.previous4input = self.previous3input
        self.previous3input = self.previous2input
        self.previous2input = self.previousinput
        self.previousinput = inp


    def total_rew_changes(self):
        if self.total_rew > self.best:
            self.best = self.total_rew
        self.total_avg += self.total_rew
        if self.total_rew > 0:
            self.over_zero += 1.0

    def run(self):
        for i_episodee in range(self.epos):
                #init loop vars
                i_episode = i_episodee+1
                time = 0
                self.total_rew= self.init_rew

                #get state
                self.observation = self.env.reset()

                self.prepare_states(self.observation)

                #make input out of set of states
                inp = cv2.subtract(self.observation,self.previous_observation)


                self.prepare_inputs(inp)

                #take a cyclic snap if its special episode
                self.epos_snapshot(i_episode)

                #start episode
                while 1:
                    #self.env.render()

                    #use input to pick an action
                    action=self.decide_action(inp)

                    # slide window of past inputs
                    self.slide_inputs(inp)

                    # get next observation and preprocess it (do an action) in order to get new input
                    self.observation, reward, done, info = self.env.step(action)
                    self.observation = self.preprocess_observation(self.observation)


                    # get next input (use observation)
                    inp = cv2.subtract(self.observation, self.previous_observation)

                    #slide previous observation to current
                    self.previous_observation=self.observation

                    # remember state x action -> state
                    if(time>4):
                        self.memory.append((self.previousinput, action, reward, inp, done, self.previous2input,
                                        self.previous3input, self.previous4input))


                    self.total_rew+=reward

                    if(self.log_handler is not None):

                        self.log_handler.write(str(self.previousinput) + "," + str(action) + "," + str(reward) + "," + str(observation) + "," + str(done) + "," + str(self.previous2input) + "," + str(self.previous3input) + "," + str(self.previous4input) + "\n")

                    time+=1

                    if done:
                        self.sum_for_average+=time
                        break



                self.total_rew_changes()

                self.replay(i_episode)

                print("Episode {} / {}. Score: {} Total: {}, explore: {}".format(i_episode, self.epos, time, self.total_rew, self.exploration_rate))


        print("Average: {}".format(self.sum_for_average / self.epos))




#'FlappyBird-v0'
X = DQN_Agent(env='FlappyBird-v0', epos=50000, action_size=2, state_size=100, dim=2,  action_idle=1, action_idle_multiply=5, frames_input=4, second_size=50, lr=4e-5, init_reward=5, ddqn=False, past_epos=0)
X.run()