from keras.models import load_model, Sequential   
from keras.layers import Dense 
from keras.optimizers import Adam
import numpy as np
import random

class dqn():
    def __init__(self, input_size, output_size, memory_size, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.epsilon = 1
        self.train_start = 500

        self.model = self.make_model()
        self.target_model = self.make_model()

        self.copy_weights()
        self.memory=[]

    def copy_weights(self):     #update target model
        self.target_model.set_weights(self.model.get_weights())

    def make_model(self):
        model=Sequential()
        model.add(Dense(900, input_shape= (self.input_size,), activation= 'relu')) # input layer
        model.add(Dense(900, activation='relu')) # hidden layer
        model.add(Dense(900, activation='relu')) # hidden layer
        model.add(Dense(self.output_size)) # output layer

        model.compile(loss = "mean_squared_error", optimizer = Adam(lr = self.learning_rate))
        model.summary()

        return model
    
    def get_action(self,state):
        if random.random()< self.epsilon:
            self.q_value = np.zeros(self.input_size)
            return random.choice(range(self.output_size))
        else:
            self.q_value = self.model.predict(state)
            action = self.model.predict(state)
            return np.argmax(action[0])

    def get_Qvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + 0.99*np.amax(next_target)

    '''
    memory is ...
    memory = [(state, action, reward, new_state, done),(state, action, reward, new_state, done), .... ]
    '''

    def save_data(self, state, action, reward, new_state, done):
        if len(self.memory)>self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, new_state, done))

    def train_model(self, batch_size, target = False):
        
        mini_batch = random.sample(self.memory,batch_size)
        X_batch = np.empty((0, self.input_size), dtype=np.float64)
        Y_batch = np.empty((0, self.output_size), dtype=np.float64)

        for state, action, reward, new_state, done in mini_batch:

            self.q_value=self.model.predict(state)

            if target:
                new_target = self.target_model.predict(new_state)
            else:
                new_target = self.model.predict(new_state)

            new_q_value = self.get_Qvalue(reward, new_target, done)

            X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
            Y_sample = self.q_value.copy()

            Y_sample[0][action] = new_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if done:
                X_batch = np.append(X_batch, np.array([new_state.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[reward] * self.output_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=128, epochs=1, verbose=0)  

    def save_model(self):
        self.model.save('model.h5')

    def load_model(self):
        self.model = load_model('model.h5')
        self.target_model = load_model('model.h5')