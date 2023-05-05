import numpy as np
import random
import cv2 as cv
from collections import deque

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.optimizers import adam_v2

class DQNAgent:
    def __init__(self, game_env, state_shape=(84, 84), action_size=3, memory_size=100000, batch_size=64, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.98, learning_rate=0.001):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(*self.state_shape, 1)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon or np.random.rand() < 0.05:
            return random.randrange(self.action_size)

        state = self.preprocess_state(state)
        state = np.expand_dims(state, axis=(0, -1))
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                next_state = self.preprocess_state(next_state)
                next_state = np.expand_dims(next_state, axis=(0, -1))
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            state = self.preprocess_state(state)
            state = np.expand_dims(state, axis=(0, -1))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def preprocess_state(self, state):
        gray_state = cv.cvtColor(state, cv.COLOR_BGR2GRAY)
        resized_state = cv.resize(gray_state, self.state_shape)
        return resized_state

    def save(self, file_name):
        self.model.save(file_name)

    def load(self, file_name):
        self.model = load_model(file_name)