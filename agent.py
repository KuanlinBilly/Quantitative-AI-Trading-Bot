from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import random
import numpy as np

# Define the Deep Q-Learning agent
class DQLAgent:
    def __init__(self, gamma=0.95, hidden_units=24, opt=Adam,
                 lr=0.001, finish=False, env=None):
        # Initialize the RL agent with default parameters
        self.env = env
        self.finish = finish
        self.epsilon = 1.0 # Initial exploration rate
        self.epsilon_min = 0.01 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Rate at which exploration rate decays
        self.gamma = gamma # Discount factor
        self.batch_size = 32 # Size of experience replay batch
        self.max_treward = 0 # Maximum total reward achieved by the agent
        self.averages = [] # List of average rewards per episode
        self.memory = deque(maxlen=200) # Replay memory for experience replay
        self.observation_space_size = env.observation_space.shape[0] # Size of observation space
        self.model = self._build_model(hidden_units, opt, lr) # Build the neural network model

    def _build_model(self, hidden_units, opt, lr):
        # Build the neural network with 2 hidden layers and a linear output layer.
        # The input layer size is determined by the size of the observation space.
        # The output layer size is determined by the number of actions in the action space.
        model = Sequential()
        model.add(Dense(hidden_units, input_dim=self.observation_space_size, activation='relu'))
        model.add(Dense(hidden_units, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=opt(learning_rate=lr))
        return model

    def save(self):
        # Save the model to disk
        self.model.save('saved_model/dql_model')

    def load(self):
        # Load the model from disk
        self.model = tf.keras.models.load_model(
            'saved_model/dql_model')

    def act(self, state):
        # Select an action based on an epsilon-greedy policy.
        # With probability epsilon, the agent selects a random action.
        # Otherwise, the agent selects the action with the highest Q-value for the current state.
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        action = self.model.predict(state)[0]
        return np.argmax(action)

    def replay(self):
        # Sample a batch of experiences from memory and update the Q-values for each state-action pair.
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                reward += self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            target = self.model.predict(state)
            target[0, action] = reward
            self.model.fit(state, target, epochs=1,
                           verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # Decrease exploration rate by a factor of epsilon_decay

    def learn(self, episodes):
        # Train the agent for a fixed number of episodes.
        # At each time step, the agent selects an action, observes a reward, and updates its Q-values.
        trewards = []
        for e in range(1, episodes + 1):
            state = self.env.reset()
            state = np.reshape(state, [1, self.observation_space_size])
            for i in range(len(self.env.data)):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.observation_space_size])
                self.memory.append([state, action, reward,
                                    next_state, done])
                state = next_state
                if done:
                    treward = i + 1
                    trewards.append(treward)
                    av = sum(trewards[-25:]) / 25
                    self.averages.append(av)
                    self.max_treward = max(self.max_treward, treward)
                    templ = 'episode: {:4d}/{} | treward: {:4d} | ' + \
                        'av: {:6.1f} | max: {:4d}'
                    print(
                        templ.format(e, episodes, treward, av,
                                     self.max_treward), end='\r'
                    )
                    break
            if av > 195 and self.finish:
                break
            if len(self.memory) > self.batch_size:
                self.replay()
        self.save()

    def test(self, episodes):
        """
        Test the trained agent on a fixed number of episodes.
        
        Parameters:
            episodes (int): Number of episodes to test the agent on.
        
        Returns:
            list: List of total rewards achieved by the agent in each episode.
        """
        self.load()
        self.epsilon = 0
        trewards = []
        for e in range(1, episodes + 1):
            state = self.env.reset()
            for i in range(len(self.env.data)):
                if self.env.bar >= len(self.env.data - 1):
                    break
                state = np.reshape(state, [1, self.observation_space_size])
                action = np.argmax(self.model.predict(state)[0])
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                if done:
                    treward = i + 1
                    trewards.append(treward)
                    print('episode: {:4d}/{} | treward: {:4d}'
                          .format(e, episodes, treward), end='\r')

        return trewards