# Import necessary modules
from env import Environment
from agent import DQLAgent
import matplotlib.pyplot as plt
import numpy as np

# Create an instance of the environment and set the number of episodes to train the agent for
env = Environment('BTC-USD', 'Close', '5d')
episodes = 10

# Create an instance of the DQLAgent with a discount factor of 0.5 and the given environment
agent = DQLAgent(gamma=0.5, env=env)

# Train the agent for the specified number of episodes
agent.learn(episodes=episodes)

# Plot the moving average of the rewards achieved by the agent during training
plt.figure(figsize=(10, 6))
x = range(len(agent.averages))
y = np.polyval(np.polyfit(x, agent.averages, deg=3), x)
plt.plot(agent.averages, label='moving average')
plt.plot(x, y, 'r--', label='regression')
plt.xlabel('episodes')
plt.ylabel('total reward')
plt.legend()

# Test the trained agent on a single episode
agent.test(1)

# Calculate the returns and strategy of the agent on the test data and plot the results
agent.env.data_['returns'] = env.data_['Close'].pct_change()
agent.env.data_['strategy'] = env.data['action'] * env.data_['returns']
agent.env.data_['strategy'].cumsum().plot()
