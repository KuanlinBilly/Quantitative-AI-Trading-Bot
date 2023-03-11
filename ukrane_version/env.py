import numpy as np
import random
import pandas as pd
from dataset import Dataset

# Action space: refers to the set of possible actions that an agent can take in a given state
class Actions:
    def __init__(self, n): #How many action the agent can take, like buy & sell => n = 2 (actions)
        self.n = n

    def sample(self): #Sometimes we do random actions, to find/explore new relationships in the data
        return random.randint(0, self.n - 1)

#observation: return the shape of n 
class Observation_space:
    def __init__(self, n) -> None:
        self.shape = (n,)


class Environment:
# features = 'Close'
    def __init__(self, symbol, features, days): #env = Environment('BTC-USD','Close','5d')
        self.days = days
        self.symbol = symbol #tick name of the stock
        self.features = features #features of the stock, like open, close, high, low, volume
        self.observation_space = Observation_space(4) #for lookback
        self.look_back = self.observation_space.shape[0] #lookback observation
        self.action_space = Actions(2) #only buy or sell => 2 actions in default
        self.min_accuracy = 0.55 # minimum accuracy of the agent model
        self._get_data()
        self._create_features()

    def _get_data(self): #data collection from module dataset.py for the agent

        self.data_ = Dataset().get_data(days=str(self.days), ticker=self.symbol, interval="1h")
        #data_ = Dataset().get_data(days='5d', ticker=['BTC-USD'], interval="1h") #sample input
        self.data = pd.DataFrame(self.data_[self.features].copy())

    def _create_features(self): #feature engineering
        self.data['return'] = np.log(self.data['Close'] / #daily return, 每筆資料的return
                                self.data['Close'].shift(1))
        self.data.dropna(inplace=True)
        #Normalize the data, cuz we'll feed the data into the neural network later!
        # Assume data is a numpy array containing the dataset, Calculate minimum and maximum values for each feature
        min_vals = np.min(self.data, axis=0)
        max_vals = np.max(self.data, axis=0)
        self.data = (self.data - min_vals) / (max_vals - min_vals)   # Perform min-max normalization
        #self.data = (self.data - self.data.mean()) / self.data.std() #Scaler(standardization)
        self.data['market direction'] = np.where(self.data['return'] > 0, 1, 0) #1 = up, 0 = down
        self.data['action'] = 0
    
    #current state:  return the current state of the trading environment.
    #what really exist in the current moment, like we can see the the past data for 4 days(look_back) in default
    def _get_state(self):  #current state:  return the current state of the trading environment.
        return self.data[self.features].iloc[
            self.bar - self.look_back:self.bar].values

    def reset(self): #reset the environment to initial state, when the done signal is received
        self.data['action'] = 0
        self.total_reward = 0
        self.accuracy = 0
        self.bar = self.look_back
        state = self.data[self.features].iloc[
            self.bar - self.look_back:self.bar]
        return state.values

    def step(self, action): #reward function based on the action taken by the agent
        # Get the current market direction
        current_market_direction = self.data['market direction'].iloc[self.bar]

        # Check if the agent's action matches the current market direction
        if action == current_market_direction:
            correct = True
        else:
            correct = False
        
        # Update action taken by agent in the data
        self.data['action'].iloc[self.bar] = action
        
        # Calculate reward based on correctness of action
        if correct == True:
            reward = 1
        else:
            reward = 0

        self.total_reward += reward #total reward of the agent
        self.bar += 1 # current lookback data in the data.
        self.accuracy = self.total_reward / (self.bar - self.look_back)

        # Check if episode is done, stop function =>done = True => stop the training agent
        if self.bar >= len(self.data) - 1:
            done = True
        elif reward == 1:
            done = False
        elif (self.accuracy < self.min_accuracy and #stop the training agent if the accuracy is too low
                self.bar > self.look_back + 10):
            done = True
        else:
            done = False

        state = self._get_state()
        info = {}

        return state, reward, done, info  