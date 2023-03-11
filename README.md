# Quantitative AI Trading Bot
## Introduction
### Key concepts in Reinforcement Learning include:
* Agent: This is our AI that we train to interact within the environment
* Environment: This can be anything that gives us an observable state
* State: This is the current position of the agent
* Action: Based on the state, the agent determines the optimal action to take
* Reward: The environment returns a reward, which can be positive or negative

### In this case:
* Agent: Trading Algorithm
* Environment: Asset Market
* State: THistorical Price, Volume
* Action: Buy, Sell 
* Reward: Next period returns based on the action chosen and final return of the portfolio

### Data Source
* Symbol: S&P 500 (^GSPC)
* Date: 01/01/2009 to 05/08/2020 
* Frequency: hours 
* Attributes:
    * Open: The price at which the Symbol started trading during each hour.
    * High: The highest price reached by the Symbol during each hour.
    * Low: The lowest price reached by the Symbolx during each hour.
    * Close: The price at which the Symbol finished trading during each hour.
    * Adj Close: The adjusted closing price of the Symbol during each hour, taking into account any corporate actions that may have affected the stock price.
    * Volume: The total number of shares traded during each hour.

* How we get the stock data? The dataset is downloaded using yfinance package.

## Agent
We implement a Deep Q-Learning (DQL) agent. The agent uses a neural network to approximate the Q-values of state-action pairs, and learns to take actions that maximize long-term cumulative rewards.

## Environment
The trading environment in which the agent will interact.

It takes three parameters:
* symbol: The name of the stock symbol that the agent will trade.
* features: The features of the stock that the agent will use for trading, such as open, close, high, low, and volume.
* days: The number of days of data that the agent will use for trading.

The Environment class in "env.py" contains methods for getting the data for the specified stock symbol and features, creating features for the data, resetting the environment to its initial state, and stepping through the environment based on the agent's actions.

## State
The state is defined as a historical sequence of observations or features of the stock market data. And it is defined as a function of the current position of the bar, which is initialized to the look-back period.

During the training process, the agent interacts with the environment by selecting actions based on its current state, and receives a reward signal that is dependent on the correctness of its action. The agent updates its policy based on the received reward and the updated state of the environment.

## Reward Function

The reward function based on the correctness of the action taken by the agent. The function returns a reward of 1 if the action taken by the agent matches the current market direction, and 0 if it does not. 

## Action
An action represents a decision made by the agent in the environment. Specifically, the environment is designed to simulate trading of a stock, and the action space consists of two possible actions - buying or selling.

 
## Q & A
1. Why use Deep Reinforcement Learning (DRL) for algo trading?
* DRL doesn’t need **large labeled training datasets**. This is a significant advantage since the amount of data grows exponentially today, it becomes very time-and-labor-consuming to label a large dataset.
* Stock trading is a continuous process of testing new ideas, getting feedback from the market, and trying to optimize trading strategies over time. We can model the stock trading process as the Markov decision process which is the very foundation of Reinforcement Learning.
* Deep reinforcement learning algorithms can outperform human players in many challenging games. For example, on March 2016, DeepMind’s AlphaGo program, a deep reinforcement learning algorithm, beat the world champion Lee Sedol at the game of Go.
* The stock market provides sequential feedback. DRL can sequentially increase the model performance during the training process.

