# Quantitative AI Trading Bot
## Problem Statement
Utilizing Reinforcement Learning to trade US stocks through Python, and hope to build a profitable trading bot!

<img title="Trading Bot" alt="Alt text" src="https://cdn.activestate.com/wp-content/uploads/2020/05/Trading_hero.jpg">

## Introduction
### Key concepts in Reinforcement Learning include:
* Agent: This is our AI that we train to interact within the environment
* Environment: This can be anything that gives us an observable state
* State: This is the current position https://github.com/KuanlinBilly/Quantitative-AI-Trading-Bot-2.0/blob/main/README.mdof the agent
* Action: Based on the state, the agent determines the optimal action to take
* Reward: The environment returns a reward, which can be positive or negative

### In this case:
* Agent: Trading Algorithm
* Environment: Asset Market
* State: THistorical Price, Volume
* Action: Buy, Sell 
* Reward: Next period returns based on the action chosen and final return of the portfolio

### Steps
1. Data Collection 
2. Data preprocessing
3. Design Environment & Agent
4. Implement DRL Algorithms
5. Backtest Our Strategy

### Data Collection 
Yahoo Finance is a website that provides stock data, financial news, financial reports, etc. All the data provided by Yahoo Finance is free. We can use the python package "yfinance" to help us gather the stock data. In this project we use S&P 500 for paper trading.
#### Info of the dataset
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

### Data preprocessing
Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
In this project, we do the following feature engineering: 
* Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: Moving Average,RSI and KD.

* Add Return column:   
```
data['return'] = np.log( data['Close'] /  data['Close'].shift(1))
```  
  
* Add Market Direction column:   
```
np.where(self.data['return'] > 0, 1, 0) #1 = up, 0 = down
```

## Design Environment & Agent

### Agent
We implement Deep Q-Learning (DQL), a critic-only approach to the agent. The agent uses a neural network to approximate the Q-values of state-action pairs, and learns to take actions that maximize long-term cumulative rewards.

### Environment
The trading environment in which the agent will interact.  

Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.

Our trading environments, takes three input parameters:
* symbol: The name of the stock symbol that the agent will trade.
* features: The features of the stock that the agent will use for trading, such as open, close, high, low, and volume.
* days: The number of days of data that the agent will use for trading.

As we mentioned above, the Environment class in "env.py" contains methods for getting the data for the specified stock symbol and features, creating features for the data, resetting the environment to its initial state, and stepping through the environment based on the agent's actions.

### State
The state is defined as a historical sequence of observations or features of the stock market data. And it is defined as a function of the current position of the bar, which is initialized to the look-back period.

During the training process, the agent interacts with the environment by selecting actions based on its current state, and receives a reward signal that is dependent on the correctness of its action. The agent updates its policy based on the received reward and the updated state of the environment.

### Reward Function

The reward function based on the correctness of the action taken by the agent. The function returns a reward of 1 if the action taken by the agent matches the current market direction, and 0 if it does not. 

### Action
An action represents a decision made by the agent in the environment. Specifically, the environment is designed to simulate trading of a stock, and the action space consists of two possible actions - buying or selling.


## Q & A
1. Why use Deep Reinforcement Learning (DRL) for algo trading?
* DRL doesn’t need **large labeled training datasets**. This is a significant advantage since the amount of data grows exponentially today, it becomes very time-and-labor-consuming to label a large dataset.
* Stock trading is a continuous process of testing new ideas, getting feedback from the market, and trying to optimize trading strategies over time. We can model the stock trading process as the Markov decision process which is the very foundation of Reinforcement Learning.
* Deep reinforcement learning algorithms can outperform human players in many challenging games. For example, on March 2016, DeepMind’s AlphaGo program, a deep reinforcement learning algorithm, beat the world champion Lee Sedol at the game of Go.
* The stock market provides sequential feedback. DRL can sequentially increase the model performance during the training process.

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)
- [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [Creating a Market Trading Bot Using Open AI Gym Anytrading](https://analyticsindiamag.com/creating-a-market-trading-bot-using-open-ai-gym-anytrading/)
- [How To Build An Algorithmic Trading Bot With Python](https://www.activestate.com/blog/how-to-build-an-algorithmic-trading-bot/)
