# Quantitative AI Trading Bot
## Introduction
### Key concepts in reinforcement learning include:
* Agent: This is our AI that we train to interact within the environment
* Environment: This can be anything that gives us an observable state
* State: This is the current position of the agent
* Action: Based on the state, the agent determines the optimal action to take
* Reward: The environment returns a reward, which can be positive or negative

### In this case:
* Agent: Trading Algorithm
* Environment: Asset Market
* State: THistorical Price, Volume
* Action: Buy, Sell, Hold
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

## Reward Function

* Sharpe Ratio: 

## Agent
We implement a Deep Q-Learning (DQL) agent. The agent uses a neural network to approximate the Q-values of state-action pairs, and learns to take actions that maximize long-term cumulative rewards.

## Ensemble Trading Strategy
Our purpose is to develop a highly robust trading strategy. So we use an ensemble method to automatically select the best performing agent among PPO, A2C, and DDPG agents based on the Sharpe ratio. The ensemble process is described as follows:

* Step 1. We use a growing window of 𝑛 months to retrain our three agents concurrently. In this paper we retrain our three agents at every 3 months.

* Step 2. We validate all 3 agents by using a 12-month validation- rolling window followed by the growing window we used for train- ing to pick the best performing agent which has the highest Sharpe ratio. We also adjust risk-aversion by using turbulence index in our validation stage.

* Step 3. After validation, we only use the best model which has the highest Sharpe ratio to predict and trade for the next quarter.

## Q & A
1. Why use Deep Reinforcement Learning (DRL) for algo trading?
* DRL doesn’t need **large labeled training datasets**. This is a significant advantage since the amount of data grows exponentially today, it becomes very time-and-labor-consuming to label a large dataset.
* Stock trading is a continuous process of testing new ideas, getting feedback from the market, and trying to optimize trading strategies over time. We can model the stock trading process as the Markov decision process which is the very foundation of Reinforcement Learning.
* Deep reinforcement learning algorithms can outperform human players in many challenging games. For example, on March 2016, DeepMind’s AlphaGo program, a deep reinforcement learning algorithm, beat the world champion Lee Sedol at the game of Go.
* The stock market provides sequential feedback. DRL can sequentially increase the model performance during the training process.

