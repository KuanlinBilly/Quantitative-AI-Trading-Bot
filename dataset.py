import pandas as pd #data manipulation
import numpy as np #numerical calculations
import yfinance as yf  #幫我們抓取分析資料

 
class Dataset:
    def __init__(self) -> None:
        pass

    def get_data(self, days: str, ticker: str, interval=str) -> pd.DataFrame():
        return yf.download(period=days, tickers=[ticker], interval=interval)
 
'''for debug

data_ = Dataset().get_data(days='1y', ticker='2330.TW', interval="1h")
'''