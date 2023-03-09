import pandas as pd #data manipulation
import numpy as np #numerical calculations
import yfinance as yf   

 
class Dataset:
    def __init__(self) -> None:
        pass

    def get_data(self, days: str, ticker: str, interval=str) -> pd.DataFrame():
        return yf.download(period=days, tickers=ticker, interval=interval)
