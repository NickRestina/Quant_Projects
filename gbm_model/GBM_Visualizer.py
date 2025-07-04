import numpy as np
import math
import pandas as pd 
import matplotlib.pyplot as plt
import requests
import time
from datetime import datetime, timedelta
from alpaca.trading.client import *
from alpaca.trading.requests import *
from alpaca.trading.enums import *
from alpaca.data.live import StockDataStream
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from alpaca.data.enums import *
from alpaca_configfile import API_KEY, SECRET_KEY
ticker = 'CPER'
index_for_data = 5
days = 10

URL = 'https://paper-api.alpaca.markets/v2'
trading_client = TradingClient(API_KEY, SECRET_KEY)
headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY
}


time_value = datetime.now() - timedelta(days = days*index_for_data) #length of data for drift rate and volatility 
start_date = time_value.strftime('%Y-%m-%d')


def closing_prices(ticker):
    hist_url = 'https://data.alpaca.markets/v2/stocks/'+ticker+'/bars?timeframe=1Day&start='+start_date+'&limit=1000&adjustment=raw&feed=iex&sort=asc'
    response = requests.get(hist_url, headers=headers)
    hist_data = response.json()
    c_values = [item['c'] for item in hist_data["bars"]]
    closing_prices =pd.DataFrame(c_values)
    return closing_prices

def latest_price(ticker):
    url = 'https://data.alpaca.markets/v2/stocks/'+ticker+'/trades/latest?feed=iex'
    lp_response = requests.get(url, headers=headers)
    data = (lp_response.text).split(':')
    data_5 = data[5]
    latest_price = eval(data_5[:len(data_5)-4]) 
    return latest_price

log_returns = pd.DataFrame(columns = ['lgrt'])
log_returns = np.log(closing_prices(ticker) / closing_prices(ticker).shift(1))
log_returns = log_returns.dropna()
drift_rate = log_returns.mean().values[0]
daily_volatility = log_returns.std().values[0]


# drift coefficent
mu = drift_rate * 252
# number of steps
n = 500
# time in years
T = days/252
# number of sims
M = 1000
# initial stock price
S0 = latest_price(ticker)
# volatility
sigma = daily_volatility * np.sqrt(days)

dt = T/n
St = np.exp(
    (mu - sigma**2/2)*dt + sigma*np.random.normal(0, np.sqrt(dt), size = (M,n)).T
    
    )

St = np.vstack([np.ones(M), St])

St = S0 * St.cumprod(axis=0)

prices_at_end = St[n]

price_prediction = prices_at_end.mean()
max_value = prices_at_end.max()
min_value = prices_at_end.min()
print(latest_price(ticker))
print(price_prediction)
print("Top Limit" + str(max_value))
print("Bottom Limit" + str(min_value))

def graph():
    # Define time interval correctly
    time = np.linspace(0,T,n+1)

    # Require numpy array that is the same shape as St
    tt = np.full(shape=(M,n+1), fill_value=time).T
    plt.plot(tt, St)
    plt.xlabel("Years $(t)$")
    plt.ylabel("Stock Price $(S_t)$")
    plt.title(
    "Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(S0, mu, sigma)
    )
    plt.show()

graph()
