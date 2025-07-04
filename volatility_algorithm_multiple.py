import numpy as np
import pandas as pd 
import requests
import time
from alpaca.trading.client import *
from alpaca.trading.requests import *
from alpaca.trading.enums import *
from alpaca.data.live import StockDataStream
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from alpaca.data.enums import *
from datetime import datetime

run_script = True
n = 420 # n is equal to interval length of data 
ordersize = 5

API_KEY = 'PKPJCBXHLLEB25GDCOE3'
SECRET_KEY = 'Xi0tfoQYIvugpJ88FVQdTdwoaCUdd4uOPi9n6682'
URL = 'https://paper-api.alpaca.markets/v2'
trading_client = TradingClient(API_KEY, SECRET_KEY)
headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY
}

stock_ticker_list = [
    'NVDA',
    'AMD',
    'AAPL',
    'MSFT',
    'TSM',
    'TSLA'
]
url_list = []
stockprice_list = []
prices = []
for ticker in stock_ticker_list:
        url = 'https://data.alpaca.markets/v2/stocks/'+ticker+'/trades/latest?feed=iex'
        url_list.append(url) 

def call_latest_price():
    print(datetime.now())
    for url in (url_list):
        data = (requests.get(url, headers=headers).text).split(':')[5]
        price = eval(data[:len(data)-4])
        prices.append(price)
    stockprice_list.append({ticker: price for ticker, price in zip(stock_ticker_list, prices)})
    price_dataframe = pd.DataFrame(stockprice_list)
    prices.clear()
    current_index = len(price_dataframe) -1
    index_n_ago = current_index - n
    if index_n_ago >= 0:
        for column in price_dataframe.columns:
            price_n_ago = price_dataframe.iloc[index_n_ago][column]
            latest_price = price_dataframe.iloc[current_index][column]
            pct_change = (latest_price - price_n_ago)/price_n_ago
            print(column + ': ' + str(pct_change))
            if pct_change <= -.007 and float(trading_client.get_account().cash) > n * latest_price:
                    order = MarketOrderRequest(
                    symbol = column,
                    qty = ordersize,
                    side = OrderSide.BUY,
                    time_in_force = TimeInForce.DAY,
                    order_class = OrderClass.BRACKET,
                    take_profit = TakeProfitRequest(limit_price= round((latest_price + (latest_price*0.0047)), 2)),#percentage change for selling
                    stop_loss = StopLossRequest(stop_price= round((latest_price-(latest_price *0.005)), 2))#percentage change for stop loss
                    )   
                    trading_client.submit_order(order)
    print('')
    if len(stockprice_list) > n:
        stockprice_list.pop(0)

    print(price_dataframe)
          
    time.sleep(2)

while run_script == True:
    call_latest_price()
















    
   







    
   
