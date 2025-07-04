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

# Main execution flag - set to False to stop the trading bot
run_script = True
n = 420  # Lookback period - number of data points to compare current price against
ordersize = 5  # Number of shares to buy per trade

# Alpaca API credentials 
API_KEY = "" 
SECRET_KEY = "" 
URL = 'https://paper-api.alpaca.markets/v2'  # Paper trading (not real money)
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


url_list = []  #Stores urls for each ticker
stockprice_list = []  # stores historical price data
prices = []  # Temporary list to store current prices

# URLs for fetching latest trade data for each stock
for ticker in stock_ticker_list:
    url = 'https://data.alpaca.markets/v2/stocks/' + ticker + '/trades/latest?feed=iex'
    url_list.append(url)

def call_latest_price():
    """
    Main function that:
    1. Fetches current prices for all monitored stocks
    2. Calculates percentage changes over the lookback period
    3. Executes buy orders if conditions are met
    4. Manages the price history database
    """
    

    print(datetime.now())
    
    # Fetch latest prices for all stocks
    for url in url_list:
        # The response format requires splitting on ':' and extracting the price
        data = (requests.get(url, headers=headers).text).split(':')[5]
        price = eval(data[:len(data)-4])  # Extract price value from response
        prices.append(price)
    
    # Create a dictionary mapping tickers to their current prices
    current_prices = {ticker: price for ticker, price in zip(stock_ticker_list, prices)}
    stockprice_list.append(current_prices)
    
    # Convert price history to DataFrame for easier analysis
    price_dataframe = pd.DataFrame(stockprice_list)
    
    # Clear the temporary prices list for next iteration
    prices.clear()
    
    # Calculate indices for current and historical comparison
    current_index = len(price_dataframe) - 1  # Most recent price data
    index_n_ago = current_index - n  # Price data from n periods ago
    
    # Only proceed if we have enough historical data
    if index_n_ago >= 0:
        # Analyze each stock for trading opportunities
        for column in price_dataframe.columns:
            # Get prices for percentage change calculation
            price_n_ago = price_dataframe.iloc[index_n_ago][column]
            latest_price = price_dataframe.iloc[current_index][column]
            
            # Calculate percentage change over the lookback period
            pct_change = (latest_price - price_n_ago) / price_n_ago
            print(column + ': ' + str(pct_change))
            
            # Trading logic: Buy if stock dropped 0.7% during time interval
            if pct_change <= -0.007 and float(trading_client.get_account().cash) > n * latest_price:
                # Create a bracket order (buy with automatic take-profit and stop-loss)
                order = MarketOrderRequest(
                    symbol=column,
                    qty=ordersize,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(
                        limit_price=round((latest_price + (latest_price * 0.01)), 2)
                    ),
                    stop_loss=StopLossRequest(
                        stop_price=round((latest_price - (latest_price * 0.005)), 2)
                    )
                )
    
                trading_client.submit_order(order)

    
    # Limit the size of price history to prevent memory issues
    if len(stockprice_list) > n:
        stockprice_list.pop(0) 
    print(price_dataframe)
    time.sleep(2)
while run_script == True:
    call_latest_price()















    
   







    
   
