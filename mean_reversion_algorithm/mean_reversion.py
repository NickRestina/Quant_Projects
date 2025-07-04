import numpy as np
import pandas as pd 
import matplotlib as plt  
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
from alpaca_configfile import API_KEY, SECRET_KEY  # Import credentials from external config file

# Alpaca API configuration
URL = 'https://paper-api.alpaca.markets/v2'  # Paper trading  (not real money)
trading_client = TradingClient(API_KEY, SECRET_KEY)


headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY
}

# Calculate start date for historical data collection (9 weeks ago)
time_value = datetime.now() - timedelta(weeks=9)
start_date = time_value.strftime('%Y-%m-%d')

stock_ticker_list = [
    'INTC',  
    'AMD',   
    'AAPL',  
    'MSFT',  
    'TSM',   
    'TSLA' 
]

def comparision_fund_zscore(ticker):
    """
    Calculate the Z-score for a given stock/ETF to determine how many standard deviations
    the current price is from the historical mean.
    
    Z-score formula: (current_price - mean_price) / standard_deviation
    
    Interpretation:
    - Z-score < -2: Extremely oversold (more than 2 std dev below mean)
    - Z-score < -1: Oversold (more than 1 std dev below mean)
    - Z-score > 1: Overbought (more than 1 std dev above mean)
    - Z-score > 2: Extremely overbought (more than 2 std dev above mean)
    
    Args:
        ticker (str): Stock symbol (e.g., 'QQQ', 'SPY')
    
    Returns:
        pandas.Series: Z-score value for the stock
    """
    # Construct URLs for historical and current price data
    cf_hist_url = f'https://data.alpaca.markets/v2/stocks/{ticker}/bars?timeframe=1Day&start={start_date}&limit=1000&adjustment=raw&feed=sip&sort=asc'
    cf_latest_url = f'https://data.alpaca.markets/v2/stocks/{ticker}/trades/latest?feed=iex'
    
    # Fetch historical price data
    h_response = requests.get(cf_hist_url, headers=headers)
    cf_hist_data = h_response.json()
    cf_c_values = [item['c'] for item in cf_hist_data['bars']]  # Extract closing prices
    cf_closing_prices = pd.DataFrame(cf_c_values)
    
    # Fetch current/latest price
    lp_response = requests.get(cf_latest_url, headers=headers)
    data = (lp_response.text).split(':')
    data_5 = data[5]
    cf_latest_price = eval(data_5[:len(data_5)-4])  
    
    # Calculate statistical measures
    sd = cf_closing_prices.std()    # Standard deviation of historical prices
    mean = cf_closing_prices.mean()  # Mean of historical prices
    
    # Calculate Z-score: how many standard deviations is current price from mean
    z_score = (cf_latest_price - mean) / sd
    
    return z_score

def buy_prices(stock_ticker_list):
    """
    Execute limit buy orders for a list of stocks at prices that are 1 standard deviation
    below the historical mean (mean - 1*std_dev).
    
    This implements a mean reversion strategy assuming that stocks trading below
    their historical average by 1 standard deviation will eventually revert to the mean.
    
    Args:
        stock_ticker_list (list): List of stock tickers to place orders for
    """
    for ticker in stock_ticker_list:
        # Fetch historical data for each stock
        hist_url = f'https://data.alpaca.markets/v2/stocks/{ticker}/bars?timeframe=1Day&start={start_date}&limit=1000&adjustment=raw&feed=sip&sort=asc'
        response = requests.get(hist_url, headers=headers)
        hist_data = response.json()
        c_values = [item['c'] for item in hist_data['bars']]  # Extract closing prices
        closing_prices = pd.DataFrame(c_values)
        
        # Calculate statistical measures
        sd = closing_prices.std()    # Standard deviation
        mean = closing_prices.mean()  # Historical mean price
        
        # Calculate buy price: 1 standard deviation below the mean
        # This assumes the stock is "cheap" when it's trading below this level
        buy_price = (-1 * sd) + mean  # Equivalent to: mean - sd
        
        # Create limit order to buy at the calculated price
        limit_data = LimitOrderRequest(
            symbol=ticker,
            qty=20,  # Fixed quantity of 20 shares per order
            limit_price=round(buy_price[0], 2),  # Round to 2 decimal places
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,  # Order expires at end of trading day
        )
        
        # Submit the limit order to Alpaca
        trading_client.submit_order(limit_data)

#The program needs stop loss and take profit functionality
# Currently the system only places buy orders without exit strategies

# Main execution logic
print("QQQ Z-Score:", comparision_fund_zscore('QQQ'))

# Trading trigger: Only execute trades when QQQ (NASDAQ ETF) is oversold
# QQQ Z-score < -1 indicates the overall tech market is more than 1 standard deviation below its mean
# This suggests a potential buying opportunity in individual tech stocks
if comparision_fund_zscore('QQQ')[0] < -1:
    print("Market oversold condition detected. Placing buy orders...")
    buy_prices(stock_ticker_list)
else:
    print("Market not in oversold condition. No trades executed.")
