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
from alpaca_configfile import SECRET_KEY, API_KEY  

# Alpaca API configuration
URL = 'https://paper-api.alpaca.markets/v2'  # Paper trading endpoint
trading_client = TradingClient(API_KEY, SECRET_KEY)

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY
}

# Complete S&P 500 stock list for analysis
SP500 = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'BRK.B', 'LLY', 'AVGO', 'TSLA',
    'JPM', 'WMT', 'UNH', 'XOM', 'V', 'MA', 'PG', 'COST', 'JNJ', 'ORCL', 'HD', 'ABBV',
    'BAC', 'KO', 'MRK', 'NFLX', 'CVX', 'CRM', 'ADBE', 'AMD', 'PEP', 'TMO', 'TMUS',
    'LIN', 'ACN', 'MCD', 'CSCO', 'QCOM', 'DHR', 'ABT', 'WFC', 'TXN', 'GE', 'PM',
    'INTU', 'IBM', 'AXP', 'AMGN', 'AMAT', 'NOW', 'ISRG', 'VZ', 'CAT', 'GS', 'MS',
    'PFE', 'DIS', 'NEE', 'RTX', 'SPGI', 'UBER', 'CMCSA', 'UNP', 'PGR', 'T', 'LOW',
    'LMT', 'HON', 'COP', 'BLK', 'SYK', 'REGN', 'TJX', 'ELV', 'NKE', 'VRTX', 'BKNG',
    'SCHW', 'MU', 'ETN', 'C', 'BSX', 'LRCX', 'PLD', 'ANET', 'CB', 'UPS', 'ADI',
    'KLAC', 'BA', 'MMC', 'PANW', 'MDT', 'ADP', 'SBUX', 'KKR', 'DE', 'BX', 'AMT',
    'BMY', 'HCA', 'SO', 'FI', 'CI', 'MDLZ', 'GILD', 'ICE', 'SHW', 'INTC', 'MO',
    'DUK', 'MCO', 'SNPS', 'ZTS', 'CL', 'WM', 'GD', 'APH', 'EQIX', 'CTAS', 'TT',
    'CDNS', 'PH', 'CME', 'ABNB', 'NOC', 'CVS', 'EOG', 'AON', 'ITW', 'TDG', 'CMG',
    'MCK', 'WELL', 'MSI', 'MMM', 'PYPL', 'FDX', 'PNC', 'ECL', 'BDX', 'USB', 'TGT',
    'ORLY', 'CSX', 'NXPI', 'RSG', 'SLB', 'CRWD', 'AJG', 'FCX', 'MAR', 'CARR',
    'APD', 'MPC', 'CEG', 'EMR', 'ROP', 'PSX', 'AFL', 'DHI', 'NEM', 'FTNT', 'TFC',
    'PSA', 'AZO', 'NSC', 'WMB', 'ADSK', 'COF', 'HLT', 'O', 'OXY', 'AEP', 'SPG',
    'OKE', 'MET', 'GM', 'SRE', 'GEV', 'CHTR', 'CPRT', 'TRV', 'PCAR', 'ROST',
    'DLR', 'BK', 'VLO', 'KDP', 'KMB', 'CCI', 'AIG', 'ALL', 'LEN', 'URI', 'GWW',
    'D', 'KMI', 'COR', 'JCI', 'MNST', 'TEL', 'PAYX', 'STZ', 'MPWR', 'MSCI', 'IQV',
    'MCHP', 'LHX', 'FIS', 'FICO', 'ODFL', 'HUM', 'AMP', 'KHC', 'HES', 'EW', 'F',
    'CMI', 'KVUE', 'CNC', 'PRU', 'RCL', 'A', 'IDXX', 'PEG', 'NDAQ', 'PCG', 'HSY',
    'EA', 'HWM', 'PWR', 'GEHC', 'FAST', 'YUM', 'GIS', 'ACGL', 'KR', 'VRSK', 'AME',
    'DOW', 'SYY', 'EXC', 'CTSH', 'OTIS', 'IT', 'CTVA', 'IR', 'SMCI', 'EFX', 'EXR',
    'HPQ', 'FANG', 'BKR', 'ED', 'NUE', 'GLW', 'EL', 'CBRE', 'MRNA', 'DFS', 'XEL',
    'DD', 'VICI', 'RMD', 'GRMN', 'MLM', 'LULU', 'ON', 'VMC', 'HIG', 'XYL', 'EIX',
    'IRM', 'LYB', 'TRGP', 'AVB', 'CSGP', 'LVS', 'MTD', 'DXCM', 'CDW', 'ROK',
    'BIIB', 'BRO', 'WTW', 'TSCO', 'PPG', 'AXON', 'WEC', 'DVN', 'ANSS', 'ADM',
    'WAB', 'GPN', 'K', 'HAL', 'VST', 'AWK', 'EBAY', 'FITB', 'MTB', 'EQR', 'NTAP',
    'VLTO', 'DG', 'NVR', 'CAH', 'TTWO', 'DAL', 'PHM', 'DTE', 'ETR', 'IFF', 'TYL',
    'DOV', 'BR', 'FE', 'CHD', 'FTV', 'HPE', 'DECK', 'TROW', 'FSLR', 'STT', 'VTR',
    'RJF', 'KEYS', 'ROL', 'SBAC', 'ES', 'GDDY', 'PPL', 'STE', 'ZBH', 'SW', 'TSN',
    'WRB', 'AEE', 'LYV', 'WY', 'INVH', 'TER', 'STX', 'CBOE', 'WST', 'BF.B', 'WDC',
    'PTC', 'DLTR', 'MKC', 'MOH', 'CINF', 'CPAY', 'WAT', 'HBAN', 'ARE', 'ATO',
    'HUBB', 'LDOS', 'CMS', 'RF', 'CCL', 'TDY', 'GPC', 'BALL', 'LH', 'BLDR', 'EQT',
    'OMC', 'HOLX', 'CFG', 'COO', 'BAX', 'APTV', 'SYF', 'BBY', 'J', 'ESS', 'CLX',
    'STLD', 'ULTA', 'ALGN', 'WBD', 'PFG', 'ZBRA', 'MAA', 'CTRA', 'HRL', 'NTRS',
    'PKG', 'VRSN', 'FOX', 'L', 'JBHT', 'SWKS', 'DRI', 'EXPE', 'AVY', 'NRG', 'EXPD',
    'CNP', 'DGX', 'MAS', 'TXT', 'IP', 'EG', 'NWS', 'LUV', 'ENPH', 'TPL', 'NWSA',
    'FDS', 'DPZ', 'GEN', 'AKAM', 'KEY', 'UHS', 'KIM', 'DOC', 'SWK', 'IEX', 'RVTY',
    'AMCR', 'LNT', 'CPB', 'CF', 'SNA', 'CAG', 'NI', 'CE', 'VTRS', 'PNR', 'NDSN',
    'UAL', 'UDR', 'BG', 'PODD', 'EVRG', 'POOL', 'TRMB', 'JNPR', 'CPT', 'SJM',
    'REG', 'DVA', 'KMX', 'AES', 'JKHY', 'JBL', 'MGM', 'INCY', 'BEN', 'IPG', 'TECH',
    'AOS', 'CHRW', 'ALLE', 'HST', 'EPAM', 'FFIV', 'EMN', 'TFX', 'TAP', 'BXP',
    'CTLT', 'APA', 'LKQ', 'HII', 'QRVO', 'CRL', 'SOLV', 'RL', 'PNW', 'AIZ', 'WBA',
    'FRT', 'TPR', 'MHK', 'HAS', 'ALB', 'BIO', 'GNRC', 'PAYC', 'MTCH', 'LW', 'MOS',
    'HSIC', 'DAY', 'MKTX', 'GL', 'WYNN', 'CZR', 'FMC', 'IVZ', 'BBWI', 'BWA',
    'PARA', 'NCLH', 'AAL', 'ETSY'
]

# Smaller test set for faster execution during development and bug fixing
test_stocks = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'BRK.B', 'LLY', 'AVGO', 'TSLA',
    'JPM', 'WMT', 'UNH', 'XOM', 'V', 'MA', 'PG', 'COST', 'JNJ', 'ORCL', 'HD', 'ABBV', 'BAC'
]

# Model parameters
index_for_data = 3  # Multiplier for historical data period (3x the prediction period)
days = 20  # Number of days into the future for price prediction
pcts = []  # List to store percentage change predictions for all stocks

# Calculate start date for historical data collection
time_value = datetime.now() - timedelta(days=days * index_for_data)
start_date = time_value.strftime('%Y-%m-%d')

def closing_prices(ticker):
    """
    Fetch historical closing prices for a given stock ticker

    Returns:
        pd.DataFrame: DataFrame containing closing prices
    """
    # Construct API URL for historical daily bars
    hist_url = f'https://data.alpaca.markets/v2/stocks/{ticker}/bars?timeframe=1Day&start={start_date}&limit=1000&adjustment=raw&feed=sip&sort=asc'
    response = requests.get(hist_url, headers=headers)
    hist_data = response.json()
    
    # Extract closing prices from the response
    c_values = [item['c'] for item in hist_data["bars"]]
    closing_prices = pd.DataFrame(c_values)
    return closing_prices

def latest_price(ticker):
    """
    Get the most recent trade price for a stock
    
    Returns:
        float: Latest trade price
    """
    url = f'https://data.alpaca.markets/v2/stocks/{ticker}/trades/latest?feed=iex'
    lp_response = requests.get(url, headers=headers)
    
    # Parse the response to extract price (using string manipulation due to API format)
    data = (lp_response.text).split(':')
    data_5 = data[5]
    latest_price = eval(data_5[:len(data_5)-4])  
    return latest_price

def gbm_model(ticker):
    """
    Implement Geometric Brownian Motion model for stock price prediction
    
    This function:
    1. Calculates historical log returns and volatility
    2. Estimates drift rate and standard deviation
    3. Runs Monte Carlo simulation to predict future price
    4. Stores percentage change prediction
    
    """
    # Calculate log returns 
    log_returns = np.log(closing_prices(ticker) / closing_prices(ticker).shift(1))
    
    # Calculate statistics for GBM model
    dr = log_returns.mean()  # Average log return (drift rate)
    mean = closing_prices(ticker).mean()  # Average closing price
    sd = closing_prices(ticker).std()  # Standard deviation of closing prices
    
    # Extract scalar values for calculations
    drift_rate = dr[0]
    standard_deviation = sd[0] / mean  
    
    # GBM model parameters
    mu = drift_rate * 252  # Annualized drift coefficient (252 trading days/year)
    n = 1000  # Number of time steps in simulation
    T = days / 252  # Time horizon in years (252 trading days/year)
    M = 10000  # Number of Monte Carlo simulations
    S0 = latest_price(ticker)  # Current stock price (starting point)
    sigma = standard_deviation[0]  # Volatility parameter
    dt = T / n  # Time step size
    
    # Generate random price paths using GBM formula
    # St = S0 * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # where Z is normally distributed random variable
    St = np.exp(
        (mu - sigma**2/2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T
    )
    
    # Add initial condition and calculate cumulative product for price paths
    St = np.vstack([np.ones(M), St])
    St = S0 * St.cumprod(axis=0)
    
    # Extract final prices from all simulation paths
    prices_at_end = St[n]
    price_prediction = prices_at_end.mean()  # Average predicted price
    
    # Calculate percentage change from current price
    pct_change = (price_prediction - S0) / S0
    pcts.append(pct_change)
    
    # Display results
    print(ticker)
    print(f'Latest Price: {latest_price(ticker)}')
    print(f'Estimated Price in {days} days: {round(price_prediction, 3)}')
    print(pct_change)
    print('')

def run_command(input_stocks):
    """
    Main execution function that:
    1. Runs GBM analysis on all input stocks
    2. Ranks stocks by predicted performance
    3. Executes trades on top performers if conditions are met
    
    Args:
        input_stocks (list): List of stock tickers to analyze
    """
    # Clear previous results
    pcts.clear()
    
    # Run GBM model for each stock
    for ticker in input_stocks:
        gbm_model(ticker)
        time.sleep(2)  # Rate limiting to avoid API throttling
    
    # Create results DataFrame
    pcts_dataframe = pd.DataFrame({'Ticker': input_stocks, 'Performance': pcts})
    pcts_dataframe['Performance'] = pcts_dataframe['Performance'].astype(float)
    
    # Rank stocks by predicted performance
    largest = pcts_dataframe.nlargest(20, 'Performance', keep="all")  # Top 20 performers
    smallest = pcts_dataframe.nsmallest(20, 'Performance', keep="all")  # Bottom 20 performers
    
    # Get the 20th best performer's prediction as threshold
    test_estimate = largest["Performance"].iloc[19]
    
    # Position sizing- makes every position $2,000 to account for differences in prices between stocks
    buy_size = round(2000/latest_price(ticker), 2)
    
    # Display results
    print("Biggest Gains:")
    print(largest)
    print("")
    print("Biggest Losses:")
    print(smallest)
    
    # Execute trades only if the 20th best stock has >2% expected return
    if test_estimate > 0.02:
        print("\nExecuting trades for top performers...")
        
        # Place orders for all top performers
        for index, row in largest.iterrows():
            ticker = row['Ticker']
            estimation = row['Performance']
            
            # Calculate bracket order levels
            current_price = latest_price(ticker)
            take_profit = round(((estimation + 1) * current_price), 2)  # Target based on prediction
            stop_loss = round((current_price * 0.96), 2)  # 4% stop loss
            
            # Display order details
            print(f"\n{ticker}")
            print(f"Take Profit: {take_profit}")
            print(f"Stop Loss: {stop_loss}")
            
            # Create bracket order (buy with automatic take-profit and stop-loss)
            order = MarketOrderRequest(
                symbol=ticker,
                qty=buy_size,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit),
                stop_loss=StopLossRequest(stop_price=stop_loss)
            )
            
            
            trading_client.submit_order(order)

# Execute the trading system on all S&P 500 stocks
run_command(test_stocks)
