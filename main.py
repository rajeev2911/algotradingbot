# Enhanced Indian Stock Screener with Forex Trading
# This script analyzes multiple Indian stocks and forex pairs to identify top performers

# Core Imports
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import logging
import requests
from bs4 import BeautifulSoup
import pickle

# Cache directory
CACHE_DIR = 'data/cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def get_nse_stocks(use_cache=True, cache_days=1):
    """
    Fetch all stocks listed on NSE (National Stock Exchange of India)
    Returns a list of stock symbols with .NS suffix for Yahoo Finance
    """
    cache_file = os.path.join(CACHE_DIR, 'nse_stocks.pkl')
    
    # Check if cache exists and is recent
    if use_cache and os.path.exists(cache_file):
        modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - modified_time < timedelta(days=cache_days):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass  # If cache loading fails, fetch fresh data
    
    try:
        # Method 1: Scrape from NSE website
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get('https://www.nseindia.com/regulations/listing-compliance/nse-market-capitalisation-all-companies', headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract stock symbols from the table
        stocks = []
        table = soup.find('table', {'id': 'equityStockTable'})
        if table:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header row
                cols = row.find_all('td')
                if len(cols) > 1:
                    symbol = cols[1].text.strip()
                    stocks.append(f"{symbol}.NS")
        
        # If web scraping fails, fall back to top stocks
        if not stocks:
            stocks = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
                "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
                "SUNPHARMA.NS", "TATAMOTORS.NS", "BAJFINANCE.NS", "TITAN.NS", "WIPRO.NS"
            ]
            
            # Try Method 2: Use Yahoo Finance API to get NIFTY 500 constituents
            try:
                nifty500_url = "https://finance.yahoo.com/quote/%5ECNX/components?p=%5ECNX"
                response = requests.get(nifty500_url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                table = soup.find('table', {'class': 'W(100%)'})
                if table:
                    rows = table.find_all('tr')
                    stocks = []
                    for row in rows[1:]:  # Skip header
                        symbol = row.find('td').text.strip()
                        if symbol:
                            stocks.append(f"{symbol}.NS")
            except:
                pass  # Fall back to the default list if this fails too
        
        # Cache the results
        if stocks:
            with open(cache_file, 'wb') as f:
                pickle.dump(stocks, f)
        
        return stocks
    
    except Exception as e:
        print(f"Error fetching NSE stocks: {e}")
        # Fallback to default list
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
            "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
            "SUNPHARMA.NS", "TATAMOTORS.NS", "BAJFINANCE.NS", "TITAN.NS", "WIPRO.NS"
        ]

def get_forex_pairs(use_cache=True, cache_days=1):
    """
    Get a comprehensive list of forex pairs available on Yahoo Finance
    """
    cache_file = os.path.join(CACHE_DIR, 'forex_pairs.pkl')
    
    # Check if cache exists and is recent
    if use_cache and os.path.exists(cache_file):
        modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - modified_time < timedelta(days=cache_days):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass  # If cache loading fails, fetch fresh data
    
    try:
        # Comprehensive list of major forex pairs
        pairs = [
            # Major pairs
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
            
            # INR pairs (Indian Rupee)
            "INR=X", "EURINR=X", "GBPINR=X", "JPYINR=X", "AUDINR=X", "CADINR=X", "CHFINR=X", "NZDINR=X",
            
            # EUR pairs
            "EURGBP=X", "EURJPY=X", "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURNZD=X",
            
            # GBP pairs
            "GBPJPY=X", "GBPAUD=X", "GBPCAD=X", "GBPCHF=X", "GBPNZD=X",
            
            # JPY pairs
            "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
            
            # Other major crosses
            "AUDCAD=X", "AUDCHF=X", "AUDNZD=X", "CADCHF=X", "NZDCAD=X", "NZDCHF=X"
        ]
        
        # Cache the results
        with open(cache_file, 'wb') as f:
            pickle.dump(pairs, f)
        
        return pairs
    
    except Exception as e:
        print(f"Error creating forex pairs list: {e}")
        # Fallback to default list
        return [
            "INR=X", "EURINR=X", "GBPINR=X", "JPYINR=X", "EURUSD=X", 
            "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X"
        ]

# Use dynamic lists instead of hardcoded ones
indian_stocks = get_nse_stocks()
forex_pairs = get_forex_pairs()

# Limit the number of stocks/pairs to analyze if needed (to avoid API limits)
MAX_STOCKS = 50  # Adjust based on your needs
MAX_FOREX = 20   # Adjust based on your needs

if len(indian_stocks) > MAX_STOCKS:
    print(f"Limiting analysis to top {MAX_STOCKS} Indian stocks")
    indian_stocks = indian_stocks[:MAX_STOCKS]

if len(forex_pairs) > MAX_FOREX:
    print(f"Limiting analysis to top {MAX_FOREX} forex pairs")
    forex_pairs = forex_pairs[:MAX_FOREX]

# Analysis parameters
end_date = date.today()
start_date = end_date - timedelta(days=180)  # 6 months of data
interval = "1d"  # Daily data for broader analysis

# Technical indicator parameters
sma_short = 20
sma_long = 50
ema_period = 20
rsi_period = 14
bb_period = 20
macd_fast = 12
macd_slow = 26
macd_signal = 9

# Download data
def download_multiple_assets(tickers, start, end, interval="1d"):
    """Download market data for multiple tickers (stocks or forex)"""
    print(f"Downloading data for {len(tickers)} assets from {start} to {end}")

    all_data = {}
    for ticker in tqdm(tickers):
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, threads=False)
            if len(df) > 0:
                all_data[ticker] = df
            else:
                print(f"No data found for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    print(f"Successfully downloaded data for {len(all_data)} assets")
    return all_data

# Technical indicators
def calculate_indicators(df):
    """Calculate various technical indicators for analysis"""
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure we're working with Series objects, not DataFrames
    close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    # Moving Averages
    df['sma_short'] = close_series.rolling(sma_short).mean()
    df['sma_long'] = close_series.rolling(sma_long).mean()
    df['ema'] = close_series.ewm(span=ema_period, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = close_series.rolling(bb_period).mean()
    df['bb_std'] = close_series.rolling(bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # RSI (Relative Strength Index)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    df['ema_fast'] = close_series.ewm(span=macd_fast, adjust=False).mean()
    df['ema_slow'] = close_series.ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # Calculate returns
    df['daily_return'] = close_series.pct_change()
    df['cum_return'] = (1 + df['daily_return']).cumprod() - 1

    # Calculate volatility (rolling 30-day)
    df['volatility'] = df['daily_return'].rolling(30).std() * np.sqrt(252)  # Annualized

    # Initialize golden cross and death cross columns
    df['golden_cross'] = False
    df['death_cross'] = False

    # Need at least sma_long+1 data points to calculate crosses
    if len(df) > sma_long + 1:
        # Process golden crosses and death crosses - one row at a time to avoid Series truth value errors
        for i in range(sma_long + 1, len(df)):
            curr_short = df['sma_short'].iloc[i]
            curr_long = df['sma_long'].iloc[i]
            prev_short = df['sma_short'].iloc[i-1]
            prev_long = df['sma_long'].iloc[i-1]

            # Golden cross: short MA crosses above long MA
            if (not pd.isna(curr_short) and not pd.isna(curr_long) and
                not pd.isna(prev_short) and not pd.isna(prev_long)):
                if curr_short > curr_long and prev_short <= prev_long:
                    df.iloc[i, df.columns.get_loc('golden_cross')] = True
                # Death cross: short MA crosses below long MA
                elif curr_short < curr_long and prev_short >= prev_long:
                    df.iloc[i, df.columns.get_loc('death_cross')] = True

    return df

def calculate_performance_metrics(df, asset_type='stock'):
    """Calculate performance metrics for ranking stocks or forex pairs"""
    metrics = {}

    # Filter out NaN values
    returns = df['daily_return'].dropna()

    if len(returns) > 0:
        # Return metrics
        metrics['total_return'] = df['cum_return'].iloc[-1] if not pd.isna(df['cum_return'].iloc[-1]) else 0
        metrics['annualized_return'] = ((1 + metrics['total_return']) ** (252 / len(returns)) - 1) if metrics['total_return'] > -1 else -1

        # Risk metrics
        metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized volatility
        metrics['sharpe_ratio'] = (metrics['annualized_return'] / metrics['volatility']) if metrics['volatility'] > 0 else 0

        # Trend metrics - using safer evaluation
        try:
            # Use explicit scalar comparison with .item()
            close_value = df['Close'].iloc[-1].item() if isinstance(df['Close'].iloc[-1], pd.Series) else df['Close'].iloc[-1]
            sma_long_value = df['sma_long'].iloc[-1].item() if isinstance(df['sma_long'].iloc[-1], pd.Series) else df['sma_long'].iloc[-1]
            sma_short_value = df['sma_short'].iloc[-1].item() if isinstance(df['sma_short'].iloc[-1], pd.Series) else df['sma_short'].iloc[-1]

            # Now compare scalar values
            metrics['above_sma50'] = close_value > sma_long_value if not pd.isna(sma_long_value) else False
            metrics['above_sma20'] = close_value > sma_short_value if not pd.isna(sma_short_value) else False
        except Exception:
            # Fallback for any issues
            metrics['above_sma50'] = False
            metrics['above_sma20'] = False

        # Momentum metrics
        metrics['rsi'] = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50

        # MACD metrics
        metrics['macd'] = df['macd'].iloc[-1] if not pd.isna(df['macd'].iloc[-1]) else 0
        metrics['macd_signal'] = df['macd_signal'].iloc[-1] if not pd.isna(df['macd_signal'].iloc[-1]) else 0
        metrics['macd_histogram'] = df['macd_histogram'].iloc[-1] if not pd.isna(df['macd_histogram'].iloc[-1]) else 0
        
        # MACD bullish crossover (MACD line crosses above signal line)
        if len(df) > 2:
            curr_macd = df['macd'].iloc[-1]
            curr_signal = df['macd_signal'].iloc[-1]
            prev_macd = df['macd'].iloc[-2]
            prev_signal = df['macd_signal'].iloc[-2]
            
            metrics['macd_bullish_cross'] = (curr_macd > curr_signal and prev_macd <= prev_signal) if not pd.isna(curr_macd) and not pd.isna(curr_signal) else False
            metrics['macd_bearish_cross'] = (curr_macd < curr_signal and prev_macd >= prev_signal) if not pd.isna(curr_macd) and not pd.isna(curr_signal) else False
        else:
            metrics['macd_bullish_cross'] = False
            metrics['macd_bearish_cross'] = False

        # Recent performance (last 30 days)
        if len(df) > 30:
            recent_return = df['cum_return'].iloc[-1] - df['cum_return'].iloc[-30] if not pd.isna(df['cum_return'].iloc[-30]) else 0
            metrics['recent_30d_return'] = recent_return if not pd.isna(recent_return) else 0
        else:
            metrics['recent_30d_return'] = 0

        # Technical signals - ensure safe boolean evaluation
        try:
            golden_cross_recent = df['golden_cross'].iloc[-min(30, len(df)):]
            death_cross_recent = df['death_cross'].iloc[-min(30, len(df)):]

            # Use .any() or .sum() > 0 to safely evaluate Series
            metrics['golden_cross_last_30d'] = golden_cross_recent.sum() > 0
            metrics['death_cross_last_30d'] = death_cross_recent.sum() > 0
        except Exception:
            metrics['golden_cross_last_30d'] = False
            metrics['death_cross_last_30d'] = False

        # Forex specific metrics
        if asset_type == 'forex':
            if len(df) > 10:
                # Fix for the "Series truth value is ambiguous" error
                close_current = df['Close'].iloc[-1]
                close_10days_ago = df['Close'].iloc[-10]
                
                # Ensure we're working with scalar values, not Series
                if isinstance(close_current, pd.Series):
                    close_current = close_current.item()
                if isinstance(close_10days_ago, pd.Series):
                    close_10days_ago = close_10days_ago.item()
                
                # Now calculate momentum with scalar values
                if not pd.isna(close_10days_ago) and close_10days_ago != 0:
                    metrics['momentum_10d'] = (close_current / close_10days_ago - 1)
                else:
                    metrics['momentum_10d'] = 0
            else:
                metrics['momentum_10d'] = 0
                
            # Volatility comparison (forex pairs can have different volatility characteristics)
            mean_close = df['Close'].mean()
            if isinstance(mean_close, pd.Series):
                mean_close = mean_close.item()
            metrics['normalized_volatility'] = metrics['volatility'] / mean_close if mean_close > 0 else 0

    else:
        # Default values if not enough data
        metrics = {
            'total_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'above_sma50': False,
            'above_sma20': False,
            'rsi': 50,
            'macd': 0,
            'macd_signal': 0,
            'macd_histogram': 0,
            'macd_bullish_cross': False,
            'macd_bearish_cross': False,
            'recent_30d_return': 0,
            'golden_cross_last_30d': False,
            'death_cross_last_30d': False
        }
        
        # Add forex specific default metrics if needed
        if asset_type == 'forex':
            metrics['momentum_10d'] = 0
            metrics['normalized_volatility'] = 0

    return metrics

# Screening function
def screen_assets(all_data, asset_type='stock'):
    """Screen assets (stocks or forex) based on technical and fundamental criteria"""
    results = []

    for ticker, df in all_data.items():
        # Skip if not enough data
        if len(df) < max(sma_long, bb_period, rsi_period, 30):
            print(f"Not enough data for {ticker}, skipping...")
            continue

        # Calculate indicators
        df_with_indicators = calculate_indicators(df)

        # Calculate performance metrics
        metrics = calculate_performance_metrics(df_with_indicators, asset_type)

        # Add ticker to metrics
        metrics['ticker'] = ticker
        metrics['asset_type'] = asset_type
        
        # Different name formatting for stocks vs forex
        if asset_type == 'stock':
            metrics['name'] = ticker.split('.')[0]
        else:
            # Format forex pair names nicely (e.g., USD/INR instead of INR=X)
            if ticker.endswith('=X'):
                if ticker == 'INR=X':
                    metrics['name'] = 'USD/INR'
                elif ticker.startswith('USD'):
                    base = ticker[3:6]
                    metrics['name'] = f'USD/{base}'
                elif ticker.endswith('INR=X'):
                    base = ticker[:3]
                    metrics['name'] = f'{base}/INR'
                else:
                    parts = ticker.split('=')[0]
                    if len(parts) >= 6:
                        base = parts[:3]
                        quote = parts[3:6]
                        metrics['name'] = f'{base}/{quote}'
                    else:
                        metrics['name'] = ticker
            else:
                metrics['name'] = ticker
                
        metrics['current_price'] = df_with_indicators['Close'].iloc[-1]
        metrics['volume'] = df_with_indicators['Volume'].iloc[-1] if 'Volume' in df_with_indicators else 0

        # Calculate average volume (30 days)
        if 'Volume' in df_with_indicators:
            metrics['avg_volume_30d'] = df_with_indicators['Volume'].iloc[-min(30, len(df)):].mean() if len(df) > 0 else 0
        else:
            metrics['avg_volume_30d'] = 0

        results.append(metrics)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def score_assets(results_df, asset_type='stock'):
    """Score assets (stocks or forex) based on multiple performance metrics"""
    if len(results_df) == 0:
        return pd.DataFrame()

    # Work on a copy
    df = results_df.copy()

    # Initialize technical_score as float to allow fractional increments
    df['technical_score'] = 0.0

    # Return score (0–40)
    if 'total_return' in df.columns and not df['total_return'].isna().all():
        max_ret, min_ret = df['total_return'].max(), df['total_return'].min()
        rng = max_ret - min_ret
        df['return_score'] = 40 * (df['total_return'] - min_ret) / rng if rng > 0 else 20
    else:
        df['return_score'] = 0

    # Recent performance (0–20)
    if 'recent_30d_return' in df.columns and not df['recent_30d_return'].isna().all():
        max_r, min_r = df['recent_30d_return'].max(), df['recent_30d_return'].min()
        rng = max_r - min_r
        df['recent_score'] = 20 * (df['recent_30d_return'] - min_r) / rng if rng > 0 else 10
    else:
        df['recent_score'] = 0

    # Sharpe ratio (0–20)
    if 'sharpe_ratio' in df.columns and not df['sharpe_ratio'].isna().all():
        max_s, min_s = df['sharpe_ratio'].max(), df['sharpe_ratio'].min()
        rng = max_s - min_s
        df['sharpe_score'] = 20 * (df['sharpe_ratio'] - min_s) / rng if rng > 0 else 10
    else:
        df['sharpe_score'] = 0

    # Technical signal increments
    if 'above_sma20' in df.columns:
        df.loc[df['above_sma20'].astype(bool), 'technical_score'] += 5
    if 'above_sma50' in df.columns:
        df.loc[df['above_sma50'].astype(bool), 'technical_score'] += 5
    if 'golden_cross_last_30d' in df.columns:
        df.loc[df['golden_cross_last_30d'].astype(bool), 'technical_score'] += 5
    if 'death_cross_last_30d' in df.columns:
        df.loc[df['death_cross_last_30d'].astype(bool), 'technical_score'] -= 5
    if 'macd_bullish_cross' in df.columns:
        df.loc[df['macd_bullish_cross'].astype(bool), 'technical_score'] += 5
    if 'macd_bearish_cross' in df.columns:
        df.loc[df['macd_bearish_cross'].astype(bool), 'technical_score'] -= 5

    # Positive MACD value bonus (2.5 points)
    if 'macd' in df.columns:
        pos_mask = df['macd'] > 0
        for idx in df.index[pos_mask]:
            df.at[idx, 'technical_score'] += 2.5

    # RSI scoring
    df['rsi_score'] = 0
    if 'rsi' in df.columns:
        normal = (df['rsi'] >= 40) & (df['rsi'] <= 60)
        df.loc[normal, 'rsi_score'] = 5
        extreme = (df['rsi'] < 30) | (df['rsi'] > 70)
        df.loc[extreme, 'rsi_score'] = -5
        buy = (df['rsi'] >= 30) & (df['rsi'] < 40)
        df.loc[buy, 'rsi_score'] = 10
        sell = (df['rsi'] > 60) & (df['rsi'] <= 70)
        df.loc[sell, 'rsi_score'] = -10

    # Forex momentum scoring
    if asset_type == 'forex' and 'momentum_10d' in df.columns:
        max_m, min_m = df['momentum_10d'].max(), df['momentum_10d'].min()
        rng = max_m - min_m
        df['momentum_score'] = 10 * (df['momentum_10d'] - min_m) / rng if rng > 0 else 5
    else:
        df['momentum_score'] = 0

    # Sum all scores
    score_cols = ['return_score', 'recent_score', 'sharpe_score',
                  'technical_score', 'rsi_score', 'momentum_score']
    valid = [c for c in score_cols if c in df.columns]
    df['total_score'] = df[valid].sum(axis=1)

    # Sort descending
    return df.sort_values('total_score', ascending=False).reset_index(drop=True)


# Visualization function
def plot_asset(df, ticker, asset_name, asset_type='stock'):
    """Plot price charts with indicators for a single asset"""
    plt.figure(figsize=(14, 12))
    title_prefix = "Stock" if asset_type == 'stock' else "Forex Pair"
    plt.suptitle(f"{title_prefix} Technical Analysis: {asset_name} ({ticker})", fontsize=16)

    # Price and MAs
    plt.subplot(4, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price', linewidth=2)
    plt.plot(df.index, df['sma_short'], label=f'SMA({sma_short})', linestyle='--')
    plt.plot(df.index, df['sma_long'], label=f'SMA({sma_long})', linestyle='--')
    plt.plot(df.index, df['bb_upper'], label='BB Upper', color='gray', alpha=0.3)
    plt.plot(df.index, df['bb_lower'], label='BB Lower', color='gray', alpha=0.3)
    plt.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='gray', alpha=0.1)

    # Mark golden crosses and death crosses - safely find indices
    golden_crosses_idx = df.index[df['golden_cross'] == True]
    death_crosses_idx = df.index[df['death_cross'] == True]

    # Check if any crosses exist before plotting
    if len(golden_crosses_idx) > 0:
        # Use safe plotting method
        for idx in golden_crosses_idx:
            plt.scatter([idx], [df.loc[idx, 'Close']],
                       marker='^', color='green', s=150)
        # Add label just once for the legend
        plt.scatter([], [], marker='^', color='green', s=150, label='Golden Cross')

    if len(death_crosses_idx) > 0:
        # Use safe plotting method
        for idx in death_crosses_idx:
            plt.scatter([idx], [df.loc[idx, 'Close']],
                       marker='v', color='red', s=150)
        # Add label just once for the legend
        plt.scatter([], [], marker='v', color='red', s=150, label='Death Cross')

    plt.title("Price with Moving Averages and Bollinger Bands")
    plt.ylabel("Price" if asset_type == 'forex' else "Price (₹)")
    plt.grid(True)
    plt.legend()

    # RSI
    plt.subplot(4, 1, 2)
    plt.plot(df.index, df['rsi'], label='RSI', color='purple')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    plt.title("Relative Strength Index (RSI)")
    plt.ylabel("RSI")
    plt.grid(True)
    plt.legend()
    
    # MACD
    plt.subplot(4, 1, 3)
    plt.plot(df.index, df['macd'], label='MACD', color='blue')
    plt.plot(df.index, df['macd_signal'], label='Signal', color='red')
    plt.bar(df.index, df['macd_histogram'], label='Histogram', color='gray', alpha=0.3)
    plt.title("MACD (Moving Average Convergence Divergence)")
    plt.ylabel("MACD")
    plt.grid(True)
    plt.legend()

    # Cumulative returns
    plt.subplot(4, 1, 4)
    plt.plot(df.index, df['cum_return'] * 100, label='Cumulative Return (%)', color='green')
    plt.title("Cumulative Return")
    plt.ylabel("Return (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Visualization function for top assets
def plot_top_assets(all_data, top_assets_df, n=3, asset_type='stock'):
    """Plot price charts with indicators for top N assets"""
    top_n = min(n, len(top_assets_df))

    for i in range(top_n):
        ticker = top_assets_df['ticker'].iloc[i]
        asset_name = top_assets_df['name'].iloc[i]

        if ticker not in all_data:
            print(f"Data for {ticker} not found")
            continue

        df = calculate_indicators(all_data[ticker])
        plot_asset(df, ticker, asset_name, asset_type)

# Generate trading signals
def generate_trading_signals(all_data, results_df, top_n=5, asset_type=None):
    """Generate trading signals for top performing assets"""
    signals = []
    
    for i in range(min(top_n, len(results_df))):
        ticker = results_df['ticker'].iloc[i]
        asset_name = results_df['name'].iloc[i]
        asset_type_val = asset_type if asset_type is not None else results_df['asset_type'].iloc[i]
        
        if ticker not in all_data:
            continue
            
        df = calculate_indicators(all_data[ticker])
        
        # Current price
        current_price = df['Close'].iloc[-1]
        
        # Signal strength (0-100)
        signal_strength = min(100, max(0, results_df['total_score'].iloc[i]))
        
        # Determine signal type based on indicators
        signal_type = "NEUTRAL"
        confidence = "MEDIUM"
        
        # RSI conditions
        rsi = df['rsi'].iloc[-1]
        
        # MACD conditions
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        macd_hist = df['macd_histogram'].iloc[-1]
        
        # Golden/Death cross
        golden_cross_recent = results_df['golden_cross_last_30d'].iloc[i]
        death_cross_recent = results_df['death_cross_last_30d'].iloc[i]
        
        # Trend conditions
        above_sma20 = results_df['above_sma20'].iloc[i]
        above_sma50 = results_df['above_sma50'].iloc[i]
        
        # Check for bullish signals
        bullish_signals = 0
        if rsi > 50 and rsi < 70:
            bullish_signals += 1
        if macd > 0:
            bullish_signals += 1
        if macd > macd_signal:
            bullish_signals += 1
        if golden_cross_recent:
            bullish_signals += 2
        if above_sma20 and above_sma50:
            bullish_signals += 2
            
        # Check for bearish signals
        bearish_signals = 0
        if rsi < 50 and rsi > 30:
            bearish_signals += 1
        if macd < 0:
            bearish_signals += 1
        if macd < macd_signal:
            bearish_signals += 1
        if death_cross_recent:
            bearish_signals += 2
        if not above_sma20 and not above_sma50:
            bearish_signals += 2
            
        # Strong buy/sell signals
        if rsi < 30 and macd > macd_signal:
            signal_type = "STRONG BUY"
            confidence = "HIGH"
        elif rsi > 70 and macd < macd_signal:
            signal_type = "STRONG SELL"
            confidence = "HIGH"
        # Regular buy/sell based on signal count
        elif bullish_signals >= 4:
            signal_type = "BUY"
            confidence = "HIGH" if bullish_signals >= 5 else "MEDIUM"
        elif bearish_signals >= 4:
            signal_type = "SELL"
            confidence = "HIGH" if bearish_signals >= 5 else "MEDIUM"
        elif bullish_signals >= 3:
            signal_type = "BUY"
            confidence = "LOW"
        elif bearish_signals >= 3:
            signal_type = "SELL"
            confidence = "LOW"
            
        # Create signal dictionary
        signal = {
            'ticker': ticker,
            'name': asset_name,
            'asset_type': asset_type_val,
            'current_price': current_price,
            'signal': signal_type,
            'confidence': confidence,
            'rsi': rsi,
            'macd': macd,
            'signal_strength': signal_strength,
            'above_sma20': above_sma20,
            'above_sma50': above_sma50,
            'golden_cross_recent': golden_cross_recent,
            'death_cross_recent': death_cross_recent
        }
        
        signals.append(signal)
    
    # Convert to DataFrame
    signals_df = pd.DataFrame(signals)
    return signals_df

# Main function
def main():
    """Main function to run the stock and forex analysis"""
    print("Starting Indian Stock and Forex Analysis...")
    
    # Download data for Indian stocks
    stock_data = download_multiple_assets(indian_stocks, start_date, end_date, interval)
    
    # Calculate indicators for each stock
    stock_data_with_indicators = {}
    for ticker, df in stock_data.items():
        if len(df) > 0:
            stock_data_with_indicators[ticker] = calculate_indicators(df)
    
    # Screen stocks based on technical criteria
    stock_results = screen_assets(stock_data_with_indicators, asset_type='stock')
    
    # Score and rank stocks
    stock_scored = score_assets(stock_results, asset_type='stock')
    
    # Download data for forex pairs
    forex_data = download_multiple_assets(forex_pairs, start_date, end_date, interval)
    
    # Calculate indicators for each forex pair
    forex_data_with_indicators = {}
    for ticker, df in forex_data.items():
        if len(df) > 0:
            forex_data_with_indicators[ticker] = calculate_indicators(df)
    
    # Screen forex pairs based on technical criteria
    forex_results = screen_assets(forex_data_with_indicators, asset_type='forex')
    
    # Score and rank forex pairs
    forex_scored = score_assets(forex_results, asset_type='forex')
    
    # Generate trading signals for top stocks
    top_stock_tickers = stock_scored.head(5).index.tolist()
    stock_signals = generate_trading_signals(stock_data_with_indicators, stock_scored, top_n=5)
    
    # Generate trading signals for top forex pairs
    top_forex_tickers = forex_scored.head(5).index.tolist()
    forex_signals = generate_trading_signals(forex_data_with_indicators, forex_scored, top_n=5, asset_type='forex')
    
    # Run trading simulation on top stocks
    stock_simulation = run_algo_trading_simulation(stock_data_with_indicators, top_stock_tickers, initial_capital=100000, asset_type='stock')
    
    # Run trading simulation on top forex pairs
    forex_simulation = run_algo_trading_simulation(forex_data_with_indicators, top_forex_tickers, initial_capital=100000, asset_type='forex')
    
    # Save historical data for tracking
    save_historical_data(stock_scored, forex_scored)
    
    # Simulate options/futures data (placeholder - you would replace with actual data)
    options_today = generate_options_data()
    
    print("Analysis complete!")
    
    return {
        'stock_data': stock_data_with_indicators,
        'stock_results': stock_results,
        'stock_scored': stock_scored,
        'forex_data': forex_data_with_indicators,
        'forex_results': forex_results,
        'forex_scored': forex_scored,
        'stock_signals': stock_signals,
        'forex_signals': forex_signals,
        'stock_simulation': stock_simulation,
        'forex_simulation': forex_simulation,
        'options_today': options_today
    }

def save_historical_data(stock_data, forex_data):
    """Save today's top stocks and forex data for historical tracking"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Create data directory if it doesn't exist
    os.makedirs('data/history', exist_ok=True)
    
    # Convert DataFrames to dictionaries, ensuring all values are JSON serializable
    def convert_to_json_safe(df):
        if df.empty:
            return []
        
        # First convert to records
        records = df.head(10).to_dict(orient='records')
        
        # Then ensure all values are JSON serializable
        for record in records:
            for key, value in list(record.items()):
                # Convert numpy/pandas types to Python native types
                if hasattr(value, 'item'):
                    try:
                        record[key] = value.item()  # Convert numpy types to native Python
                    except:
                        record[key] = str(value)  # Fallback to string conversion
                elif isinstance(value, (pd.Series, pd.DataFrame)):
                    record[key] = str(value)  # Convert Series/DataFrame to string
                elif pd.isna(value):
                    record[key] = None  # Convert NaN/NaT to None
        
        return records
    
    # Convert data to JSON-safe format
    top_stocks = convert_to_json_safe(stock_data)
    top_forex = convert_to_json_safe(forex_data)
    
    # Create history entry
    history_entry = {
        'date': today,
        'top_stocks': top_stocks,
        'top_forex': top_forex
    }
    
    # Save to JSON file
    history_file = f'data/history/{today}.json'
    with open(history_file, 'w') as f:
        json.dump(history_entry, f, indent=4)
    
    # Update history index
    update_history_index(today)
    
    return history_entry

def update_history_index(date):
    """Update the history index file with new date entry"""
    index_file = 'data/history/index.json'
    
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            try:
                history_index = json.load(f)
            except json.JSONDecodeError:
                history_index = {'dates': []}
    else:
        history_index = {'dates': []}
    
    if date not in history_index['dates']:
        history_index['dates'].append(date)
        # Sort dates in descending order (newest first)
        history_index['dates'] = sorted(history_index['dates'], reverse=True)
    
    with open(index_file, 'w') as f:
        json.dump(history_index, f, indent=4)

def generate_options_data():
    """Generate sample options/futures data (placeholder)"""
    # This is a placeholder - you would replace with actual options data fetching
    options_data = [
        {
            'symbol': 'NIFTY',
            'expiry': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'strike': 22000,
            'call_price': 450.75,
            'put_price': 380.25,
            'call_oi': 12500,
            'put_oi': 15800,
            'call_volume': 3200,
            'put_volume': 4100,
            'call_iv': 18.5,
            'put_iv': 19.2,
            'recommendation': 'Bullish'
        },
        {
            'symbol': 'BANKNIFTY',
            'expiry': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'strike': 48000,
            'call_price': 850.50,
            'put_price': 720.75,
            'call_oi': 8500,
            'put_oi': 9200,
            'call_volume': 2100,
            'put_volume': 2500,
            'call_iv': 20.2,
            'put_iv': 21.5,
            'recommendation': 'Neutral'
        },
        {
            'symbol': 'RELIANCE',
            'expiry': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'strike': 2800,
            'call_price': 95.25,
            'put_price': 85.50,
            'call_oi': 5200,
            'put_oi': 4800,
            'call_volume': 1200,
            'put_volume': 980,
            'call_iv': 22.8,
            'put_iv': 23.5,
            'recommendation': 'Bullish'
        }
    ]
    return options_data

def get_key_stocks_today():
    """Get key stocks for today based on various criteria"""
    results = main()
    stock_scored = results['stock_scored']
    
    # Get top 5 stocks by overall score
    top_overall = stock_scored.head(5).to_dict(orient='records')
    
    # Get top 3 momentum stocks (highest RSI)
    top_momentum = stock_scored.sort_values('rsi', ascending=False).head(3).to_dict(orient='records')
    
    # Get top 3 value stocks (lowest volatility with positive returns)
    positive_return = stock_scored[stock_scored['total_return'] > 0]
    top_value = positive_return.sort_values('volatility').head(3).to_dict(orient='records')
    
    return {
        'top_overall': top_overall,
        'top_momentum': top_momentum,
        'top_value': top_value
    }

def get_historical_data(days=7):
    """Get historical data for the past N days"""
    index_file = 'data/history/index.json'
    
    if not os.path.exists(index_file):
        return []
    
    with open(index_file, 'r') as f:
        try:
            history_index = json.load(f)
        except json.JSONDecodeError:
            return []
    
    # Get the most recent N days
    recent_dates = history_index['dates'][:days]
    history_data = []
    
    for date in recent_dates:
        history_file = f'data/history/{date}.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                try:
                    day_data = json.load(f)
                    history_data.append(day_data)
                except json.JSONDecodeError:
                    continue
    
    return history_data

def backtest_strategy(ticker, data, start_date=None, end_date=None, initial_capital=100000):
    """Backtest a simple trading strategy for a given ticker"""
    # 1. Calculate indicators
    df = calculate_indicators(data)

    # 2. Drop only the initial NaN daily_return row if present
    if 'daily_return' in df.columns:
        df = df[df['daily_return'].notna()]

    # 3. Apply optional date filtering
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    # If not enough data to backtest, return defaults
    if len(df) < 2:
        return {
            'ticker': ticker,
            'start_date': None,
            'end_date': None,
            'initial_capital': initial_capital,
            'final_capital': initial_capital,
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'dataframe': df
        }

    # 4. Initialize positions and signals
    df['position'] = 0
    df['signal'] = 0

    # Helper to extract pure Python scalars
    def _val(x):
        return x.item() if hasattr(x, 'item') else x

    # 5. Backtesting loop
    for i in range(1, len(df)):
        # Current values as Python scalars
        rsi_cur   = _val(df.at[df.index[i], 'rsi']) if 'rsi' in df.columns else None
        macd_cur  = _val(df.at[df.index[i], 'macd']) if 'macd' in df.columns else None
        sig_cur   = _val(df.at[df.index[i], 'macd_signal']) if 'macd_signal' in df.columns else None
        close_cur = _val(df.at[df.index[i], 'Close'])
        sma20_cur = _val(df.at[df.index[i], 'sma_short']) if 'sma_short' in df.columns else None

        # Previous values
        rsi_prev   = _val(df.at[df.index[i-1], 'rsi']) if 'rsi' in df.columns else None
        macd_prev  = _val(df.at[df.index[i-1], 'macd']) if 'macd' in df.columns else None
        sig_prev   = _val(df.at[df.index[i-1], 'macd_signal']) if 'macd_signal' in df.columns else None
        close_prev = _val(df.at[df.index[i-1], 'Close'])
        sma20_prev = _val(df.at[df.index[i-1], 'sma_short']) if 'sma_short' in df.columns else None

        # Carry forward previous position
        prev_pos = df.at[df.index[i-1], 'position']
        df.at[df.index[i], 'position'] = prev_pos

        buy_signal = False
        sell_signal = False

        # Buy conditions
        if macd_cur is not None and sig_cur is not None and macd_cur > sig_cur and macd_prev <= sig_prev:
            buy_signal = True
        if rsi_cur is not None and rsi_prev is not None and rsi_cur > 30 and rsi_prev <= 30:
            buy_signal = True
        if sma20_cur is not None and close_cur > sma20_cur and close_prev <= sma20_prev:
            buy_signal = True
        if 'golden_cross' in df.columns and df.at[df.index[i], 'golden_cross']:
            buy_signal = True

        # Sell conditions
        if macd_cur is not None and sig_cur is not None and macd_cur < sig_cur and macd_prev >= sig_prev:
            sell_signal = True
        if rsi_cur is not None and rsi_cur > 70:
            sell_signal = True
        if sma20_cur is not None and close_cur < sma20_cur and close_prev >= sma20_prev:
            sell_signal = True
        if 'death_cross' in df.columns and df.at[df.index[i], 'death_cross']:
            sell_signal = True

        # Update signal and position
        if buy_signal and prev_pos <= 0:
            df.at[df.index[i], 'signal'] = 1
            df.at[df.index[i], 'position'] = 1
        elif sell_signal and prev_pos >= 0:
            df.at[df.index[i], 'signal'] = -1
            df.at[df.index[i], 'position'] = -1

    # 6. Compute returns
    df['strategy_return'] = df['position'].shift(1) * df['daily_return']
    df['strategy_return'].fillna(0, inplace=True)
    df['strategy_cum_return'] = (1 + df['strategy_return']).cumprod() - 1
    df['buy_hold_cum_return'] = (1 + df['daily_return']).cumprod() - 1

    # 7. Portfolio value and drawdown
    df['portfolio_value'] = initial_capital * (1 + df['strategy_cum_return'].fillna(0))
    df['drawdown'] = df['portfolio_value'] / df['portfolio_value'].cummax() - 1

    # 8. Performance metrics
    end_value    = df['portfolio_value'].iloc[-1]
    total_return = (end_value / initial_capital - 1) * 100
    days         = (df.index[-1] - df.index[0]).days or 1
    annual_return= ((end_value / initial_capital) ** (365 / days) - 1) * 100
    max_drawdown = df['drawdown'].min() * 100

    vol    = df['strategy_return'].std()
    sharpe = (df['strategy_return'].mean() / vol * np.sqrt(252)) if vol > 0 else 0.0

    trades   = df[df['signal'] != 0]
    n_trades = len(trades)
    win_rate = (len(trades[trades['strategy_return'] > 0]) / n_trades * 100) if n_trades > 0 else 0.0

    return {
        'ticker': ticker,
        'start_date': df.index[0] if not df.empty else None,
        'end_date': df.index[-1] if not df.empty else None,
        'initial_capital': initial_capital,
        'final_capital': end_value if not df.empty else initial_capital,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'num_trades': n_trades,
        'win_rate': win_rate,
        'dataframe': df
    }

def run_algo_trading_simulation(all_data, top_tickers, initial_capital=100000, asset_type='stock'):
    """Run algorithmic trading simulation on top assets"""
    results = []
    
    print(f"\nRunning algorithmic trading simulation for top {asset_type}s...")
    for ticker in top_tickers:
        if ticker not in all_data:
            print(f"Data for {ticker} not found, skipping...")
            continue
            
        print(f"Backtesting {ticker}...")
        backtest_result = backtest_strategy(ticker, all_data[ticker], initial_capital=initial_capital)
        results.append(backtest_result)
        
    # Display results
    print(f"\nAlgorithmic Trading Results for {asset_type.capitalize()}s:")
    if len(results) > 0:
        results_df = pd.DataFrame([
            {
                'Ticker': r['ticker'],
                'Total Return (%)': f"{r['total_return']:.2f}%",
                'Annual Return (%)': f"{r['annual_return']:.2f}%",
                'Max Drawdown (%)': f"{r['max_drawdown']:.2f}%",
                'Sharpe Ratio': round(r['sharpe_ratio'], 2),
                'Trades': r['num_trades'],
                'Win Rate (%)': f"{r['win_rate']:.1f}%",
                'Final Capital': f"₹{r['final_capital']:,.2f}" if asset_type == 'stock' else f"${r['final_capital']:,.2f}"
            }
            for r in results
        ])
        
        print(results_df.to_string(index=False))
        
        # Plot returns for the best performing asset
        best_idx = 0
        best_return = -float('inf')
        for i, r in enumerate(results):
            if r['total_return'] > best_return:
                best_return = r['total_return']
                best_idx = i
                
        if best_idx < len(results):
            best_result = results[best_idx]
            plot_trading_performance(best_result, asset_type)
    else:
        print(f"No valid {asset_type} data for simulation")
        
    return results

def plot_trading_performance(backtest_result, asset_type='stock'):
    """
    Plot the performance of a trading strategy, handling missing 'signal' gracefully
    to avoid KeyError and empty-legend warnings.
    """
    df = backtest_result.get('dataframe', pd.DataFrame())
    ticker = backtest_result.get('ticker', '')

    plt.figure(figsize=(14, 12))
    title_prefix = "Stock" if asset_type == 'stock' else "Forex"
    plt.suptitle(f"{title_prefix} Trading Strategy Performance for {ticker}", fontsize=16)

    # 1) Price and signals
    plt.subplot(3, 1, 1)
    if 'Close' in df.columns:
        plt.plot(df.index, df['Close'], label='Close Price')

    # Only plot buy/sell if 'signal' exists
    if 'signal' in df.columns:
        # Buy signals
        buys = df[df['signal'] == 1]
        if not buys.empty:
            plt.scatter(buys.index, buys['Close'],
                        marker='^', s=100, label='Buy Signal')

        # Sell signals
        sells = df[df['signal'] == -1]
        if not sells.empty:
            plt.scatter(sells.index, sells['Close'],
                        marker='v', s=100, label='Sell Signal')

    plt.title("Price and Trading Signals")
    plt.ylabel("Price (₹)" if asset_type == 'stock' else "Price")
    plt.grid(True)
    # Draw legend only if there are labeled artists
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()

    # 2) Portfolio value
    plt.subplot(3, 1, 2)
    if 'portfolio_value' in df.columns:
        plt.plot(df.index, df['portfolio_value'], label='Portfolio Value')

    plt.title("Portfolio Value Over Time")
    plt.ylabel("Value (₹)" if asset_type == 'stock' else "Value")
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()

    # 3) Strategy vs Buy & Hold returns
    plt.subplot(3, 1, 3)
    if 'strategy_cum_return' in df.columns:
        plt.plot(df.index, df['strategy_cum_return'] * 100,
                 label='Strategy Return')
    if 'buy_hold_cum_return' in df.columns:
        plt.plot(df.index, df['buy_hold_cum_return'] * 100,
                 label='Buy & Hold Return')

    plt.title("Strategy vs Buy & Hold Returns")
    plt.ylabel("Return (%)")
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Run the main program directly if needed for testing
if __name__ == "__main__":
    results = main()
    print("Analysis completed successfully.")
    print(f"Found {len(results['stock_scored'])} scored stocks.")
    print(f"Found {len(results['forex_scored'])} forex pairs.")
    print(f"Generated {len(results['options_today'])} options recommendations.")

    plt.subplot(3, 1, 2)
    if 'portfolio_value' in df.columns:
        plt.plot(df.index, df['portfolio_value'], label='Portfolio Value')

    plt.title("Portfolio Value Over Time")
    plt.ylabel("Value (₹)" if asset_type == 'stock' else "Value")
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()

    # 3) Strategy vs Buy & Hold returns
    plt.subplot(3, 1, 3)
    if 'strategy_cum_return' in df.columns:
        plt.plot(df.index, df['strategy_cum_return'] * 100,
                 label='Strategy Return')
    if 'buy_hold_cum_return' in df.columns:
        plt.plot(df.index, df['buy_hold_cum_return'] * 100,
                 label='Buy & Hold Return')

    plt.title("Strategy vs Buy & Hold Returns")
    plt.ylabel("Return (%)")
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Run the main program directly if needed for testing
if __name__ == "__main__":
    results = main()
    print("Analysis completed successfully.")
    print(f"Found {len(results['stock_scored'])} scored stocks.")
    print(f"Found {len(results['forex_scored'])} forex pairs.")
    print(f"Generated {len(results['options_today'])} options recommendations.")
