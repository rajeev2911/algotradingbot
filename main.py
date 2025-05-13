import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_screener.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Configuration
class Config:
    # Stock Universe - can be expanded
    SP500_STOCKS = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'UNH', 'JPM',
        'V', 'JNJ', 'PG', 'XOM', 'MA', 'HD', 'CVX', 'MRK', 'LLY', 'ABBV'
    ]  # Top 20 stocks by market cap - can be expanded
    
    # Time periods for analysis
    LOOKBACK_PERIOD = '30d'  # For historical data
    LOOKBACK_PERIOD_LONG = '90d'  # For longer trend analysis
    
    # Pattern detection parameters
    MOMENTUM_THRESHOLD = 0.03  # 3% momentum over period
    VOLUME_THRESHOLD = 1.5     # 50% higher than average volume
    
    # Trading parameters
    SLTP_RATIO = 2.0
    ORDER_UNITS = 100  # Number of shares to trade
    
    # Risk management
    MAX_POSITION_SIZE_USD = 5000  # Maximum position size in USD
    MAX_TOTAL_RISK_PCT = 0.02     # Maximum risk per trade (2% of account)
    
    # Scan frequency
    SCAN_INTERVAL_MINUTES = 60


def get_stock_data(ticker, period=Config.LOOKBACK_PERIOD):
    """Fetch historical stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval="1d")
        
        if len(data) < 10:  # Ensure we have enough data
            logger.warning(f"Not enough data for {ticker}, skipping")
            return None
            
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None


def calculate_technical_indicators(df):
    """Calculate technical indicators for analysis"""
    if df is None or len(df) < 10:
        return None
    
    try:
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Calculate moving averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Calculate RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD (Moving Average Convergence Divergence)
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volume indicators
        df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_5']
        
        # Momentum indicators
        df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None


def identify_patterns(df):
    """Identify trading patterns and generate signals"""
    if df is None or len(df) < 20:
        return None, "Insufficient data"
    
    try:
        signals = {}
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Trend determination
        uptrend = latest['SMA_10'] > latest['SMA_20']
        
        # Momentum check
        strong_momentum = latest['Momentum_5d'] > Config.MOMENTUM_THRESHOLD
        
        # Volume check
        high_volume = latest['Volume_Ratio'] > Config.VOLUME_THRESHOLD
        
        # MACD crossover
        macd_crossover = (prev['MACD'] < prev['MACD_Signal']) and (latest['MACD'] > latest['MACD_Signal'])
        
        # RSI conditions
        oversold = latest['RSI'] < 30
        overbought = latest['RSI'] > 70
        
        # Calculate pattern scores
        bullish_score = 0
        bearish_score = 0
        
        # Bullish conditions
        if uptrend: bullish_score += 1
        if strong_momentum and latest['Momentum_5d'] > 0: bullish_score += 1
        if high_volume and latest['Close'] > prev['Close']: bullish_score += 1
        if macd_crossover: bullish_score += 1
        if oversold: bullish_score += 1
        
        # Bearish conditions
        if not uptrend: bearish_score += 1
        if strong_momentum and latest['Momentum_5d'] < 0: bearish_score += 1
        if high_volume and latest['Close'] < prev['Close']: bearish_score += 1
        if overbought: bearish_score += 1
        if prev['MACD'] > prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']: bearish_score += 1
        
        # Determine signal
        signal = 0  # No signal
        signal_type = "None"
        
        if bullish_score >= 3:
            signal = 2  # Buy
            signal_type = "Bullish"
        elif bearish_score >= 3:
            signal = 1  # Sell
            signal_type = "Bearish"
        
        signals = {
            'signal': signal,
            'signal_type': signal_type,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'current_price': latest['Close'],
            'momentum_5d': latest['Momentum_5d'],
            'rsi': latest['RSI'],
            'volume_ratio': latest['Volume_Ratio'],
            'macd': latest['MACD'],
            'uptrend': uptrend
        }
        
        return signals, None
    except Exception as e:
        logger.error(f"Error identifying patterns: {e}")
        return None, str(e)


def calculate_position_size(price, account_balance, max_risk_pct):
    """Calculate position size based on account risk management"""
    max_position = min(
        Config.MAX_POSITION_SIZE_USD,
        account_balance * max_risk_pct
    )
    shares = int(max_position / price)
    return max(1, min(shares, Config.ORDER_UNITS))


def calculate_sl_tp(current_price, atr, is_buy):
    """Calculate Stop Loss and Take Profit levels based on ATR"""
    if is_buy:
        sl = current_price - atr * 1.5
        tp = current_price + atr * Config.SLTP_RATIO * 1.5
    else:
        sl = current_price + atr * 1.5
        tp = current_price - atr * Config.SLTP_RATIO * 1.5
    
    return round(sl, 2), round(tp, 2)


def calculate_atr(df, period=14):
    """Calculate Average True Range for volatility measurement"""
    if df is None or len(df) < period:
        return 1.0  # Default fallback
    
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    
    return atr if not np.isnan(atr) else 1.0


def analyze_stock(ticker, account_balance=10000):
    """Analyze a single stock and return trading opportunities"""
    logger.info(f"Analyzing {ticker}")
    
    # Get stock data
    df = get_stock_data(ticker)
    if df is None:
        return None
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    if df is None:
        return None
    
    # Calculate ATR for volatility measurement
    atr = calculate_atr(df)
    
    # Identify patterns
    signals, error = identify_patterns(df)
    
    if error or signals is None:
        logger.warning(f"Error analyzing {ticker}: {error}")
        return None
    
    current_price = signals['current_price']
    
    # Calculate position size
    position_size = calculate_position_size(
        current_price, 
        account_balance, 
        Config.MAX_TOTAL_RISK_PCT
    )
    
    # Calculate stop loss and take profit
    sl, tp = calculate_sl_tp(
        current_price, 
        atr, 
        is_buy=(signals['signal'] == 2)
    )
    
    # Build result
    result = {
        'ticker': ticker,
        'signal': signals['signal'],
        'signal_type': signals['signal_type'],
        'current_price': current_price,
        'bullish_score': signals['bullish_score'],
        'bearish_score': signals['bearish_score'],
        'momentum_5d': signals['momentum_5d'],
        'rsi': signals['rsi'],
        'volume_ratio': signals['volume_ratio'],
        'recommended_position': position_size,
        'stop_loss': sl,
        'take_profit': tp
    }
    
    return result


def screen_stocks(account_balance=10000):
    """Screen all stocks in the universe and identify top opportunities"""
    logger.info(f"Starting stock screening for {len(Config.SP500_STOCKS)} stocks")
    
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        stock_analyses = list(executor.map(
            lambda ticker: analyze_stock(ticker, account_balance),
            Config.SP500_STOCKS
        ))
    
    # Filter out None results and sort by signal strength
    results = [r for r in stock_analyses if r is not None]
    
    # Separate buy and sell signals
    buy_signals = [r for r in results if r['signal'] == 2]
    sell_signals = [r for r in results if r['signal'] == 1]
    
    # Sort buy signals by bullish score (descending)
    buy_signals.sort(key=lambda x: (x['bullish_score'], x['momentum_5d']), reverse=True)
    
    # Sort sell signals by bearish score (descending)
    sell_signals.sort(key=lambda x: (x['bearish_score'], -x['momentum_5d']), reverse=True)
    
    return {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'no_signals': len(results) - len(buy_signals) - len(sell_signals)
    }


def display_top_stocks(screening_results, top_n=5):
    """Display the top stock opportunities"""
    timestamp = screening_results['timestamp']
    buy_signals = screening_results['buy_signals']
    sell_signals = screening_results['sell_signals']
    
    print(f"\n===== STOCK SCREENING RESULTS ({timestamp}) =====")
    
    print(f"\n----- TOP {min(top_n, len(buy_signals))} BUY OPPORTUNITIES -----")
    if buy_signals:
        for i, stock in enumerate(buy_signals[:top_n]):
            print(f"{i+1}. {stock['ticker']} - ${stock['current_price']:.2f} | "
                  f"Bullish Score: {stock['bullish_score']} | "
                  f"Momentum: {stock['momentum_5d']*100:.1f}% | "
                  f"RSI: {stock['rsi']:.1f} | "
                  f"Vol Ratio: {stock['volume_ratio']:.1f}x")
            print(f"   → Position: {stock['recommended_position']} shares | "
                  f"SL: ${stock['stop_loss']:.2f} | "
                  f"TP: ${stock['take_profit']:.2f}")
    else:
        print("No buy signals detected")
    
    print(f"\n----- TOP {min(top_n, len(sell_signals))} SELL/SHORT OPPORTUNITIES -----")
    if sell_signals:
        for i, stock in enumerate(sell_signals[:top_n]):
            print(f"{i+1}. {stock['ticker']} - ${stock['current_price']:.2f} | "
                  f"Bearish Score: {stock['bearish_score']} | "
                  f"Momentum: {stock['momentum_5d']*100:.1f}% | "
                  f"RSI: {stock['rsi']:.1f} | "
                  f"Vol Ratio: {stock['volume_ratio']:.1f}x")
            print(f"   → Position: {stock['recommended_position']} shares | "
                  f"SL: ${stock['stop_loss']:.2f} | "
                  f"TP: ${stock['take_profit']:.2f}")
    else:
        print("No sell signals detected")
    
    print(f"\nTotal stocks analyzed: {len(buy_signals) + len(sell_signals) + screening_results['no_signals']}")
    print(f"Stocks with buy signals: {len(buy_signals)}")
    print(f"Stocks with sell signals: {len(sell_signals)}")
    print(f"Stocks with no signals: {screening_results['no_signals']}")
    print("\n================================================")


def run_scheduled_screening():
    """Run the stock screening on a schedule"""
    while True:
        try:
            print(f"\nRunning stock screening at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            results = screen_stocks()
            display_top_stocks(results)
            
            next_run = datetime.now() + timedelta(minutes=Config.SCAN_INTERVAL_MINUTES)
            print(f"Next screening scheduled for {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            
            time.sleep(Config.SCAN_INTERVAL_MINUTES * 60)
        except KeyboardInterrupt:
            print("Screening stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in screening cycle: {e}")
            time.sleep(300)  # Wait 5 minutes on error


if __name__ == "__main__":
    print("Stock Screener and Trading Bot")
    print("------------------------------")
    print(f"Monitoring {len(Config.SP500_STOCKS)} stocks")
    
    # Run once immediately
    results = screen_stocks()
    display_top_stocks(results)
    
    # Option to run continuously
    run_continuous = input("\nRun continuous screening every hour? (y/n): ").lower() == 'y'
    
    if run_continuous:
        run_scheduled_screening()
    else:
        print("Screening complete. Exiting program.")