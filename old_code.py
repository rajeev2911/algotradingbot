# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import time
# from concurrent.futures import ThreadPoolExecutor
# import logging
# import os
# from oandapyV20 import API
# import oandapyV20.endpoints.orders as orders
# import oandapyV20.endpoints.positions as positions
# import oandapyV20.endpoints.accounts as accounts
# import oandapyV20.endpoints.pricing as pricing
# from oandapyV20.exceptions import V20Error
# import sys
# import yfinance as yf

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("indian_stock_algo_trader.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger()

# # OANDA API Configuration - Using your credentials
# class OandaConfig:
#     API_KEY = "f41d23e6f8e8a11d9db1f013b9f92b9b-d8042713fa92ee348434fbf479e3d52a"
#     ACCOUNT_ID = "101-001-31711066-001"
#     PRACTICE = True  # Set to False for live trading

# # Trading Configuration
# class Config:
#     # Default Indian Stock Universe - Top NSE stocks with proper NSE symbols
#     TRADABLE_INSTRUMENTS = [
#         'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
#         'HINDUNILVR', 'KOTAKBANK', 'BHARTIARTL', 'ITC',
#         'SBIN', 'BAJFINANCE', 'ASIANPAINT', 'LT', 'AXISBANK'
#     ]

#     # Execution mode ("simulation", "oanda_forex_only", "oanda_full")
#     EXECUTION_MODE = "simulation"  # We'll adjust this during initialization

#     # Time periods for analysis
#     LOOKBACK_PERIOD = '30d'  # For historical data
#     LOOKBACK_PERIOD_LONG = '90d'  # For longer trend analysis

#     # Pattern detection parameters
#     MOMENTUM_THRESHOLD = 0.02  # 2% momentum over period
#     VOLUME_THRESHOLD = 1.5     # 50% higher than average volume

#     SIGNAL_THRESHOLD = 2       # Minimum score for signal generation

#     # Trading parameters
#     SLTP_RATIO = 2.0           # Take profit to stop loss ratio

#     # Risk management
#     MAX_POSITION_SIZE_PCT = 0.05  # Maximum position size as % of account
#     MAX_TOTAL_RISK_PCT = 0.01     # Maximum risk per trade (1% of account)

#     # Scan frequency
#     SCAN_INTERVAL_MINUTES = 60

#     # Auto-trading settings
#     AUTO_TRADE = True         # Enable auto trading
#     MAX_OPEN_POSITIONS = 5     # Maximum number of simultaneous positions

# # Initialize OANDA API client
# api = API(access_token=OandaConfig.API_KEY, environment="practice" if OandaConfig.PRACTICE else "live")

# # FOREX pairs for Indian stocks simulation - these will be the instruments we use on OANDA
# # since OANDA doesn't directly support Indian stocks
# FOREX_PAIRS_MAPPING = {
#     'RELIANCE': 'USD_INR',
#     'TCS': 'EUR_USD',
#     'HDFCBANK': 'GBP_USD',
#     'INFY': 'USD_CAD',
#     'ICICIBANK': 'USD_JPY',
#     'HINDUNILVR': 'EUR_GBP',
#     'KOTAKBANK': 'AUD_USD',
#     'BHARTIARTL': 'USD_CHF',
#     'ITC': 'EUR_JPY',
#     'SBIN': 'GBP_JPY',
#     'BAJFINANCE': 'AUD_CAD',
#     'ASIANPAINT': 'EUR_AUD',
#     'LT': 'AUD_JPY',
#     'AXISBANK': 'NZD_USD'
# }

# def get_account_info():
#     """Get account details from OANDA"""
#     try:
#         r = accounts.AccountDetails(OandaConfig.ACCOUNT_ID)
#         response = api.request(r)
#         account_info = response['account']
#         return {
#             'balance': float(account_info['balance']),
#             'currency': account_info['currency'],
#             'open_positions': len(account_info.get('positions', [])),
#             'margin_available': float(account_info['marginAvailable']),
#             'margin_used': float(account_info['marginUsed']),
#             'margin_rate': float(account_info.get('marginRate', '0.05'))
#         }
#     except V20Error as e:
#         logger.error(f"Error fetching account info: {e}")
#         return None

# def get_open_positions():
#     """Get currently open positions"""
#     try:
#         r = positions.OpenPositions(OandaConfig.ACCOUNT_ID)
#         response = api.request(r)
#         return response['positions']
#     except V20Error as e:
#         logger.error(f"Error fetching open positions: {e}")
#         return []

# def create_alternative_execution_method():
#     """Create an alternative execution method since Indian stocks aren't available on OANDA"""
#     logger.info("Setting up alternative execution method for Indian stocks")

#     # Check if OANDA credentials are valid
#     try:
#         # Test OANDA connection
#         r = accounts.AccountSummary(OandaConfig.ACCOUNT_ID)
#         api.request(r)
#         logger.info("OANDA connection successful. Using forex pairs as proxies for Indian stocks.")
#         return "oanda_forex_only"
#     except V20Error as e:
#         logger.warning(f"OANDA connection failed: {e}. Using simulation mode.")
#         return "simulation"

# def simulate_order(instrument, units, is_buy, stop_loss, take_profit):
#     """Simulate order execution for stocks not available on OANDA"""
#     logger.info(f"SIMULATION: Placing order for {instrument}: {units} units, SL={stop_loss:.2f}, TP={take_profit:.2f}")

#     # Get current price
#     price_info = get_instrument_price(instrument)
#     if not price_info:
#         return {'success': False, 'error': f"Could not get price for {instrument}"}

#     # Simulate order execution
#     execution_price = price_info['ask'] if is_buy else price_info['bid']

#     # Generate a simulated order ID
#     order_id = f"sim_{instrument}_{int(time.time())}"

#     # Log the simulated order
#     with open("simulated_orders.csv", "a") as f:
#         f.write(f"{datetime.now()},{order_id},{instrument},{units},{is_buy},{execution_price},{stop_loss},{take_profit}\n")

#     return {
#         'success': True,
#         'order_id': order_id,
#         'units': units,
#         'instrument': instrument,
#         'is_buy': is_buy,
#         'stop_loss': stop_loss,
#         'take_profit': take_profit,
#         'execution_price': execution_price,
#         'simulated': True
#     }

# def get_instrument_price(symbol):
#     """Get current stock price using yfinance"""
#     try:
#         # For Indian stocks, append .NS for NSE
#         ticker = yf.Ticker(f"{symbol}.NS")
#         data = ticker.history(period="1d")

#         if len(data) > 0:
#             latest = data.iloc[-1]
#             # Estimate bid/ask with a small spread for the simulation
#             mid_price = latest['Close']
#             spread = mid_price * 0.001  # 0.1% spread

#             return {
#                 'instrument': symbol,
#                 'bid': mid_price - spread/2,
#                 'ask': mid_price + spread/2,
#                 'spread': spread,
#                 'timestamp': str(datetime.now())
#             }
#         else:
#             logger.warning(f"No price data found for {symbol}")
#             return None
#     except Exception as e:
#         logger.error(f"Error fetching price for {symbol}: {e}")
#         return None

# def get_historical_data(symbol, period="30d", interval="1d"):
#     """Get historical stock data using yfinance"""
#     try:
#         # For Indian stocks, append .NS for NSE
#         ticker = yf.Ticker(f"{symbol}.NS")
#         df = ticker.history(period=period, interval=interval)

#         if df.empty:
#             logger.warning(f"No historical data returned for {symbol}")
#             return None

#         return df
#     except Exception as e:
#         logger.error(f"Error fetching historical data for {symbol}: {e}")
#         return None

# def calculate_technical_indicators(df):
#     """Calculate technical indicators for stock analysis"""
#     if df is None or len(df) < 10:
#         return None

#     try:
#         # Make a copy to avoid SettingWithCopyWarning
#         df = df.copy()

#         # Calculate moving averages
#         df['SMA_10'] = df['Close'].rolling(window=10).mean()
#         df['SMA_20'] = df['Close'].rolling(window=20).mean()
#         df['SMA_50'] = df['Close'].rolling(window=50).mean()

#         # Calculate RSI (Relative Strength Index)
#         delta = df['Close'].diff()
#         gain = delta.where(delta > 0, 0).fillna(0)
#         loss = -delta.where(delta < 0, 0).fillna(0)

#         avg_gain = gain.rolling(window=14).mean()
#         avg_loss = loss.rolling(window=14).mean()

#         # Fix for division by zero
#         epsilon = 1e-10
#         rs = avg_gain / (avg_loss + epsilon)
#         df['RSI'] = 100 - (100 / (1 + rs))

#         # Calculate MACD (Moving Average Convergence Divergence)
#         df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
#         df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
#         df['MACD'] = df['EMA_12'] - df['EMA_26']
#         df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

#         # Volume indicators
#         df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
#         df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA_5'] + epsilon)

#         # Momentum indicators
#         df['Close_shifted_5'] = df['Close'].shift(5).fillna(df['Close'])
#         df['Close_shifted_10'] = df['Close'].shift(10).fillna(df['Close'])

#         # Calculate momentum
#         df['Momentum_5d'] = (df['Close'] / df['Close_shifted_5']) - 1
#         df['Momentum_10d'] = (df['Close'] / df['Close_shifted_10']) - 1

#         # Fill any remaining NaN values
#         df = df.bfill().ffill()

#         return df
#     except Exception as e:
#         logger.error(f"Error calculating indicators: {e}")
#         return None

# def identify_patterns(df):
#     """Identify stock trading patterns and generate signals"""
#     if df is None or len(df) < 20:
#         return None, "Insufficient data"

#     try:
#         # Make sure we have enough data points after calculations
#         if len(df) < 2:
#             return None, "Not enough data points after calculations"
#         if df.isnull().values.any():
#             df = df.bfill().ffill()

#         signals = {}
#         latest = df.iloc[-1]
#         prev = df.iloc[-2]

#         # Trend determination
#         uptrend = (latest['SMA_10'] > latest['SMA_20']) and (latest['Close'] > latest['SMA_50'])

#         # Momentum check
#         momentum_threshold = Config.MOMENTUM_THRESHOLD
#         strong_momentum = abs(latest['Momentum_5d']) > momentum_threshold

#         # Volume check
#         volume_threshold = Config.VOLUME_THRESHOLD
#         high_volume = latest['Volume_Ratio'] > volume_threshold

#         # MACD crossover
#         macd_crossover = (prev['MACD'] < prev['MACD_Signal']) and (latest['MACD'] > latest['MACD_Signal'])
#         macd_crossunder = (prev['MACD'] > prev['MACD_Signal']) and (latest['MACD'] < latest['MACD_Signal'])

#         # RSI conditions
#         oversold = latest['RSI'] < 30
#         overbought = latest['RSI'] > 70

#         # Calculate pattern scores
#         bullish_score = 0
#         bearish_score = 0

#         # Bullish conditions
#         if uptrend: bullish_score += 1
#         if strong_momentum and latest['Momentum_5d'] > 0: bullish_score += 1
#         if high_volume and latest['Close'] > prev['Close']: bullish_score += 1
#         if macd_crossover: bullish_score += 1
#         if oversold: bullish_score += 1

#         # Bearish conditions
#         if not uptrend: bearish_score += 1
#         if strong_momentum and latest['Momentum_5d'] < 0: bearish_score += 1
#         if high_volume and latest['Close'] < prev['Close']: bearish_score += 1
#         if overbought: bearish_score += 1
#         if macd_crossunder: bearish_score += 1

#         # Determine signal
#         signal = 0  # No signal
#         signal_type = "None"

#         if bullish_score >= Config.SIGNAL_THRESHOLD:
#             signal = 2  # Buy
#             signal_type = "Bullish"
#         elif bearish_score >= Config.SIGNAL_THRESHOLD:
#             signal = 1  # Sell
#             signal_type = "Bearish"

#         signals = {
#             'signal': signal,
#             'signal_type': signal_type,
#             'bullish_score': bullish_score,
#             'bearish_score': bearish_score,
#             'current_price': latest['Close'],
#             'momentum_5d': latest['Momentum_5d'],
#             'rsi': latest['RSI'],
#             'volume_ratio': latest['Volume_Ratio'],
#             'macd': latest['MACD'],
#             'uptrend': uptrend
#         }

#         return signals, None
#     except Exception as e:
#         logger.error(f"Error identifying patterns: {e}")
#         return None, str(e)

# def calculate_atr(df, period=14):
#     """Calculate Average True Range for volatility measurement"""
#     if df is None or len(df) < period:
#         return 0.02  # Default fallback (2%)

#     try:
#         high = df['High']
#         low = df['Low']
#         close = df['Close'].shift(1)

#         # Fill NaN values for first row
#         close = close.fillna(df['Close'])

#         tr1 = high - low
#         tr2 = abs(high - close)
#         tr3 = abs(low - close)

#         tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
#         atr = tr.rolling(window=period).mean().iloc[-1]

#         return atr if not np.isnan(atr) else 0.02
#     except Exception as e:
#         logger.error(f"Error calculating ATR: {e}")
#         return 0.02

# def calculate_position_size(price, account_balance, max_risk_pct, atr):
#     """Calculate position size based on account risk management"""
#     try:
#         # Calculate the stop loss distance
#         stop_loss_distance = atr * 1.5

#         # Calculate the risk per trade
#         risk_amount = account_balance * max_risk_pct

#         # Calculate shares based on risk per rupee
#         shares = int(risk_amount / stop_loss_distance)

#         # Make sure we buy at least 1 share
#         shares = max(1, shares)

#         # Apply max position size constraint
#         max_shares = int(account_balance * Config.MAX_POSITION_SIZE_PCT / price)
#         shares = min(shares, max_shares)

#         return shares
#     except Exception as e:
#         logger.error(f"Error calculating position size: {e}")
#         return 1  # Default to 1 share if calculation fails

# def calculate_sl_tp(current_price, atr, is_buy):
#     """Calculate Stop Loss and Take Profit levels based on ATR"""
#     try:
#         # Use a minimum ATR value to prevent very tight stops
#         atr = max(atr, current_price * 0.01)  # At least 1% of price

#         if is_buy:
#             sl = current_price - atr * 1.5
#             tp = current_price + atr * Config.SLTP_RATIO * 1.5
#         else:
#             sl = current_price + atr * 1.5
#             tp = current_price - atr * Config.SLTP_RATIO * 1.5

#         # Round to 2 decimal places
#         sl = round(sl, 2)
#         tp = round(tp, 2)

#         return sl, tp
#     except Exception as e:
#         logger.error(f"Error calculating SL/TP: {e}")
#         # Provide fallback values
#         if is_buy:
#             sl = round(current_price * 0.95, 2)
#             tp = round(current_price * 1.05, 2)
#         else:
#             sl = round(current_price * 1.05, 2)
#             tp = round(current_price * 0.95, 2)
#         return sl, tp

# def get_available_instruments():
#     """Get list of instruments available for trading on OANDA"""
#     try:
#         r = accounts.AccountInstruments(OandaConfig.ACCOUNT_ID)
#         response = api.request(r)
#         instruments = response.get('instruments', [])

#         available_instruments = {}
#         for instrument in instruments:
#             name = instrument.get('name')
#             display_name = instrument.get('displayName')
#             available_instruments[name] = display_name

#         logger.info(f"Retrieved {len(available_instruments)} available instruments from OANDA")
#         return available_instruments
#     except V20Error as e:
#         logger.error(f"Error fetching available instruments: {e}")
#         return {}

# def place_order(instrument, units, is_buy, stop_loss, take_profit):
#     """Place a market order with stop loss and take profit"""
#     # Map the Indian stock to a corresponding forex pair for OANDA
#     if Config.EXECUTION_MODE == "oanda_forex_only" and instrument in FOREX_PAIRS_MAPPING:
#         oanda_instrument = FOREX_PAIRS_MAPPING[instrument]
#         logger.info(f"Mapping Indian stock {instrument} to forex pair {oanda_instrument} for OANDA trading")

#         try:
#             # Format the price strings for OANDA
#             stop_loss_str = f"{stop_loss:.5f}"
#             take_profit_str = f"{take_profit:.5f}"

#             order_data = {
#                 "order": {
#                     "units": str(units if is_buy else -units),
#                     "instrument": oanda_instrument,
#                     "timeInForce": "FOK",
#                     "type": "MARKET",
#                     "positionFill": "DEFAULT",
#                     "stopLossOnFill": {
#                         "price": stop_loss_str,
#                         "timeInForce": "GTC"
#                     },
#                     "takeProfitOnFill": {
#                         "price": take_profit_str,
#                         "timeInForce": "GTC"
#                     }
#                 }
#             }

#             logger.info(f"Placing OANDA order for {instrument} (via {oanda_instrument}): {units} units, SL={stop_loss_str}, TP={take_profit_str}")

#             r = orders.OrderCreate(OandaConfig.ACCOUNT_ID, data=order_data)
#             response = api.request(r)

#             logger.info(f"Order placed: {response}")

#             # Extract order ID and units
#             order_id = response.get('orderCreateTransaction', {}).get('id')
#             order_units = response.get('orderCreateTransaction', {}).get('units')

#             return {
#                 'success': True,
#                 'order_id': order_id,
#                 'units': order_units,
#                 'instrument': instrument,
#                 'is_buy': is_buy,
#                 'stop_loss': stop_loss,
#                 'take_profit': take_profit,
#                 'oanda_instrument': oanda_instrument,
#                 'simulated': False
#             }
#         except V20Error as e:
#             logger.error(f"Error placing order for {instrument} via {oanda_instrument}: {e}")
#             return {'success': False, 'error': str(e)}
#     else:
#         # Use simulation mode if not using OANDA or if the instrument is not mapped
#         return simulate_order(instrument, units, is_buy, stop_loss, take_profit)

# def check_delisted_stocks():
#     """Check for and remove any delisted stocks from the universe"""
#     to_remove = []
#     for stock in Config.TRADABLE_INSTRUMENTS:
#         try:
#             # Attempt to get recent data
#             ticker = yf.Ticker(f"{stock}.NS")
#             data = ticker.history(period="5d")

#             if data.empty:
#                 logger.warning(f"Stock {stock} appears to be delisted - removing from universe")
#                 to_remove.append(stock)
#         except Exception as e:
#             logger.warning(f"Error checking {stock}, possibly delisted: {e}")
#             to_remove.append(stock)

#     # Remove delisted stocks from the universe
#     for stock in to_remove:
#         if stock in Config.TRADABLE_INSTRUMENTS:
#             Config.TRADABLE_INSTRUMENTS.remove(stock)

#     if to_remove:
#         logger.info(f"Removed {len(to_remove)} delisted stocks from universe: {to_remove}")

# def close_position(instrument):
#     """Close an open position for a stock"""
#     try:
#         # Check if we can trade this instrument via OANDA
#         if instrument in FOREX_PAIRS_MAPPING and Config.EXECUTION_MODE == "oanda_forex_only":
#             oanda_instrument = FOREX_PAIRS_MAPPING[instrument]

#             close_data = {"longUnits": "ALL", "shortUnits": "ALL"}
#             r = positions.PositionClose(OandaConfig.ACCOUNT_ID, instrument=oanda_instrument, data=close_data)
#             response = api.request(r)

#             logger.info(f"Position closed for {instrument} via {oanda_instrument}: {response}")
#             return {'success': True, 'response': response}
#         else:
#             # Simulation mode
#             logger.info(f"SIMULATION: Closing position for {instrument}")
#             return {'success': True, 'simulated': True}
#     except V20Error as e:
#         logger.error(f"Error closing position for {instrument}: {e}")
#         return {'success': False, 'error': str(e)}

# def analyze_stock(symbol):
#     """Analyze a single stock and return trading opportunities"""
#     logger.info(f"Analyzing {symbol}")

#     try:
#         # Get current market price
#         price_info = get_instrument_price(symbol)
#         if not price_info:
#             logger.warning(f"Could not get current price for {symbol}")
#             return None

#         # Get historical data
#         df = get_historical_data(symbol, period="60d")
#         if df is None or len(df) < 20:
#             logger.warning(f"Insufficient historical data for {symbol}")
#             return None

#         # Calculate technical indicators
#         df = calculate_technical_indicators(df)
#         if df is None:
#             logger.warning(f"Failed to calculate technical indicators for {symbol}")
#             return None

#         # Calculate ATR for volatility measurement
#         atr = calculate_atr(df)

#         # Identify patterns with adjusted threshold
#         signals, error = identify_patterns(df)

#         if error or signals is None:
#             logger.warning(f"Error analyzing {symbol}: {error}")
#             return None

#         # Get account info for position sizing
#         account_info = get_account_info()
#         if not account_info:
#             logger.error("Unable to retrieve account information")
#             return None

#         account_balance = account_info['balance']

#         current_price = signals['current_price']

#         # Calculate position size
#         position_size = calculate_position_size(
#             current_price,
#             account_balance,
#             Config.MAX_TOTAL_RISK_PCT,
#             atr
#         )

#         # Calculate stop loss and take profit levels
#         sl, tp = calculate_sl_tp(
#             current_price,
#             atr,
#             is_buy=(signals['signal'] == 2)
#         )

#         # Build result
#         result = {
#             'instrument': symbol,
#             'signal': signals['signal'],
#             'signal_type': signals['signal_type'],
#             'current_price': current_price,
#             'bid': price_info['bid'],
#             'ask': price_info['ask'],
#             'spread_pct': round((price_info['ask'] - price_info['bid']) / price_info['bid'] * 100, 2),
#             'bullish_score': signals['bullish_score'],
#             'bearish_score': signals['bearish_score'],
#             'momentum_5d': signals['momentum_5d'],
#             'rsi': signals['rsi'],
#             'volume_ratio': signals['volume_ratio'],
#             'recommended_shares': position_size,
#             'stop_loss': sl,
#             'take_profit': tp,
#             'atr': atr
#         }

#         logger.info(f"Successfully analyzed {symbol}")
#         return result
#     except Exception as e:
#         logger.error(f"Error in analyze_stock for {symbol}: {e}")
#         return None

# def screen_stocks():
#     """Screen all stocks in the universe and identify top opportunities"""
#     logger.info(f"Starting stock screening for {len(Config.TRADABLE_INSTRUMENTS)} stocks")

#     results = []

#     # Use ThreadPoolExecutor for parallel processing
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         stock_analyses = list(executor.map(
#             analyze_stock,
#             Config.TRADABLE_INSTRUMENTS
#         ))

#     # Filter out None results
#     results = [r for r in stock_analyses if r is not None]

#     # Separate buy and sell signals
#     buy_signals = [r for r in results if r['signal'] == 2]
#     sell_signals = [r for r in results if r['signal'] == 1]

#     # Sort buy signals by bullish score (descending)
#     buy_signals.sort(key=lambda x: (x['bullish_score'], x['momentum_5d']), reverse=True)

#     # Sort sell signals by bearish score (descending)
#     sell_signals.sort(key=lambda x: (x['bearish_score'], -x['momentum_5d']), reverse=True)

#     return {
#         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         'buy_signals': buy_signals,
#         'sell_signals': sell_signals,
#         'no_signals': len(results) - len(buy_signals) - len(sell_signals)
#     }

# def display_top_stocks(screening_results, top_n=5):
#     """Display the top trading opportunities"""
#     timestamp = screening_results['timestamp']
#     buy_signals = screening_results['buy_signals']
#     sell_signals = screening_results['sell_signals']

#     print(f"\n===== INDIAN STOCK TRADING OPPORTUNITIES ({timestamp}) =====")

#     print(f"\n----- TOP {min(top_n, len(buy_signals))} BUY OPPORTUNITIES -----")
#     if buy_signals:
#         for i, stock in enumerate(buy_signals[:top_n]):
#             print(f"{i+1}. {stock['instrument']} - ₹{stock['current_price']:.2f} | "
#                   f"Bullish Score: {stock['bullish_score']} | "
#                   f"Momentum: {stock['momentum_5d']*100:.2f}% | "
#                   f"RSI: {stock['rsi']:.1f} | "
#                   f"Spread: {stock['spread_pct']}%")
#             print(f"   → Recommend: {stock['recommended_shares']} shares | "
#                   f"SL: ₹{stock['stop_loss']:.2f} | "
#                   f"TP: ₹{stock['take_profit']:.2f}")
#     else:
#         print("No buy signals detected")

#     print(f"\n----- TOP {min(top_n, len(sell_signals))} SELL/SHORT OPPORTUNITIES -----")
#     if sell_signals:
#         for i, stock in enumerate(sell_signals[:top_n]):
#             print(f"{i+1}. {stock['instrument']} - ₹{stock['current_price']:.2f} | "
#                   f"Bearish Score: {stock['bearish_score']} | "
#                   f"Momentum: {stock['momentum_5d']*100:.2f}% | "
#                   f"RSI: {stock['rsi']:.1f} | "
#                   f"Spread: {stock['spread_pct']}%")
#             print(f"   → Recommend: {stock['recommended_shares']} shares | "
#                   f"SL: ₹{stock['stop_loss']:.2f} | "
#                   f"TP: ₹{stock['take_profit']:.2f}")
#     else:
#         print("No sell signals detected")

#     print(f"\nTotal stocks analyzed: {len(buy_signals) + len(sell_signals) + screening_results['no_signals']}")
#     print(f"Stocks with buy signals: {len(buy_signals)}")
#     print(f"Stocks with sell signals: {len(sell_signals)}")
#     print(f"Stocks with no signals: {screening_results['no_signals']}")
#     print("\n================================================")

# def execute_trades(screening_results):
#     """Execute trades based on the signals"""
#     if not Config.AUTO_TRADE:
#         logger.info("Auto-trading disabled. No trades executed.")
#         return

#     logger.info("Auto-trading enabled. Executing trades based on signals.")

#     # Get current open positions
#     open_positions = get_open_positions()
#     open_symbols = []

#     # Extract symbols from open positions
#     for position in open_positions:
#         instrument = position.get('instrument', '')
#         # Map back from OANDA instrument to stock symbol if needed
#         for stock, forex in FOREX_PAIRS_MAPPING.items():
#             if instrument == forex:
#                 open_symbols.append(stock)
#                 break

#     logger.info(f"Current open positions: {open_symbols}")

#     # Check if we have room for new positions
#     available_slots = Config.MAX_OPEN_POSITIONS - len(open_symbols)
#     if available_slots <= 0:
#         logger.info(f"Maximum number of positions ({Config.MAX_OPEN_POSITIONS}) already reached. No new trades.")
#         return

#     # Process buy signals first
#     executed_trades = 0
#     for signal in screening_results['buy_signals']:
#         # Skip if we already have this position
#         if signal['instrument'] in open_symbols:
#             logger.info(f"Position already open for {signal['instrument']}. Skipping.")
#             continue

#         # Skip if score is below minimum threshold
#         if signal['bullish_score'] < Config.SIGNAL_THRESHOLD:
#             continue

#         # Place buy order
#         shares = signal['recommended_shares']
#         stop_loss = signal['stop_loss']
#         take_profit = signal['take_profit']

#         logger.info(f"Placing BUY order for {signal['instrument']}: {shares} shares at ~₹{signal['current_price']:.2f}")

#         order_result = place_order(
#             signal['instrument'],
#             shares,
#             True,  # is_buy
#             stop_loss,
#             take_profit
#         )

#         if order_result['success']:
#             logger.info(f"Successfully placed BUY order for {signal['instrument']}")
#             executed_trades += 1
#         else:
#             logger.error(f"Failed to place BUY order for {signal['instrument']}")

#         # Break if we've filled all available slots
#         if executed_trades >= available_slots:
#             break

#     # If we still have slots, process sell signals for short positions
#     available_slots = Config.MAX_OPEN_POSITIONS - len(open_symbols) - executed_trades
#     if available_slots <= 0:
#         return

#     for signal in screening_results['sell_signals']:
#         # Skip if we already have this position
#         if signal['instrument'] in open_symbols:
#             logger.info(f"Position already open for {signal['instrument']}. Skipping.")
#             continue

#         # Skip if score is below minimum threshold
#         if signal['bearish_score'] < Config.SIGNAL_THRESHOLD:
#             continue

#         # Place sell order
#         shares = signal['recommended_shares']
#         stop_loss = signal['stop_loss']
#         take_profit = signal['take_profit']

#         logger.info(f"Placing SELL/SHORT order for {signal['instrument']}: {shares} shares at ~₹{signal['current_price']:.2f}")

#         order_result = place_order(
#             signal['instrument'],
#             shares,
#             False,  # is_buy
#             stop_loss,
#             take_profit
#         )

#         if order_result['success']:
#             logger.info(f"Successfully placed SELL/SHORT order for {signal['instrument']}")
#             executed_trades += 1
#         else:
#             logger.error(f"Failed to place SELL/SHORT order for {signal['instrument']}")

#         # Break if we've filled all available slots
#         if executed_trades >= available_slots:
#             break

# def fetch_top_indian_stocks(top_n=15):
#     """Fetch top performing Indian stocks based on recent performance"""
#     logger.info(f"Fetching top {top_n} Indian stocks...")

#     try:
#         # Here we would ideally use a more comprehensive API for Indian markets
#         # For now, let's use a combination of indices and manual addition of major stocks

#         # NSE Nifty 50 components - we'll use yfinance to check their performance
#         potential_stocks = [
#             'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
#             'KOTAKBANK', 'BHARTIARTL', 'ITC', 'SBIN', 'BAJFINANCE', 'ASIANPAINT',
#             'LT', 'AXISBANK', 'MARUTI', 'HCLTECH', 'WIPRO', 'NTPC', 'SUNPHARMA',
#             'ONGC', 'TATAMOTORS', 'ULTRACEMCO', 'TATASTEEL', 'ADANIPORTS', 'M&M'
#         ]

#         stock_performance = []

#         # Get 1-month performance for each stock
#         for stock in potential_stocks:
#             try:
#                 data = get_historical_data(stock, period="30d")
#                 if data is not None and len(data) > 5:
#                     first_price = data['Close'].iloc[0]
#                     last_price = data['Close'].iloc[-1]
#                     performance = (last_price / first_price - 1) * 100  # Percentage change

#                     stock_performance.append({
#                         'symbol': stock,
#                         'performance': performance,
#                         'price': last_price
#                     })
#             except Exception as e:
#                 logger.warning(f"Could not fetch performance for {stock}: {e}")

#         # Sort by performance
#         stock_performance.sort(key=lambda x: x['performance'], reverse=True)

#         # Get top N performing stocks
#         top_stocks = [stock['symbol'] for stock in stock_performance[:top_n]]

#         logger.info(f"Top {len(top_stocks)} Indian stocks fetched: {top_stocks}")
#         return top_stocks

#     except Exception as e:
#         logger.error(f"Error fetching top Indian stocks: {e}")
#         # Return default stocks if fetching fails
#         return Config.TRADABLE_INSTRUMENTS[:min(top_n, len(Config.TRADABLE_INSTRUMENTS))]

# def run_trading_cycle():
#     """Run a single trading cycle"""
#     logger.info(f"Running trading cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

#     # Get account information
#     account_info = get_account_info()
#     if account_info:
#         print(f"\nAccount Balance: {account_info['currency']} {account_info['balance']:.2f}")
#         print(f"Margin Available: {account_info['currency']} {account_info['margin_available']:.2f}")
#         print(f"Open Positions: {account_info['open_positions']}")
#         print("\n")

#     # Check for any delisted stocks
#     check_delisted_stocks()

#     # Run stock screening
#     screening_results = screen_stocks()

#     # Display top opportunities
#     display_top_stocks(screening_results)

#     # Execute trades based on signals
#     execute_trades(screening_results)

#     return screening_results

# def run_continuous_trading():
#     """Run continuous trading cycles with specified interval"""
#     logger.info(f"Starting continuous trading with {Config.SCAN_INTERVAL_MINUTES} minute intervals")

#     try:
#         while True:
#             # Run a trading cycle
#             run_trading_cycle()

#             # Calculate next run time
#             next_run = datetime.now() + timedelta(minutes=Config.SCAN_INTERVAL_MINUTES)
#             logger.info(f"Next trading cycle scheduled for {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

#             # Sleep until next cycle
#             time.sleep(Config.SCAN_INTERVAL_MINUTES * 60)
#     except KeyboardInterrupt:
#         logger.info("Trading stopped by user.")
#     except Exception as e:
#         logger.error(f"Error in continuous trading: {e}")

# def run_diagnostics():
#     """Run system diagnostics to verify everything is working"""
#     print("\n===== RUNNING SYSTEM DIAGNOSTICS =====")

#     # Check OANDA connection
#     print("\nChecking OANDA API connection...")
#     try:
#         r = accounts.AccountSummary(OandaConfig.ACCOUNT_ID)
#         api.request(r)
#         print("✓ OANDA API connection successful")
#     except Exception as e:
#         print(f"✗ OANDA API connection failed: {e}")

#     # Check yfinance data retrieval
#     print("\nChecking market data retrieval...")
#     try:
#         test_stock = Config.TRADABLE_INSTRUMENTS[0]
#         data = get_historical_data(test_stock, period="5d")
#         if data is not None and len(data) > 0:
#             print(f"✓ Successfully retrieved data for {test_stock}")
#         else:
#             print(f"✗ Failed to retrieve sufficient data for {test_stock}")
#     except Exception as e:
#         print(f"✗ Market data retrieval failed: {e}")

#     # Check technical indicators calculation
#     print("\nChecking technical indicators calculation...")
#     try:
#         if data is not None:
#             indicators = calculate_technical_indicators(data)
#             if indicators is not None:
#                 print("✓ Technical indicators calculation successful")
#             else:
#                 print("✗ Technical indicators calculation failed")
#         else:
#             print("✗ Cannot check indicators - no data available")
#     except Exception as e:
#         print(f"✗ Technical indicators check failed: {e}")

#     # Check available instruments on OANDA
#     print("\nChecking available instruments on OANDA...")
#     try:
#         instruments = get_available_instruments()
#         if instruments:
#             print(f"✓ Successfully retrieved {len(instruments)} instruments from OANDA")
#         else:
#             print("✗ Failed to retrieve instruments from OANDA")
#     except Exception as e:
#         print(f"✗ Instrument retrieval failed: {e}")

#     print("\n===== DIAGNOSTICS COMPLETE =====\n")

# def run_backtest():
#     """Run a simple parameter backtest"""
#     print("\n===== RUNNING PARAMETER BACKTEST =====")

#     # Test different signal thresholds
#     thresholds = [1, 2, 3]

#     for threshold in thresholds:
#         print(f"\nTesting signal threshold = {threshold}")
#         # Temporarily change the signal threshold
#         original_threshold = Config.SIGNAL_THRESHOLD
#         Config.SIGNAL_THRESHOLD = threshold

#         # Run stock screening without executing trades
#         screening_results = screen_stocks()
#         buy_count = len(screening_results['buy_signals'])
#         sell_count = len(screening_results['sell_signals'])

#         print(f"Buy signals: {buy_count}")
#         print(f"Sell signals: {sell_count}")
#         print(f"Total signals: {buy_count + sell_count}")

#         # Restore original threshold
#         Config.SIGNAL_THRESHOLD = original_threshold

#     print("\n===== BACKTEST COMPLETE =====\n")

# def main():
#     """Main function to run the trading system"""
#     print("\n===== INDIAN STOCK ALGORITHMIC TRADING SYSTEM =====")
#     print("Auto-trading:", "Enabled" if Config.AUTO_TRADE else "Disabled")
#     print("Running on", "Practice account" if OandaConfig.PRACTICE else "Live account")

#     # Set up execution mode
#     Config.EXECUTION_MODE = create_alternative_execution_method()

#     # Fetch top Indian stocks
#     Config.TRADABLE_INSTRUMENTS = fetch_top_indian_stocks(15)

#     # Check if we can connect to the OANDA API
#     try:
#         r = accounts.AccountSummary(OandaConfig.ACCOUNT_ID)
#         api.request(r)
#         print("Successfully connected to OANDA API")

#         # Get account information
#         account_info = get_account_info()
#         if account_info:
#             print(f"Account Balance: {account_info['currency']} {account_info['balance']:.2f}")
#             print(f"Margin Available: {account_info['currency']} {account_info['margin_available']:.2f}")
#     except Exception as e:
#         print(f"Failed to connect to OANDA API: {e}")
#         print("Running in simulation mode only")
#         Config.EXECUTION_MODE = "simulation"

#     # Main menu loop
#     while True:
#         print("\nWhat would you like to do?")
#         print("1. Run trading cycle")
#         print("2. Run continuous trading")
#         print("3. Run diagnostics")
#         print("4. Run parameter backtest")
#         print("5. Check available instruments")
#         print("6. Update stock universe automatically")

#         try:
#             choice = input("Enter choice (1-6): ")

#             if choice == '1':
#                 run_trading_cycle()
#             elif choice == '2':
#                 run_continuous_trading()
#             elif choice == '3':
#                 run_diagnostics()
#             elif choice == '4':
#                 run_backtest()
#             elif choice == '5':
#                 instruments = get_available_instruments()
#                 print(f"\nAvailable instruments on OANDA ({len(instruments)}):")
#                 for name, display_name in instruments.items():
#                     print(f"- {name}: {display_name}")
#             elif choice == '6':
#                 Config.TRADABLE_INSTRUMENTS = fetch_top_indian_stocks(15)
#                 print(f"Updated stock universe: {Config.TRADABLE_INSTRUMENTS}")
#             else:
#                 print("Invalid choice. Please try again.")
#         except KeyboardInterrupt:
#             print("\nExiting program...")
#             break
#         except Exception as e:
#             logger.error(f"Error in main menu: {e}")
#             print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()



























# # Core Imports
# import pandas as pd
# import numpy as np
# import datetime
# import yfinance as yf
# import matplotlib.pyplot as plt

# # Configuration
# ticker = "^NSEI"  # NSE Index
# end_date = datetime.date.today()
# start_date = end_date - pd.Timedelta(days=30)  # Extended period for better analysis
# sma_period = 12
# ema_period = 20  # Define the EMA period that was missing
# bb_period = 20   # Bollinger Band period

# # Download data
# def download_data(ticker, start, end, interval="5m"):
#     """Download market data for the specified ticker and timeframe"""
#     print(f"Downloading {ticker} data from {start} to {end}")
#     df = yf.download(ticker, start=start, end=end, interval=interval)

#     # Fix MultiIndex columns if they exist
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.droplevel(1)  # Drop the ticker level

#     print(f"Downloaded {len(df)} data points")
#     print(df.head())
#     return df

# # Technical indicators
# def get_sma(prices, period):
#     """Calculate Simple Moving Average"""
#     return prices.rolling(period).mean()

# def get_ema(prices, period):
#     """Calculate Exponential Moving Average"""
#     return prices.ewm(span=period, adjust=False).mean()

# def get_bollinger_bands(prices, period, num_std=2):
#     """Calculate Bollinger Bands"""
#     sma = get_sma(prices, period)
#     std = prices.rolling(period).std()
#     bollinger_up = sma + std * num_std
#     bollinger_down = sma - std * num_std
#     return bollinger_up, bollinger_down

# def compute_daily_returns(data):
#     """Compute log returns based on Close prices"""
#     data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
#     return data

# # Main process
# print(f"Starting analysis for {ticker}")

# # Download data
# df = download_data(ticker, start_date, end_date)
# df.info()  # Display column information for debugging

# # Calculate technical indicators
# df['sma'] = get_sma(df['Close'], sma_period)
# df['ema'] = get_ema(df['Close'], ema_period)
# bollinger_up, bollinger_down = get_bollinger_bands(df['Close'], bb_period)
# df['bollinger_up'] = bollinger_up
# df['bollinger_down'] = bollinger_down

# # Calculate returns
# df['returns'] = df['Close'].pct_change()

# # Strategy: Buy when price is above SMA, Sell when price is below SMA
# df['position'] = np.where(df['Close'] > df['sma'], 1, 0)
# df['position'] = df['position'].shift(1).fillna(0)  # Shift and fill NaN values

# # Calculate strategy returns
# df['strategy_returns'] = df['returns'] * df['position']

# # Calculate cumulative returns
# df['cum_market_returns'] = (1 + df['returns']).cumprod()
# df['cum_strategy_returns'] = (1 + df['strategy_returns']).cumprod()

# # Add transaction costs (example: 0.1% per trade)
# df['trades'] = df['position'].diff().abs()
# df['transaction_cost'] = df['trades'] * 0.001  # 0.1% cost per trade
# df['strategy_returns_after_costs'] = df['strategy_returns'] - df['transaction_cost']
# df['cum_strategy_returns_after_costs'] = (1 + df['strategy_returns_after_costs']).cumprod()

# # Visualize data
# def plot_price_with_indicators(df):
#     """Plot price chart with technical indicators"""
#     plt.figure(figsize=(14, 7))
#     plt.title(f'{ticker} Price with Technical Indicators')
#     plt.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
#     plt.plot(df.index, df['sma'], label=f'SMA({sma_period})', alpha=0.8)
#     plt.plot(df.index, df['ema'], label=f'EMA({ema_period})', alpha=0.8)
#     plt.plot(df.index, df['bollinger_up'], label='Bollinger Up', linestyle='--', alpha=0.6)
#     plt.plot(df.index, df['bollinger_down'], label='Bollinger Down', linestyle='--', alpha=0.6)

#     # Plot buy/sell signals
#     buy_signals = df[df['position'].diff() > 0].index
#     sell_signals = df[df['position'].diff() < 0].index

#     if len(buy_signals) > 0:
#         plt.scatter(buy_signals, df.loc[buy_signals, 'Close'], marker='^', color='green', s=100, label='Buy Signal')

#     if len(sell_signals) > 0:
#         plt.scatter(sell_signals, df.loc[sell_signals, 'Close'], marker='v', color='red', s=100, label='Sell Signal')

#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def plot_returns(df):
#     """Plot cumulative returns"""
#     plt.figure(figsize=(14, 7))
#     plt.title(f'{ticker} Cumulative Returns Comparison')
#     plt.plot(df.index, df['cum_market_returns'], label='Buy & Hold')
#     plt.plot(df.index, df['cum_strategy_returns'], label='Strategy (Before Costs)')
#     plt.plot(df.index, df['cum_strategy_returns_after_costs'], label='Strategy (After Costs)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Show the plots
# plot_price_with_indicators(df)
# plot_returns(df)

# # Print performance metrics
# print("\nPerformance Metrics:")
# print(f"Total Trading Days: {len(df) / 78:.1f}")  # Assuming 78 5-min bars per day
# print(f"Number of Trades: {df['trades'].sum() / 2:.0f}")  # Divide by 2 to count round-trip trades
# print(f"Buy & Hold Returns: {(df['cum_market_returns'].iloc[-1] - 1) * 100:.2f}%")
# print(f"Strategy Returns (Before Costs): {(df['cum_strategy_returns'].iloc[-1] - 1) * 100:.2f}%")
# print(f"Strategy Returns (After Costs): {(df['cum_strategy_returns_after_costs'].iloc[-1] - 1) * 100:.2f}%")

# # Risk metrics
# strategy_returns = df['strategy_returns'].dropna()
# market_returns = df['returns'].dropna()

# if len(strategy_returns) > 0 and strategy_returns.std() > 0:
#     print("\nRisk Metrics:")
#     # Annualizing for 5-min data (assuming 252 trading days, 78 5-min periods per day)
#     annualize_factor = np.sqrt(252 * 78)

#     print(f"Strategy Volatility: {strategy_returns.std() * annualize_factor * 100:.2f}%")
#     print(f"Market Volatility: {market_returns.std() * annualize_factor * 100:.2f}%")
#     print(f"Strategy Sharpe Ratio: {(strategy_returns.mean() / strategy_returns.std()) * annualize_factor:.2f}")
#     print(f"Market Sharpe Ratio: {(market_returns.mean() / market_returns.std()) * annualize_factor:.2f}")

#     # Calculate drawdowns
#     def calculate_drawdown(returns):
#         """Calculate maximum drawdown"""
#         cum_returns = (1 + returns).cumprod()
#         running_max = cum_returns.cummax()
#         drawdown = (cum_returns / running_max) - 1
#         return drawdown

#     dd_strategy = calculate_drawdown(df['strategy_returns'])
#     dd_market = calculate_drawdown(df['returns'])

#     print(f"Strategy Maximum Drawdown: {dd_strategy.min() * 100:.2f}%")
#     print(f"Market Maximum Drawdown: {dd_market.min() * 100:.2f}%")
# else:
#     print("\nNot enough data to calculate risk metrics.")




































# # Indian Stock Screener
# # This script analyzes multiple Indian stocks to identify top performers

# # Core Imports
# import pandas as pd
# import numpy as np
# import datetime
# import yfinance as yf
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # Configuration
# # List of top Indian stocks (NSE tickers - add '.NS' suffix for Yahoo Finance)
# indian_stocks = [
#     "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
#     "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
#     "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
#     "SUNPHARMA.NS", "TATAMOTORS.NS", "BAJFINANCE.NS", "TITAN.NS", "WIPRO.NS"
# ]

# # Analysis parameters
# end_date = datetime.date.today()
# start_date = end_date - pd.Timedelta(days=180)  # 6 months of data
# interval = "1d"  # Daily data for broader analysis

# # Technical indicator parameters
# sma_short = 20
# sma_long = 50
# ema_period = 20
# rsi_period = 14
# bb_period = 20

# # Download data
# def download_multiple_stocks(tickers, start, end, interval="1d"):
#     """Download market data for multiple tickers"""
#     print(f"Downloading data for {len(tickers)} stocks from {start} to {end}")

#     all_data = {}
#     for ticker in tqdm(tickers):
#         try:
#             df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
#             if len(df) > 0:
#                 all_data[ticker] = df
#             else:
#                 print(f"No data found for {ticker}")
#         except Exception as e:
#             print(f"Error downloading {ticker}: {e}")

#     print(f"Successfully downloaded data for {len(all_data)} stocks")
#     return all_data

# # Technical indicators
# def calculate_indicators(df):
#     """Calculate various technical indicators for analysis"""
#     # Create a copy to avoid modifying the original dataframe
#     df = df.copy()

#     # Moving Averages
#     df['sma_short'] = df['Close'].rolling(sma_short).mean()
#     df['sma_long'] = df['Close'].rolling(sma_long).mean()
#     df['ema'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

#     # Bollinger Bands
#     df['bb_middle'] = df['Close'].rolling(bb_period).mean()
#     df['bb_std'] = df['Close'].rolling(bb_period).std()
#     df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
#     df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

#     # RSI (Relative Strength Index)
#     delta = df['Close'].diff()
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)
#     avg_gain = gain.rolling(window=rsi_period).mean()
#     avg_loss = loss.rolling(window=rsi_period).mean()

#     # Calculate RS and RSI
#     rs = avg_gain / avg_loss
#     df['rsi'] = 100 - (100 / (1 + rs))

#     # Calculate returns
#     df['daily_return'] = df['Close'].pct_change()
#     df['cum_return'] = (1 + df['daily_return']).cumprod() - 1

#     # Calculate volatility (rolling 30-day)
#     df['volatility'] = df['daily_return'].rolling(30).std() * np.sqrt(252)  # Annualized

#     # Initialize golden cross and death cross columns
#     df['golden_cross'] = False
#     df['death_cross'] = False

#     # Need at least sma_long+1 data points to calculate crosses
#     if len(df) > sma_long + 1:
#         # Process golden crosses and death crosses - one row at a time to avoid Series truth value errors
#         for i in range(sma_long + 1, len(df)):
#             curr_short = df['sma_short'].iloc[i]
#             curr_long = df['sma_long'].iloc[i]
#             prev_short = df['sma_short'].iloc[i-1]
#             prev_long = df['sma_long'].iloc[i-1]

#             # Golden cross: short MA crosses above long MA
#             if (not pd.isna(curr_short) and not pd.isna(curr_long) and
#                 not pd.isna(prev_short) and not pd.isna(prev_long)):
#                 if curr_short > curr_long and prev_short <= prev_long:
#                     df.iloc[i, df.columns.get_loc('golden_cross')] = True
#                 # Death cross: short MA crosses below long MA
#                 elif curr_short < curr_long and prev_short >= prev_long:
#                     df.iloc[i, df.columns.get_loc('death_cross')] = True

#     return df

# # Performance metrics
# def calculate_performance_metrics(df):
#     """Calculate performance metrics for ranking stocks"""
#     metrics = {}

#     # Filter out NaN values
#     returns = df['daily_return'].dropna()

#     if len(returns) > 0:
#         # Return metrics
#         metrics['total_return'] = df['cum_return'].iloc[-1] if not pd.isna(df['cum_return'].iloc[-1]) else 0
#         metrics['annualized_return'] = ((1 + metrics['total_return']) ** (252 / len(returns)) - 1) if metrics['total_return'] > -1 else -1

#         # Risk metrics
#         metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized volatility
#         metrics['sharpe_ratio'] = (metrics['annualized_return'] / metrics['volatility']) if metrics['volatility'] > 0 else 0

#         # Trend metrics - using safer evaluation
#         try:
#             # Use explicit scalar comparison with .iloc[-1].item()
#             close_value = df['Close'].iloc[-1].item() if isinstance(df['Close'].iloc[-1], pd.Series) else df['Close'].iloc[-1]
#             sma_long_value = df['sma_long'].iloc[-1].item() if isinstance(df['sma_long'].iloc[-1], pd.Series) else df['sma_long'].iloc[-1]
#             sma_short_value = df['sma_short'].iloc[-1].item() if isinstance(df['sma_short'].iloc[-1], pd.Series) else df['sma_short'].iloc[-1]

#             # Now compare scalar values
#             metrics['above_sma50'] = close_value > sma_long_value if not pd.isna(sma_long_value) else False
#             metrics['above_sma20'] = close_value > sma_short_value if not pd.isna(sma_short_value) else False
#         except Exception:
#             # Fallback for any issues
#             metrics['above_sma50'] = False
#             metrics['above_sma20'] = False

#         # Momentum metrics
#         metrics['rsi'] = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50

#         # Recent performance (last 30 days)
#         if len(df) > 30:
#             recent_return = df['cum_return'].iloc[-1] - df['cum_return'].iloc[-30] if not pd.isna(df['cum_return'].iloc[-30]) else 0
#             metrics['recent_30d_return'] = recent_return if not pd.isna(recent_return) else 0
#         else:
#             metrics['recent_30d_return'] = 0

#         # Technical signals - ensure safe boolean evaluation
#         try:
#             golden_cross_recent = df['golden_cross'].iloc[-min(30, len(df)):]
#             death_cross_recent = df['death_cross'].iloc[-min(30, len(df)):]

#             # Use .any() or .sum() > 0 to safely evaluate Series
#             metrics['golden_cross_last_30d'] = golden_cross_recent.sum() > 0
#             metrics['death_cross_last_30d'] = death_cross_recent.sum() > 0
#         except Exception:
#             metrics['golden_cross_last_30d'] = False
#             metrics['death_cross_last_30d'] = False

#     else:
#         # Default values if not enough data
#         metrics = {
#             'total_return': 0,
#             'annualized_return': 0,
#             'volatility': 0,
#             'sharpe_ratio': 0,
#             'above_sma50': False,
#             'above_sma20': False,
#             'rsi': 50,
#             'recent_30d_return': 0,
#             'golden_cross_last_30d': False,
#             'death_cross_last_30d': False
#         }

#     return metrics

# # Screening function
# def screen_stocks(all_data):
#     """Screen stocks based on technical and fundamental criteria"""
#     results = []

#     for ticker, df in all_data.items():
#         # Skip if not enough data
#         if len(df) < max(sma_long, bb_period, rsi_period, 30):
#             print(f"Not enough data for {ticker}, skipping...")
#             continue

#         # Calculate indicators
#         df_with_indicators = calculate_indicators(df)

#         # Calculate performance metrics
#         metrics = calculate_performance_metrics(df_with_indicators)

#         # Add ticker to metrics
#         metrics['ticker'] = ticker
#         metrics['name'] = ticker.split('.')[0]
#         metrics['current_price'] = df_with_indicators['Close'].iloc[-1]
#         metrics['volume'] = df_with_indicators['Volume'].iloc[-1]

#         # Calculate average volume (30 days)
#         metrics['avg_volume_30d'] = df_with_indicators['Volume'].iloc[-min(30, len(df)):].mean() if len(df) > 0 else 0

#         results.append(metrics)

#     # Convert to DataFrame
#     results_df = pd.DataFrame(results)

#     return results_df

# # Create scoring system
# def score_stocks(results_df):
#     """Score stocks based on multiple performance metrics"""
#     if len(results_df) == 0:
#         return pd.DataFrame()

#     # Make a copy to avoid modifying the original dataframe
#     results_df = results_df.copy()

#     # Normalize and score each metric
#     # Returns (higher is better)
#     if 'total_return' in results_df.columns and not results_df['total_return'].isna().all():
#         max_return = results_df['total_return'].max()
#         min_return = results_df['total_return'].min()
#         range_return = max_return - min_return
#         if range_return > 0:
#             results_df['return_score'] = 40 * (results_df['total_return'] - min_return) / range_return
#         else:
#             results_df['return_score'] = 20
#     else:
#         results_df['return_score'] = 0

#     # Recent performance (higher is better)
#     if 'recent_30d_return' in results_df.columns and not results_df['recent_30d_return'].isna().all():
#         max_recent = results_df['recent_30d_return'].max()
#         min_recent = results_df['recent_30d_return'].min()
#         range_recent = max_recent - min_recent
#         if range_recent > 0:
#             results_df['recent_score'] = 20 * (results_df['recent_30d_return'] - min_recent) / range_recent
#         else:
#             results_df['recent_score'] = 10
#     else:
#         results_df['recent_score'] = 0

#     # Sharpe ratio (higher is better)
#     if 'sharpe_ratio' in results_df.columns and not results_df['sharpe_ratio'].isna().all():
#         max_sharpe = results_df['sharpe_ratio'].max()
#         min_sharpe = results_df['sharpe_ratio'].min()
#         range_sharpe = max_sharpe - min_sharpe
#         if range_sharpe > 0:
#             results_df['sharpe_score'] = 20 * (results_df['sharpe_ratio'] - min_sharpe) / range_sharpe
#         else:
#             results_df['sharpe_score'] = 10
#     else:
#         results_df['sharpe_score'] = 0

#     # Technical signals (binary scores) - using .loc to avoid SettingWithCopyWarning
#     results_df['technical_score'] = 0
#     if 'above_sma20' in results_df.columns:
#         mask_sma20 = results_df['above_sma20'].astype(bool)
#         results_df.loc[mask_sma20, 'technical_score'] += 5
#     if 'above_sma50' in results_df.columns:
#         mask_sma50 = results_df['above_sma50'].astype(bool)
#         results_df.loc[mask_sma50, 'technical_score'] += 5
#     if 'golden_cross_last_30d' in results_df.columns:
#         mask_golden = results_df['golden_cross_last_30d'].astype(bool)
#         results_df.loc[mask_golden, 'technical_score'] += 5
#     if 'death_cross_last_30d' in results_df.columns:
#         mask_death = results_df['death_cross_last_30d'].astype(bool)
#         results_df.loc[mask_death, 'technical_score'] -= 5

#     # RSI (penalize extreme values)
#     results_df['rsi_score'] = 0
#     if 'rsi' in results_df.columns:
#         mask_normal_rsi = (results_df['rsi'] >= 40) & (results_df['rsi'] <= 60)
#         results_df.loc[mask_normal_rsi, 'rsi_score'] = 5
#         mask_extreme_rsi = (results_df['rsi'] < 30) | (results_df['rsi'] > 70)
#         results_df.loc[mask_extreme_rsi, 'rsi_score'] = -5

#     # Calculate total score
#     results_df['total_score'] = (
#         results_df['return_score'] +
#         results_df['recent_score'] +
#         results_df['sharpe_score'] +
#         results_df['technical_score'] +
#         results_df['rsi_score']
#     )

#     # Sort by total score
#     results_df = results_df.sort_values('total_score', ascending=False).reset_index(drop=True)

#     return results_df

# # Visualization function
# def plot_top_stocks(all_data, top_tickers, n=3):
#     """Plot price charts with indicators for top N stocks"""
#     top_n = min(n, len(top_tickers))

#     for i in range(top_n):
#         ticker = top_tickers[i]

#         if ticker not in all_data:
#             print(f"Data for {ticker} not found")
#             continue

#         df = calculate_indicators(all_data[ticker])

#         plt.figure(figsize=(14, 10))
#         plt.suptitle(f"Technical Analysis for {ticker}", fontsize=16)

#         # Price and MAs
#         plt.subplot(3, 1, 1)
#         plt.plot(df.index, df['Close'], label='Close Price', linewidth=2)
#         plt.plot(df.index, df['sma_short'], label=f'SMA({sma_short})', linestyle='--')
#         plt.plot(df.index, df['sma_long'], label=f'SMA({sma_long})', linestyle='--')
#         plt.plot(df.index, df['bb_upper'], label='BB Upper', color='gray', alpha=0.3)
#         plt.plot(df.index, df['bb_lower'], label='BB Lower', color='gray', alpha=0.3)
#         plt.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='gray', alpha=0.1)

#         # Mark golden crosses and death crosses - safely find indices
#         golden_crosses_idx = df.index[df['golden_cross'] == True]
#         death_crosses_idx = df.index[df['death_cross'] == True]

#         # Check if any crosses exist before plotting
#         if len(golden_crosses_idx) > 0:
#             # Use safe plotting method
#             for idx in golden_crosses_idx:
#                 plt.scatter([idx], [df.loc[idx, 'Close']],
#                            marker='^', color='green', s=150)
#             # Add label just once for the legend
#             plt.scatter([], [], marker='^', color='green', s=150, label='Golden Cross')

#         if len(death_crosses_idx) > 0:
#             # Use safe plotting method
#             for idx in death_crosses_idx:
#                 plt.scatter([idx], [df.loc[idx, 'Close']],
#                            marker='v', color='red', s=150)
#             # Add label just once for the legend
#             plt.scatter([], [], marker='v', color='red', s=150, label='Death Cross')

#         plt.title("Price with Moving Averages and Bollinger Bands")
#         plt.ylabel("Price (₹)")
#         plt.grid(True)
#         plt.legend()

#         # RSI
#         plt.subplot(3, 1, 2)
#         plt.plot(df.index, df['rsi'], label='RSI', color='purple')
#         plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
#         plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
#         plt.title("Relative Strength Index (RSI)")
#         plt.ylabel("RSI")
#         plt.grid(True)
#         plt.legend()

#         # Cumulative returns
#         plt.subplot(3, 1, 3)
#         plt.plot(df.index, df['cum_return'] * 100, label='Cumulative Return (%)', color='blue')
#         plt.title("Cumulative Return")
#         plt.ylabel("Return (%)")
#         plt.grid(True)
#         plt.legend()

#         plt.tight_layout()
#         plt.subplots_adjust(top=0.9)
#         plt.show()

# # Main function
# def main():
#     print("Indian Stock Screener")
#     print("=====================")
#     print(f"Analyzing {len(indian_stocks)} Indian stocks from {start_date} to {end_date}")

#     # Download data for all stocks
#     all_data = download_multiple_stocks(indian_stocks, start_date, end_date, interval)

#     # Screen stocks and calculate scores
#     results = screen_stocks(all_data)
#     scored_results = score_stocks(results)

#     if len(scored_results) == 0:
#         print("No valid stock data found for analysis")
#         return

#     # Display top 10 stocks (or fewer if less available)
#     num_to_display = min(10, len(scored_results))
#     print(f"\nTop {num_to_display} Stocks:")

#     # Format the display table
#     display_columns = [
#         'ticker', 'name', 'current_price',
#         'total_return', 'recent_30d_return', 'sharpe_ratio',
#         'rsi', 'above_sma50', 'above_sma20', 'total_score'
#     ]

#     # Make sure we only select columns that actually exist
#     valid_columns = [col for col in display_columns if col in scored_results.columns]
#     display_df = scored_results[valid_columns].head(num_to_display)

#     # Format percentages and round numbers
#     if 'total_return' in display_df.columns:
#         display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x*100:.2f}%")
#     if 'recent_30d_return' in display_df.columns:
#         display_df['recent_30d_return'] = display_df['recent_30d_return'].apply(lambda x: f"{x*100:.2f}%")
#     if 'sharpe_ratio' in display_df.columns:
#         display_df['sharpe_ratio'] = display_df['sharpe_ratio'].round(2)
#     if 'rsi' in display_df.columns:
#         display_df['rsi'] = display_df['rsi'].round(1)
#     if 'total_score' in display_df.columns:
#         display_df['total_score'] = display_df['total_score'].round(1)

#     # Rename columns for better readability
#     column_mapping = {
#         'ticker': 'Ticker',
#         'name': 'Name',
#         'current_price': 'Price (₹)',
#         'total_return': 'Total Return',
#         'recent_30d_return': '30d Return',
#         'sharpe_ratio': 'Sharpe',
#         'rsi': 'RSI',
#         'above_sma50': '> SMA50',
#         'above_sma20': '> SMA20',
#         'total_score': 'Score'
#     }

#     # Only rename columns that exist
#     valid_mapping = {k: v for k, v in column_mapping.items() if k in display_df.columns}
#     display_df = display_df.rename(columns=valid_mapping)

#     print(display_df.to_string(index=False))

#     # Plot top 3 stocks
#     print("\nGenerating charts for top 3 stocks...")
#     plot_top_stocks(all_data, scored_results['ticker'].head(3).tolist(), 3)

#     print("\nAnalysis complete!")

# if __name__ == "__main__":
#     main()