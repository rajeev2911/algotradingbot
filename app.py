import threading
from flask import Flask, jsonify, render_template, request
import pandas as pd
import os
import sys

# Import functions from main.py
from main import main, get_key_stocks_today, get_historical_data

# Create Flask app at the module level
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.secret_key = os.urandom(24)

# Global variable for thread-safe caching
_analysis_results = None
_analysis_lock = threading.Lock()

# Helper function to convert DataFrame to JSON-safe format
def convert_to_json_safe(df):
    if df is None or df.empty:
        return []
    
    # First convert to records
    records = df.head(10).to_dict(orient='records')
    
    # Then ensure all values are JSON serializable
    for record in records:
        # Add 'rsi' field for frontend compatibility if 'rsi_short' exists but 'rsi' doesn't
        if 'rsi_short' in record and 'rsi' not in record:
            record['rsi'] = record['rsi_short']
            
        # Add 'return_30d' field for COMEX compatibility if 'recent_30d_return' exists
        if 'recent_30d_return' in record and 'return_30d' not in record:
            record['return_30d'] = record['recent_30d_return'] * 100 if record['recent_30d_return'] else 0
            
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

# Thread-safe function to get analysis results
def get_analysis_results():
    global _analysis_results
    with _analysis_lock:
        if _analysis_results is None:
            _analysis_results = main()
    return _analysis_results

# Routes defined AFTER the app is created
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/admin')
def admin_panel():
    return render_template('admin.html')

@app.route('/api/top-stocks')
def api_top_stocks():
    results = get_analysis_results()
    return jsonify(convert_to_json_safe(results['stock_scored']))

@app.route('/api/top-forex')
def api_top_forex():
    results = get_analysis_results()
    return jsonify(convert_to_json_safe(results['forex_scored']))

@app.route('/api/top-comex')
def api_top_comex():
    results = get_analysis_results()
    return jsonify(convert_to_json_safe(results['comex_scored']))

@app.route('/api/key-stocks')
def api_key_stocks():
    return jsonify(get_key_stocks_today())

@app.route('/api/key-futures')
def api_key_futures():
    results = get_analysis_results()
    return jsonify(results['options_today'])

@app.route('/api/history')
def api_history():
    days = request.args.get('days', default=7, type=int)
    return jsonify(get_historical_data(days))

@app.route('/api/stock-details/<ticker>')
def api_stock_details(ticker):
    results = get_analysis_results()
    
    ticker_with_suffix = f"{ticker}.NS" if not ticker.endswith('.NS') else ticker
    
    if ticker_with_suffix in results['stock_data']:
        df = results['stock_data'][ticker_with_suffix]
        # Convert DataFrame to dict for JSON serialization
        data = {
            'ticker': ticker_with_suffix,
            'name': ticker.replace('.NS', ''),
            'data': df.reset_index().to_dict(orient='records'),
            'last_price': float(df['Close'].iloc[-1]),
            'change_percent': float(df['Close'].pct_change().iloc[-1] * 100),
            'indicators': {
                'rsi': float(df['rsi'].iloc[-1]),
                'macd': float(df['macd'].iloc[-1]),
                'sma_20': float(df['sma_short'].iloc[-1]),
                'sma_50': float(df['sma_long'].iloc[-1])
            }
        }
        return jsonify(data)
    else:
        return jsonify({'error': f'Ticker {ticker} not found'}), 404

@app.route('/api/comex-details/<ticker>')
def api_comex_details(ticker):
    results = get_analysis_results()
    
    if ticker in results['comex_data']:
        df = results['comex_data'][ticker]
        # Convert DataFrame to dict for JSON serialization
        data = {
            'ticker': ticker,
            'name': ticker,  # Use a more descriptive name if available
            'data': df.reset_index().to_dict(orient='records'),
            'last_price': float(df['Close'].iloc[-1]),
            'change_percent': float(df['Close'].pct_change().iloc[-1] * 100),
            'indicators': {
                'rsi': float(df['rsi'].iloc[-1]),
                'macd': float(df['macd'].iloc[-1]),
                'sma_20': float(df['sma_short'].iloc[-1]),
                'sma_50': float(df['sma_long'].iloc[-1])
            }
        }
        return jsonify(data)
    else:
        return jsonify({'error': f'COMEX commodity {ticker} not found'}), 404

# Run the app only when executed directly
if __name__ == "__main__":
    # Ensure data directories exist
    os.makedirs('data/history', exist_ok=True)
    
    # Pre-load analysis results
    get_analysis_results()
    
    # Start the Flask app
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)