import flask
import requests
import time
from datetime import datetime
def fetch_current_ohlcv( limit: int = 2):
        coin = "BTC"
        interval = "5m"
        symbol = f"{coin}USDT"
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                fmt = "%Y-%m-%d %H:%M:%S"
                if not data:
                    raise ValueError("Empty response from API")
                data[0][0],data[1][0] = datetime.fromtimestamp(data[0][0]/1000),datetime.fromtimestamp(data[1][0]/1000)
                data[0][0],data[1][0] = data[0][0].strftime(fmt),data[1][0].strftime(fmt)
                print(data)
                return data
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise


fetch_current_ohlcv()