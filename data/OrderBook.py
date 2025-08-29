import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
class OrderBookService:
    def __init__(self,Interval,Coin,days):
        self.Coin = Coin
        self.Interval = Interval
        Symbol = f"{Coin}USDT"
        self.save_price_data(Symbol,Interval,days=days)
    def get_historical_klines(self,symbol, interval, start_time, end_time=None):
        

        base_url = "https://api.binance.com/api/v3/klines"
        
        start_ms = int(start_time.timestamp() * 1000)
        if end_time:
            end_ms = int(end_time.timestamp() * 1000)
        else:
            end_ms = int(datetime.now().timestamp() * 1000)
        
        all_klines = []
        
        print(f"Fetching data from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(end_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}")
        
        request_count = 0
        while start_ms < end_ms:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ms,
                'endTime': end_ms,
                'limit': 1000  
            }
            
            try:
                request_count += 1
                response = requests.get(base_url, params=params)
                response.raise_for_status() 
                
                data = response.json()
                
                if not data or len(data) == 0:
                    print("Received empty data from API. Ending data collection.")
                    break
                    
                print(f"Request #{request_count}: Retrieved {len(data)} candles. Start: {datetime.fromtimestamp(data[0][0]/1000).strftime('%Y-%m-%d %H:%M')}")
                
                all_klines.extend(data)
                
                start_ms = data[-1][0] + 1
                
                time.sleep(0.3)
                
            except requests.exceptions.RequestException as e:
                print(f"Error making request: {e}")
                time.sleep(5)  # Wait longer on error
                continue
            except (IndexError, KeyError) as e:
                print(f"Error processing data: {e}")
                break
        
        print(f"Total data points collected: {len(all_klines)}")
        
        # If no data collected, exit
        if not all_klines:
            print("No data was collected. Exiting.")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert numeric columns to float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                        'quote_asset_volume', 'taker_buy_base_asset_volume', 
                        'taker_buy_quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        return df

        
    def save_price_data(self,symbol,step, days):

        interval = f"{step}m"  
        base_symbol = symbol.replace('USDT', '')
        
        start_time = datetime.now() - timedelta(days=days)
        
        print(f"Fetching {days} days of {interval} data for {symbol}...")
        df = self.get_historical_klines(symbol, interval, start_time)
        
        if df is None or df.empty:
            print("No data available to save.")
            return
        
        # Display information about the retrieved data
        print(f"Retrieved {len(df)} candlesticks")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Create a dataframe with only the timestamp and price data
        price_df = df#[['timestamp', 'open', 'close', 'volume', 'high', 'low']]
        
        # Sort the dataframe by timestamp in descending order (newest first)
        price_df = price_df.sort_values(by='timestamp', ascending=False)
        price_df = price_df.iloc[1:].reset_index(drop=True)

        # Convert timestamp to string format for JSON serialization
        price_df['timestamp'] = price_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to JSON file named after the base cryptocurrency (e.g., "BTC.json")
        json_filename = f"{base_symbol}.json"
        
        records = price_df.to_dict(orient='records')
        
        # Save to JSON with proper formatting and precision
        with open(json_filename, 'w') as f:
            json.dump(records, f, indent=2)
        
        print(f"Price data saved to {json_filename} (newest data first)")
        
        # Display sample of the data
        print("\nSample of Price Data (15-minute intervals):")
        print(price_df.head())
        
        return price_df

if __name__ == "__main__":
    OrderBookData = OrderBookService(5,"BTC",1)