import numpy as np
import pandas as pd
import json

class FeatureEngineering:
    def __init__(self, Coin):
        self.Coin = Coin
        self.Data = []
        self.SelectedTopFeatures = []
        self.historical_data = []
        
    def Load_data(self):
        with open(f"{self.Coin}.json", 'r') as f:
            self.Data = json.load(f)
            self.historical_data = self.Data[:150]
        return self.Data
    
    def EngineerData(self):
        Data = self.Load_data()
        df = pd.DataFrame(Data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                        'close_time', 'quote_asset_volume', 'number_of_trades', 
                                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        print(df.head())
        df = df[::-1].reset_index(drop=True)

        # Technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal_line']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic oscillator
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Bollinger Bands
        df['bb_ma'] = df['close'].rolling(20).mean()
        df['bb_upper'] = df['bb_ma'] + 2 * df['close'].rolling(20).std()
        df['bb_lower'] = df['bb_ma'] - 2 * df['close'].rolling(20).std()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Price features
        df['pct_change'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['price_accel'] = df['close'].diff().diff()
        df['high_low_ratio'] = df['high'] / df['low']
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Volume features
        df['buy_volume_ratio'] = df['taker_buy_base_asset_volume'] / df['volume']
        df['vol_ma_short'] = df['volume'].rolling(5).mean()
        df['vol_ma_long'] = df['volume'].rolling(20).mean()
        df['vol_osc'] = (df['vol_ma_short'] - df['vol_ma_long']) / df['vol_ma_long']
        
        # Pattern recognition
        df['is_hammer'] = (
            (df['close'] > df['open']) & 
            ((df['close'] - df['low']) > 2 * (df['open'] - df['close'])) &
            (df['high'] - df['close']) < (df['open'] - df['close']) / 2
        ).astype(int)
        
        # Time features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Statistical features
        df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        df['quantile_25'] = df['close'].rolling(50).quantile(0.25)
        df['quantile_75'] = df['close'].rolling(50).quantile(0.75)
        
        # Composite features
        df['trend_volume_conf'] = np.where(
            (df['close'] > df['sma_20']) & (df['volume'] > df['vol_ma_short']),
            1, 0
        )
        df['rsi_macd_bull'] = np.where(
            (df['rsi'] > 50) & (df['macd'] > df['signal_line']),
            1, 0
        )

        # Remove NaN values
        df = df.dropna()

        # Create target (price goes up next period)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        print('<< After Reshaping The DataFrame >>> ')
        print(' <<< HEAD >>>')
        print(df.head())
        print('<<< TAIL >>>')
        print(df.tail(5))
        return df
    
    def SelectTopFeatures(self):
        df = self.EngineerData()
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Exclude non-feature columns
        X = df.drop(columns=['target', 'timestamp', 'ignore', 'close_time'])
        y = df['target']
        
        selector = SelectKBest(score_func=f_classif, k=20)
        selector.fit_transform(X, y) 
        selected_features = X.columns[selector.get_support()]
        print("Top 20 features:", selected_features.tolist())
        
        return selected_features
    
    def SelectedModelFeatures(self):
        Features = self.SelectTopFeatures()
        self.SelectedTopFeatures = Features
    

        df = self.EngineerData()
        print("The Selected Features are ->.",Features)
    
        Feature_df = df[Features].copy()
        Feature_df['values'] = df['close'].copy() 
        print(f"Features selected: {len(Features)} features")
        print(f"Feature_df shape: {Feature_df.shape}")
        print("<<< Selected Features DataFrame >>>")
        print(Feature_df.head())
        print(f"Final columns: {list(Feature_df.columns)}")
        
        Data = Feature_df.values.tolist()
        
        with open("Model_Data.json", 'w') as f:
            json.dump(Data, f)
        
        print(f"Saved {len(Data)} samples with {len(Data[0]) if Data else 0} features each")
        return Data
    
    def get_top_features(self):
        return self.SelectedTopFeatures
    
    def get_data(self):
        result = [list(d.values()) for d in self.historical_data]
        return result
    
    def run(self):
        self.SelectTopFeatures()
        self.SelectedModelFeatures()

if __name__ == "__main__":
    Fe = FeatureEngineering("BTC")
    Fe.run()