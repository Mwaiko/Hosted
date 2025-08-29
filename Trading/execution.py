from Models.Model import Model
import requests
import numpy as np
import pandas as pd
import time
import json
import logging
from datetime import datetime, timedelta
from data.SentimentFetcher import SentimentFetcher
import threading
from typing import List, Dict, Any, Optional, Tuple
import os
from collections import deque

class AccuracyMonitor:
    """Monitor and track prediction accuracy over time with percent gain tracking"""
    
    def __init__(self, lookforward_periods: int = 1):
        self.lookforward_periods = lookforward_periods
        self.predictions = deque(maxlen=1000)
        self.actual_outcomes = deque(maxlen=1000)
        self.prices = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        
        self.percent_changes = deque(maxlen=1000)  
        self.trading_gains = deque(maxlen=1000)    
        self.cumulative_gain = 0.0                 
        
        self.total_predictions = 0
        self.correct_predictions = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.interval_gains = deque(maxlen=1000)  
        self.last_interval_gain = 0.0         
    def add_prediction(self, prediction: int, probability: float, price: float, timestamp: float):
        self.predictions.append({
            'prediction': prediction,
            'probability': probability,
            'price': price,
            'timestamp': timestamp,
            'verified': False
        })
        
    def verify_predictions(self, current_price: float, current_timestamp: float):
        """Verify predictions and calculate percent changes and trading gains"""
        verified_count = 0
        total_interval_gain = 0.0  # NEW: Track gain for this verification cycle
        
        for pred_data in self.predictions:
            if not pred_data['verified']:
                time_diff = current_timestamp - pred_data['timestamp']
                if time_diff >= self.lookforward_periods * 60:
                    
                    # Calculate actual price change percentage
                    price_change_pct = (current_price - pred_data['price']) / pred_data['price']
                    actual_outcome = 1 if price_change_pct > 0.001 else 0
                    
                    # Store the actual percent change
                    self.percent_changes.append(price_change_pct * 100)  # Convert to percentage
                    
                    # Calculate trading gain based on signal correctness
                    trading_gain = self._calculate_trading_gain(
                        pred_data['prediction'], 
                        price_change_pct
                    )
                    self.trading_gains.append(trading_gain)
                    self.cumulative_gain += trading_gain
                    total_interval_gain += trading_gain  # NEW: Add to interval total
                    
                    # ... existing accuracy tracking code ...
                    self.total_predictions += 1
                    predicted = pred_data['prediction']
                    
                    if predicted == actual_outcome:
                        self.correct_predictions += 1
                    
                    if predicted == 1 and actual_outcome == 1:
                        self.true_positives += 1
                    elif predicted == 1 and actual_outcome == 0:
                        self.false_positives += 1
                    elif predicted == 0 and actual_outcome == 1:
                        self.false_negatives += 1
                    else:
                        self.true_negatives += 1
                    
                    pred_data['verified'] = True
                    pred_data['actual_price_change'] = price_change_pct * 100
                    pred_data['trading_gain'] = trading_gain
                    
                    self.actual_outcomes.append(actual_outcome)
                    verified_count += 1
        
        # NEW: Store interval gain if any predictions were verified
        if verified_count > 0:
            self.interval_gains.append(total_interval_gain)
            self.last_interval_gain = total_interval_gain
        
        return verified_count
    def _calculate_trading_gain(self, prediction: int, price_change_pct: float) -> float:
        """
        Calculate trading gain based on prediction and actual price movement
        
        Logic:
        - BUY signal (prediction=1): gain = price_change_pct (we benefit if price goes up)
        - SELL signal (prediction=0): gain = -price_change_pct (we benefit if price goes down)
        
        This simulates the gain/loss from following the signal
        """
        if prediction == 1: 
            return price_change_pct * 100  
        else:  
            return -price_change_pct * 100  
    def reset_metrics(self):
        self.predictions.clear()
        self.actual_outcomes.clear()
        self.prices.clear()
        self.timestamps.clear()
        self.percent_changes.clear()
        self.trading_gains.clear()
        self.interval_gains.clear() 
        
        self.cumulative_gain = 0.0
        self.last_interval_gain = 0.0  # NEW: Add this line
        self.total_predictions = 0
        self.correct_predictions = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
    def get_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate and return accuracy metrics including percent gains"""
        if self.total_predictions == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'total_predictions': 0,
                'average_percent_change': 0.0,
                'net_percent_gain': 0.0,
                'average_trading_gain': 0.0,
                'win_rate': 0.0,
                'total_positive_gains': 0,
                'total_negative_gains': 0
            }
        
        accuracy = self.correct_predictions / self.total_predictions
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0.0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        average_percent_change = np.mean(self.percent_changes) if self.percent_changes else 0.0
        average_trading_gain = np.mean(self.trading_gains) if self.trading_gains else 0.0
        
        positive_gains = [gain for gain in self.trading_gains if gain > 0]
        negative_gains = [gain for gain in self.trading_gains if gain <= 0]
        win_rate = len(positive_gains) / len(self.trading_gains) if self.trading_gains else 0.0
        
        return {
            # ... existing metrics ...
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_predictions': self.total_predictions,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'average_percent_change': average_percent_change,
            'net_percent_gain': self.cumulative_gain,
            'average_trading_gain': average_trading_gain,
            'win_rate': win_rate,
            'total_positive_gains': len(positive_gains),
            'total_negative_gains': len(negative_gains),
            'best_gain': max(self.trading_gains) if self.trading_gains else 0.0,
            'worst_loss': min(self.trading_gains) if self.trading_gains else 0.0,
            # NEW: Interval-specific metrics
            'last_interval_gain': self.last_interval_gain,
            'total_intervals': len(self.interval_gains)
        }
    
    def get_recent_performance(self, last_n: int = 10) -> Dict[str, float]:
        """Get performance metrics for the last N predictions"""
        if not self.trading_gains or len(self.trading_gains) < last_n:
            recent_gains = list(self.trading_gains)
        else:
            recent_gains = list(self.trading_gains)[-last_n:]
        
        if not recent_gains:
            return {
                'recent_average_gain': 0.0,
                'recent_win_rate': 0.0,
                'recent_net_gain': 0.0,
                'predictions_analyzed': 0
            }
        
        recent_positive = [g for g in recent_gains if g > 0]
        recent_win_rate = len(recent_positive) / len(recent_gains)
        recent_net_gain = sum(recent_gains)
        recent_average_gain = np.mean(recent_gains)
        
        return {
            'recent_average_gain': recent_average_gain,
            'recent_win_rate': recent_win_rate,
            'recent_net_gain': recent_net_gain,
            'predictions_analyzed': len(recent_gains)
        }
    
    def get_detailed_history(self, last_n: int = None) -> List[Dict]:
        """Get detailed history of predictions with their outcomes"""
        verified_predictions = [pred for pred in self.predictions if pred['verified']]
        
        if last_n:
            verified_predictions = verified_predictions[-last_n:]
        
        return [{
            'timestamp': pred['timestamp'],
            'prediction': 'BUY' if pred['prediction'] == 1 else 'SELL',
            'probability': pred['probability'],
            'entry_price': pred['price'],
            'price_change_pct': pred.get('actual_price_change', 0),
            'trading_gain': pred.get('trading_gain', 0),
            'correct': pred.get('trading_gain', 0) > 0
        } for pred in verified_predictions]
    
    def reset_metrics(self):
        """Reset all tracking metrics"""
        self.predictions.clear()
        self.actual_outcomes.clear()
        self.prices.clear()
        self.timestamps.clear()
        self.percent_changes.clear()
        self.trading_gains.clear()
        
        self.cumulative_gain = 0.0
        self.total_predictions = 0
        self.correct_predictions = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
    
class TradingMonitor:
    def __init__(self, coin: str, log_file: str = None):
        print("Initializing Monitor Class")
        self.coin = coin
        self.log_file = log_file or f"{coin}_trading.log"
        self.setup_logging()
        self.metrics = {
            'total_predictions': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'errors': 0,
            'last_prediction_time': None,
            'session_start': datetime.now()
        }
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_prediction(self, prediction: int, probability: float, price: float, accuracy_metrics: Dict = None):
        signal = "BUY" if prediction == 1 else "SELL"
        self.metrics['total_predictions'] += 1
        self.metrics['last_prediction_time'] = datetime.now()
        
        if prediction == 1:
            self.metrics['buy_signals'] += 1
        else:
            self.metrics['sell_signals'] += 1
            
        log_msg = f"PREDICTION: {signal} | Probability: {probability:.4f} | Price: {price:.4f}"
        if accuracy_metrics and accuracy_metrics['total_predictions'] > 0:
            log_msg += f" | Accuracy: {accuracy_metrics['accuracy']:.3f}"
            log_msg += f" | Net Gain: {accuracy_metrics['net_percent_gain']:.2f}%"
            log_msg += f" | Win Rate: {accuracy_metrics['win_rate']:.1%}"
        
        self.logger.info(log_msg)
        
    def log_error(self, error: Exception, context: str = ""):
        self.metrics['errors'] += 1
        self.logger.error(f"ERROR in {context}: {str(error)}")
        
    def print_session_stats(self, accuracy_metrics: Dict = None):
        runtime = datetime.now() - self.metrics['session_start']
        print("\n" + "="*70)
        print("SESSION STATISTICS")
        print("="*70)
        print(f"Coin: {self.coin}")
        print(f"Runtime: {runtime}")
        print(f"Total Predictions: {self.metrics['total_predictions']}")
        print(f"Buy Signals: {self.metrics['buy_signals']}")
        print(f"Sell Signals: {self.metrics['sell_signals']}")
        print(f"Errors: {self.metrics['errors']}")
        
        if accuracy_metrics and accuracy_metrics['total_predictions'] > 0:
            print("\nACCURACY METRICS:")
            print(f"Overall Accuracy: {accuracy_metrics['accuracy']:.3f}")
            print(f"Precision: {accuracy_metrics['precision']:.3f}")
            print(f"Recall: {accuracy_metrics['recall']:.3f}")
            print(f"F1 Score: {accuracy_metrics['f1_score']:.3f}")
            print(f"Verified Predictions: {accuracy_metrics['total_predictions']}")
            
            print("\nTRADING PERFORMANCE:")
            print(f"🎯 Net Percent Gain: {accuracy_metrics['net_percent_gain']:.2f}%")
            print(f"📈 Average Trading Gain: {accuracy_metrics['average_trading_gain']:.3f}%")
            print(f"🏆 Win Rate: {accuracy_metrics['win_rate']:.1%}")
            print(f"⬆️  Best Single Gain: {accuracy_metrics['best_gain']:.2f}%")
            print(f"⬇️  Worst Single Loss: {accuracy_metrics['worst_loss']:.2f}%")
            print(f"✅ Winning Trades: {accuracy_metrics['total_positive_gains']}")
            print(f"❌ Losing Trades: {accuracy_metrics['total_negative_gains']}")
            
            # Color-coded performance summary
            if accuracy_metrics['net_percent_gain'] > 0:
                print(f"💚 PROFITABLE SESSION: +{accuracy_metrics['net_percent_gain']:.2f}% total gain")
            else:
                print(f"🔴 LOSING SESSION: {accuracy_metrics['net_percent_gain']:.2f}% total loss")
        
        if self.metrics['last_prediction_time']:
            print(f"Last Prediction: {self.metrics['last_prediction_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
    def print_recent_performance(self, accuracy_monitor, last_n: int = 10):
        """Print recent performance summary"""
        recent_performance = accuracy_monitor.get_recent_performance(last_n)
        
        if recent_performance['predictions_analyzed'] > 0:
            print(f"\n📊 RECENT PERFORMANCE (Last {recent_performance['predictions_analyzed']} trades):")
            print(f"Recent Net Gain: {recent_performance['recent_net_gain']:.2f}%")
            print(f"Recent Avg Gain: {recent_performance['recent_average_gain']:.3f}%")
            print(f"Recent Win Rate: {recent_performance['recent_win_rate']:.1%}")
            
            if recent_performance['recent_net_gain'] > 0:
                print("🔥 Recent performance is POSITIVE!")
            else:
                print("📉 Recent performance needs improvement")
        
    def print_detailed_trades(self, accuracy_monitor, last_n: int = 5):
        """Print detailed information about recent trades"""
        history = accuracy_monitor.get_detailed_history(last_n)
        
        if history:
            print(f"\n📋 LAST {len(history)} TRADE DETAILS:")
            print("-" * 80)
            for i, trade in enumerate(history, 1):
                status = "✅ WIN" if trade['correct'] else "❌ LOSS"
                signal_emoji = "🟢" if trade['prediction'] == 'BUY' else "🔴"
                
                print(f"{i}. {signal_emoji} {trade['prediction']} | "
                      f"Price: ${trade['entry_price']:.4f} | "
                      f"Change: {trade['price_change_pct']:+.2f}% | "
                      f"Gain: {trade['trading_gain']:+.2f}% | "
                      f"{status}")
            print("-" * 80)
        
    def log_trading_signal_with_context(self, prediction: int, probability: float, 
                                      price: float, accuracy_metrics: Dict = None, 
                                      accuracy_monitor=None):
        """Enhanced logging with trading context"""
        signal = "BUY" if prediction == 1 else "SELL"
        signal_emoji = "🟢" if prediction == 1 else "🔴"
        
        self.metrics['total_predictions'] += 1
        self.metrics['last_prediction_time'] = datetime.now()
        
        if prediction == 1:
            self.metrics['buy_signals'] += 1
        else:
            self.metrics['sell_signals'] += 1
        
        # Build comprehensive log message
        log_msg = f"{signal_emoji} SIGNAL: {signal} | "
        log_msg += f"Confidence: {probability:.1%} | "
        log_msg += f"Price: ${price:.4f}"
        
        if accuracy_metrics and accuracy_metrics['total_predictions'] > 0:
            log_msg += f" | Accuracy: {accuracy_metrics['accuracy']:.1%}"
            log_msg += f" | Net: {accuracy_metrics['net_percent_gain']:+.1f}%"
            log_msg += f" | WR: {accuracy_metrics['win_rate']:.0%}"
        
        
        
        # Print recent performance if available
        if accuracy_monitor and accuracy_metrics['total_predictions'] >= 5:
            recent = accuracy_monitor.get_recent_performance(5)
            if recent['predictions_analyzed'] > 0:
                trend_emoji = "📈" if recent['recent_net_gain'] > 0 else "📉"
                print(f"{trend_emoji} Recent 5 trades: {recent['recent_net_gain']:+.1f}% "
                      f"(WR: {recent['recent_win_rate']:.0%})")
    def print_interval_summary(self, accuracy_metrics: Dict, cycle_count: int):
        """Print interval-specific summary with gains"""
        print(f"\n{'='*60}")
        print(f"🔄 INTERVAL #{cycle_count} SUMMARY")
        print(f"{'='*60}")
        
        # Basic info
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🪙 Coin: {self.coin}")
        
        if accuracy_metrics and accuracy_metrics['total_predictions'] > 0:
            # Interval-specific gain
            interval_gain = accuracy_metrics.get('last_interval_gain', 0.0)
            interval_gain_emoji = "📈" if interval_gain > 0 else "📉" if interval_gain < 0 else "➡️"
            
            print(f"\n📊 INTERVAL PERFORMANCE:")
            print(f"{interval_gain_emoji} This Interval Gain: {interval_gain:+.3f}%")
            
            # Overall performance
            print(f"\n💰 CUMULATIVE PERFORMANCE:")
            print(f"🎯 Net Percent Gain: {accuracy_metrics['net_percent_gain']:+.2f}%")
            print(f"📈 Average Trading Gain: {accuracy_metrics['average_trading_gain']:+.3f}%")
            print(f"🏆 Win Rate: {accuracy_metrics['win_rate']:.1%}")
            print(f"✅ Total Verified Predictions: {accuracy_metrics['total_predictions']}")
            print(f"📋 Completed Intervals: {accuracy_metrics.get('total_intervals', 0)}")
            
            # Performance indicators
            net_gain = accuracy_metrics['net_percent_gain']
            if net_gain > 5:
                print(f"🔥 EXCELLENT SESSION: +{net_gain:.1f}% total!")
            elif net_gain > 0:
                print(f"💚 PROFITABLE SESSION: +{net_gain:.1f}%")
            elif net_gain > -5:
                print(f"⚠️  MINOR LOSS: {net_gain:.1f}%")
            else:
                print(f"🔴 SIGNIFICANT LOSS: {net_gain:.1f}%")
        else:
            print("\n⏳ No verified predictions yet...")
        
        print(f"{'='*60}\n")
class IntervalManager:
    INTERVAL_SECONDS = {
        '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
        '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, 
        '8h': 28800, '12h': 43200, '1d': 86400
    }
    
    def __init__(self, interval: str):
        self.interval = interval
        self.interval_seconds = self.INTERVAL_SECONDS.get(interval)
        if not self.interval_seconds:
            raise ValueError(f"Unsupported interval: {interval}")
        
        self.last_processed_timestamp = 0
        self.last_candle_open_time = None  # Track the last candle's open time
            
    def get_next_interval_time_from_candle(self, candle_open_time: float) -> float:
        """
        Calculate the next interval time based on the candle's open time
        
        Args:
            candle_open_time: The open time of the current candle (in seconds)
            
        Returns:
            The timestamp when the next candle should be available
        """
        # Convert to milliseconds if it's already in seconds
        if candle_open_time < 1e12:  # If less than year 2001 in milliseconds, it's in seconds
            candle_open_ms = candle_open_time * 1000
        else:
            candle_open_ms = candle_open_time
            
        # Calculate the next candle's open time
        next_candle_open_ms = candle_open_ms + (self.interval_seconds * 1000)
        
        # Return as seconds
        return next_candle_open_ms / 1000
    
    def get_next_interval_time(self, current_time: float = None) -> float:
        """
        Legacy method - calculate next interval based on current time
        Keep for backward compatibility
        """
        if current_time is None:
            current_time = time.time()
        return current_time + (self.interval_seconds - (current_time % self.interval_seconds))
        
    def time_until_next_candle(self, candle_open_time: float, current_time: float = None) -> float:
        """
        Calculate how long to wait until the next candle should be available
        
        Args:
            candle_open_time: The open time of the current candle
            current_time: Current system time (optional, defaults to now)
            
        Returns:
            Seconds to wait until next candle
        """
        if current_time is None:
            current_time = time.time()
            
        next_candle_time = self.get_next_interval_time_from_candle(candle_open_time)
        wait_time = next_candle_time - current_time
        
        # Ensure we wait at least a small buffer (e.g., 5 seconds) to account for API delays
        return max(wait_time + 5, 5)
    
    def time_until_next_interval(self, current_time: float = None) -> float:
        """Legacy method for backward compatibility"""
        if current_time is None:
            current_time = time.time()
        return self.get_next_interval_time(current_time) - current_time

    def should_process_candle(self, candle_timestamp: int) -> bool:
        """Check if this candle should be processed (avoid duplicates)"""
        if candle_timestamp > self.last_processed_timestamp:
            self.last_processed_timestamp = candle_timestamp
            self.last_candle_open_time = candle_timestamp
            return True
        return False
    
    def get_expected_next_candle_time(self) -> Optional[float]:
        """
        Get the expected time when the next candle should be available
        based on the last processed candle
        """
        if self.last_candle_open_time is None:
            return None
        return self.get_next_interval_time_from_candle(self.last_candle_open_time)
class FeatureProcessor:
    def __init__(self, coin: str, monitor: TradingMonitor = None):
        self.coin = coin
        self.monitor = monitor
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("<<<Received DataFrame >>>")
            print(f"DataFrame shape: {df.shape}")
            print(df.head())
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'number_of_trades',
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df[::-1].reset_index(drop=True)
            min_required = 60  # Need at least 60 periods for longer moving averages
            if len(df) < min_required:
                raise ValueError(f"Insufficient data for feature engineering. Need at least {min_required} periods, got {len(df)}")
            
            print(f"Processing {len(df)} candles for feature engineering...")

            # Moving Averages
            df['sma_20'] = df['close'].rolling(window=20, min_periods=20).mean()
            df['sma_50'] = df['close'].rolling(window=50, min_periods=50).mean()
            df['ema_12'] = df['close'].ewm(span=12, adjust=False, min_periods=12).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False, min_periods=26).mean()

            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['signal_line'] = df['macd'].ewm(span=9, adjust=False, min_periods=9).mean()
            df['macd_hist'] = df['macd'] - df['signal_line']
            
            # RSI
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            
            # Avoid division by zero
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Stochastic Oscillator
            low_min = df['low'].rolling(window=14, min_periods=14).min()
            high_max = df['high'].rolling(window=14, min_periods=14).max()
            
            # Avoid division by zero in stochastic calculation
            denominator = high_max - low_min
            denominator = denominator.replace(0, np.finfo(float).eps)
            df['stoch_k'] = 100 * (df['close'] - low_min) / denominator
            df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=3).mean()
            
            # Bollinger Bands
            df['bb_ma'] = df['close'].rolling(window=20, min_periods=20).mean()
            bb_std = df['close'].rolling(window=20, min_periods=20).std()
            df['bb_upper'] = df['bb_ma'] + 2 * bb_std
            df['bb_lower'] = df['bb_ma'] - 2 * bb_std
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=14, min_periods=14).mean()
            
            # Price features
            df['pct_change'] = df['close'].pct_change()
            
            # Handle log return calculation safely
            price_ratio = df['close'] / df['close'].shift(1)
            price_ratio = price_ratio.replace([0, np.inf, -np.inf], np.nan)
            df['log_return'] = np.log(price_ratio)
            
            df['price_accel'] = df['close'].diff().diff()
            
            # Avoid division by zero in ratios
            df['high_low_ratio'] = df['high'] / df['low'].replace(0, np.finfo(float).eps)
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # VWAP calculation with cumulative operations
            cum_vol = df['volume'].cumsum()
            cum_vol = cum_vol.replace(0, np.finfo(float).eps)  # Avoid division by zero
            df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / cum_vol
            
            # Volume features
            df['buy_volume_ratio'] = df['taker_buy_base_asset_volume'] / df['volume'].replace(0, np.finfo(float).eps)
            df['vol_ma_short'] = df['volume'].rolling(window=5, min_periods=5).mean()
            df['vol_ma_long'] = df['volume'].rolling(window=20, min_periods=20).mean()
            
            # Volume oscillator with safe division
            vol_ma_long_safe = df['vol_ma_long'].replace(0, np.finfo(float).eps)
            df['vol_osc'] = (df['vol_ma_short'] - df['vol_ma_long']) / vol_ma_long_safe
            
            # Pattern recognition - Hammer pattern
            df['is_hammer'] = (
                (df['close'] > df['open']) & 
                ((df['close'] - df['low']) > 2 * (df['open'] - df['close'])) &
                (df['high'] - df['close']) < (df['open'] - df['close']) / 2
            ).astype(int)
            
            # Statistical features
            rolling_mean = df['close'].rolling(window=20, min_periods=20).mean()
            rolling_std = df['close'].rolling(window=20, min_periods=20).std()
            rolling_std = rolling_std.replace(0, np.finfo(float).eps)  # Avoid division by zero
            df['z_score'] = (df['close'] - rolling_mean) / rolling_std
            
            df['quantile_25'] = df['close'].rolling(window=50, min_periods=50).quantile(0.25)
            df['quantile_75'] = df['close'].rolling(window=50, min_periods=50).quantile(0.75)
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            # Composite features
            df['trend_volume_conf'] = np.where(
                (df['close'] > df['sma_20']) & (df['volume'] > df['vol_ma_short']),
                1, 0
            )
            df['rsi_macd_bull'] = np.where(
                (df['rsi'] > 50) & (df['macd'] > df['signal_line']),
                1, 0
            )
            
            # Drop rows with NaN values
            initial_length = len(df)
            df = df.dropna()
            final_length = len(df)
            
            if final_length == 0:
                raise ValueError("All rows were dropped due to NaN values. Check your data quality.")
            
            print(f"Dropped {initial_length - final_length} rows with NaN values")
            print(f"Final DataFrame shape: {df.shape}")
            print('<< After Feature Engineering >>> ')
            print('<<< HEAD >>>')
            print(df.head())
            print('<<< TAIL >>>')
            print(df.tail(5))

            return df
            
        except Exception as e:
            if self.monitor:
                self.monitor.log_error(e, "feature_engineering")
            else:
                print(f"Error in feature engineering: {str(e)}")
            raise

class Trader:
    def __init__(self, coin: str, interval: str, batches: int, sequences: int, 
                 features: List[str] = None, feature_data: List[List] = None):
        self.coin = coin.upper()
        self.interval = interval
        self.model = Model(coin, interval, batches, sequences)
        print("Initialized Model Class")
        self.features = features   
        
        self.monitor = TradingMonitor(coin)
        print("Initialized Trading Monitor Class")
        self.interval_manager = IntervalManager(interval)
        print("Initialized Interval Manager Class")
        self.accuracy_monitor = AccuracyMonitor(lookforward_periods=1)
        print("Initialized Accuracy Monitor Class")
        self.feature_processor = FeatureProcessor(coin, self.monitor)  # Pass monitor here
        print("Initialized Feature Processor Class")
        
        self.historical_data = feature_data
        
        self.is_running = False
        self.model_trained = False
        
        self.default_features = [
            'macd', 'signal_line', 'macd_hist', 'rsi', 'stoch_k', 'stoch_d',
            'pct_change', 'log_return', 'buy_volume_ratio', 'vol_ma_short', 
            'vol_ma_long', 'is_hammer', 'z_score'
        ]
        
        self.monitor.logger.info(f"Trader initialized for {coin} on {interval} intervals")
        
    def fetch_current_ohlcv(self, limit: int = 2) -> List[List]:
    
        symbol = f"{self.coin}USDT"
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": self.interval,
            "limit": limit
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    raise ValueError("Empty response from API")
                return data
                
            except requests.exceptions.RequestException as e:
                self.monitor.log_error(e, f"API request attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    

    
    def normalize_timestamps(self,df, col="timestamp", fmt="%Y-%m-%d %H:%M:%S"):
        def convert(x):
            try:
                # Case 1: If it's an int or float → assume timestamp in ms
                if isinstance(x, (int, float)):
                    return pd.to_datetime(x, unit="ms").strftime(fmt)

                # Case 2: If it's already datetime
                if isinstance(x, pd.Timestamp):
                    return x.strftime(fmt)

                # Case 3: If it's a string → try parsing
                return pd.to_datetime(x, errors="coerce").strftime(fmt)
            except Exception:
                return None  # fallback if unparseable

        df[col] = df[col].apply(convert)
        return df
    def process_new_candle(self, candle_data: List) -> Optional[List[float]]:
        try:
            print(f"Processing new candle data: {candle_data}...")  # Show first 5 elements
            
            # Validate input data
            if not candle_data or len(candle_data) < 11:
                raise ValueError(f"Invalid candle data length: expected at least 11 elements, got {len(candle_data) if candle_data else 0}")
            
            # Define column names for the candle data
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'close_time', 'quote_asset_volume', 'number_of_trades',
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            
            if not self.historical_data or len(self.historical_data) < 100:
                raise ValueError(f"Insufficient historical data: need at least 100 candles, have {len(self.historical_data) if self.historical_data else 0}")
            
            # Insert new candle at the beginning and remove the oldest
            self.historical_data.insert(0, candle_data)
            self.historical_data.pop()
            
            print(f"Historical data size: {len(self.historical_data)}")
            
    
            data_for_features = self.historical_data[:100]
            
            # Create DataFrame
            df = pd.DataFrame(data_for_features, columns=columns)
            df = self.normalize_timestamps(df)
            print(f"Created DataFrame with shape: {df.shape}")
            
            
            df_features = self.feature_processor.engineer_features(df)
            
            # Validate features list
            if len(self.features) == 0:
                print("No features specified, using default features")
                self.features = self.default_features
            
            missing_features = [f for f in self.features if f not in df_features.columns]
            if missing_features:
                print(f"Warning: Missing features {missing_features}")
                # Use only available features
                available_features = [f for f in self.features if f in df_features.columns]
                if not available_features:
                    raise ValueError("No valid features available")
                self.features = available_features
                print(f"Using available features: {self.features}")
            
            feature_values = df_features[self.features].iloc[-1].values.tolist()
            
            # Validate feature values
            if any(pd.isna(feature_values)):
                nan_indices = [i for i, x in enumerate(feature_values) if pd.isna(x)]
                nan_features = [self.features[i] for i in nan_indices]
                raise ValueError(f"NaN values found in features: {nan_features}")
            
            print(f"Extracted {len(feature_values)} feature values: {feature_values[:5]}..." if len(feature_values) > 5 else f"Feature values: {feature_values}")
            self.monitor.logger.debug(f"Using features: {self.features}")
            
            return feature_values
            
        except Exception as e:
            error_msg = f"Error in process_new_candle: {str(e)}"
            print(error_msg)
            if self.monitor:
                self.monitor.log_error(e, "process_new_candle")
            return None 
    def train_model(self):
        if self.model_trained == False:
            self.monitor.logger.info("Starting model training...")
            
            if len(self.historical_data) < 100:
                self.monitor.logger.warning("Insufficient historical data for training")
                return False
            
            self.model.Train()
            self.model_trained = True
            self.monitor.logger.info("Model training completed successfully")
            return True
            
    def make_prediction(self, features: List[float], candle_data: List) -> Dict[str, Any]:
        
        if True:
            if not self.model_trained:
                raise ValueError("Model not trained yet")
            pred_value,prob_value,binary_prediction = self.model.Predict(features)
            pred_value = binary_prediction
            current_price = float(candle_data[4])
            current_timestamp = float(candle_data[0]) / 1000
            
            self.accuracy_monitor.add_prediction(pred_value, prob_value, current_price, current_timestamp)
            verified_count = self.accuracy_monitor.verify_predictions(current_price, current_timestamp)
            accuracy_metrics = self.accuracy_monitor.get_accuracy_metrics()
            
            
            return {
                'prediction': pred_value,
                'probability': prob_value,
                'price': current_price,
                'timestamp': current_timestamp,
                'accuracy_metrics': accuracy_metrics,
                'verified_predictions': verified_count
            }
            
        
    def run_realtime_trading(self):
        """Updated real-time trading with proper candle-based timing"""
        if True:
            self.monitor.logger.info("="*60)
            self.monitor.logger.info("STARTING REAL-TIME TRADING SYSTEM")
            self.monitor.logger.info("="*60)
            
            # Train model first
            if not self.model_trained:
                self.monitor.logger.info("Training model...")
                if not self.train_model():
                    self.monitor.logger.error("Model training failed, cannot continue")
                    return
        
            self.is_running = True
            cycle_count = 0
            last_processed_candle_time = None
            
            # Get the first candle to establish timing
            self.monitor.logger.info("Fetching initial candle to establish timing...")
            initial_candle = self.fetch_current_ohlcv(limit=1)
            if initial_candle:
                initial_candle_data = [float(i) for i in initial_candle[0]]
                initial_open_time = initial_candle_data[0] / 1000  # Convert to seconds
                last_processed_candle_time = initial_candle_data[0]  # Keep in milliseconds
                
                # Calculate wait time based on candle open time
                wait_time = self.interval_manager.time_until_next_candle(initial_open_time)
                next_candle_time = self.interval_manager.get_next_interval_time_from_candle(initial_open_time)
                
                self.monitor.logger.info(f"Current candle opens at: {datetime.fromtimestamp(initial_open_time).strftime('%Y-%m-%d %H:%M:%S')}")
                self.monitor.logger.info(f"Next candle expected at: {datetime.fromtimestamp(next_candle_time).strftime('%Y-%m-%d %H:%M:%S')}")
                self.monitor.logger.info(f"Waiting {wait_time:.1f} seconds for next {self.interval} candle...")
                
                time.sleep(wait_time)
            
            while self.is_running:
                cycle_count += 1
                print(f"\n{'='*25} CYCLE {cycle_count} {'='*25}")
                
                # Execute trading cycle
                success, new_candle_processed = self.trading_cycle()
                
                if success:
                    self.monitor.logger.info(f"Cycle {cycle_count} completed successfully")
                    
                    # Only print interval summary if we actually processed a new candle
                    if new_candle_processed:
                        accuracy_metrics = self.accuracy_monitor.get_accuracy_metrics()
                        self.monitor.print_interval_summary(accuracy_metrics, cycle_count)
                        
                        # Update last processed candle time
                        if self.interval_manager.last_candle_open_time is not None:
                            last_processed_candle_time = self.interval_manager.last_candle_open_time
                    
                else:
                    self.monitor.logger.warning(f"Cycle {cycle_count} completed with issues")
                
                # Print session stats every 5 cycles (only if we have new data)
                if cycle_count % 5 == 0 and new_candle_processed:
                    accuracy_metrics = self.accuracy_monitor.get_accuracy_metrics()
                    self.monitor.print_session_stats(accuracy_metrics)
                
                # Calculate wait time for next candle
                current_time = time.time()
                
                if last_processed_candle_time is not None:
                    last_candle_open_seconds = last_processed_candle_time / 1000  # Convert to seconds
                    wait_time = self.interval_manager.time_until_next_candle(last_candle_open_seconds, current_time)
                    next_candle_time = self.interval_manager.get_next_interval_time_from_candle(last_candle_open_seconds)
                    
                    # Ensure we wait at least until the next interval
                    if wait_time <= 0:
                        # If wait_time is negative or zero, calculate the next interval after the current time
                        wait_time = self.interval_manager.time_until_next_interval(current_time)
                        next_candle_time = self.interval_manager.get_next_interval_time(current_time)
                    
                    if wait_time > 0:
                        next_time_str = datetime.fromtimestamp(next_candle_time).strftime('%H:%M:%S')
                        self.monitor.logger.info(f"Next candle expected at {next_time_str} (waiting {wait_time:.1f}s)")
                        
                        # Add a small buffer to ensure the new candle is available
                        actual_wait = wait_time + 2  # 2 second buffer
                        time.sleep(actual_wait)
                    else:
                        self.monitor.logger.warning("Next candle should already be available, waiting 30 seconds")
                        time.sleep(30)  # Wait 30 seconds if timing is off
                else:
                    # Fallback to time-based interval if no candle data available
                    wait_time = self.interval_manager.time_until_next_interval()
                    self.monitor.logger.info(f"Using time-based interval, waiting {wait_time:.1f}s")
                    time.sleep(wait_time + 2)  # Add buffer

    def trading_cycle(self):
        """Modified trading cycle to return whether a new candle was processed"""
        try:
            # Fetch current candle data
            candle_data_list = self.fetch_current_ohlcv(limit=2)
            if not candle_data_list or len(candle_data_list) == 0:
                self.monitor.logger.warning("Failed to fetch candle data")
                return False, False
            
            # Extract candle data and timestamp
            raw_candle_data = candle_data_list[0]
            candle_timestamp = int(candle_data_list[1][0])  # This is in milliseconds
            candle_open_time = candle_timestamp / 1000  # Convert to seconds for datetime operations
            
            print(f"Fetched Candle Data: Open Time = {datetime.fromtimestamp(candle_open_time).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Check if we should process this candle (avoid duplicates)
            if not self.interval_manager.should_process_candle(candle_timestamp):
                self.monitor.logger.info(f"Candle already processed, waiting for next interval...")
                return True, False  # Success but no new candle processed
            
            self.monitor.logger.info(f"Processing new candle: {datetime.fromtimestamp(candle_open_time).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Process candle through feature engineering
            features = self.process_new_candle(raw_candle_data)
            if features is None:
                self.monitor.logger.warning("Failed to process candle features")
                return False, False
            
            # Make prediction
            result = self.make_prediction(features, raw_candle_data)
            
            if result['prediction'] is not None:
                pred_value = result['prediction']
                prob_value = result['probability']
                
                signal = "BUY" if pred_value == 1 else "SELL"
                confidence = prob_value if pred_value == 1 else (1 - prob_value)
                
                print(f"\nTRADING SIGNAL: {signal}")
                print(f"Confidence: {confidence:.1%}")
                print(f"Current Price: ${result['price']:.4f}")
                print(f"Candle Open Time: {datetime.fromtimestamp(candle_open_time).strftime('%H:%M:%S')}")
                print(f"System Time: {datetime.now().strftime('%H:%M:%S')}")
                
                # Show when next candle is expected
                next_candle_time = self.interval_manager.get_next_interval_time_from_candle(candle_timestamp)
                print(f"Next Candle Expected: {datetime.fromtimestamp(next_candle_time).strftime('%H:%M:%S')}")
                
                # Show interval gain if available
                if result['verified_predictions'] > 0:
                    accuracy_metrics = result['accuracy_metrics']
                    interval_gain = accuracy_metrics.get('last_interval_gain', 0.0)
                    interval_emoji = "📈" if interval_gain > 0 else "📉" if interval_gain < 0 else "➡️"
                    print(f"[{interval_emoji}] Interval Gain: {interval_gain:+.3f}%")
                
                # Display accuracy metrics if available
                if result['accuracy_metrics']['total_predictions'] > 0:
                    metrics = result['accuracy_metrics']
                    print(f"Accuracy: {metrics['accuracy']:.1%} | Net Gain: {metrics['net_percent_gain']:+.2f}% | Verified: {metrics['total_predictions']}")
                
                if result['verified_predictions'] > 0:
                    print(f"Verified {result['verified_predictions']} past predictions")
                
                # Log the prediction with enhanced context
                self.monitor.log_trading_signal_with_context(
                    pred_value, prob_value, result['price'], 
                    result['accuracy_metrics'], self.accuracy_monitor
                )
                
                return True, True  # Success and new candle processed
            else:
                self.monitor.logger.warning("Prediction failed")
                return False, False
                
        except Exception as e:
            self.monitor.log_error(e, "trading_cycle")
            return False, False
    def stop_trading(self):
        """Gracefully stop the trading system"""
        self.is_running = False
        accuracy_metrics = self.accuracy_monitor.get_accuracy_metrics()
        self.monitor.print_session_stats(accuracy_metrics)
        self.monitor.logger.info("Trading system stopped")

    def run_backtest_mode(self, test_cycles: int = 10):
        """Run in backtest mode for testing"""
        self.monitor.logger.info(f"Running backtest mode for {test_cycles} cycles")
        
        if not self.model_trained:
            if not self.train_model():
                self.monitor.logger.error("Model training failed")
                return
        
        for i in range(test_cycles):
            print(f"\n--- Backtest Cycle {i+1}/{test_cycles} ---")
            self.trading_cycle()
            time.sleep(5)  # Short delay for demonstration
        
        accuracy_metrics = self.accuracy_monitor.get_accuracy_metrics()
        self.monitor.print_session_stats(accuracy_metrics)
