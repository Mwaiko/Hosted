from flask import Flask, request, jsonify
from datetime import datetime
from typing import List, Dict, Any
import json

class Backend:
    def __init__(self):
        self.app = Flask(__name__)
        self.trading_signals = []  # Store all trading signals
        self.current_positions = {}  # Track current positions by coin
        self.setup_routes()
    
    def setup_routes(self):
        
        @self.app.route('/signal', methods=['POST'])
        def receive_signal():
            try:
                data = request.get_json()
                signal = data.get('signal')
                price = data.get('price')
                coin = data.get('coin')
                regime = data.get('regime')
                
                result = self.process_trading_signal(signal, price, coin, regime)
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 400
        
        @self.app.route('/signals', methods=['GET'])
        def get_signals():
            return jsonify(self.trading_signals)
        
        @self.app.route('/positions', methods=['GET'])
        def get_positions():
            return jsonify(self.current_positions)
    
    def process_trading_signal(self, signal: str, price: float, coin: str, regime: str) -> Dict[str, Any]:
        """
        Process a trading signal for a given coin
        
        Args:
            signal (str): Trading signal - either "buy" or "sell"
            price (float): Current price of the coin
            coin (str): The cryptocurrency symbol (e.g., "BTC", "ETH")
            regime (str): Market regime indicator
            
        Returns:
            Dict: Response containing signal processing result
        """
        timestamp = datetime.now().isoformat()
        
        # Validate signal
        if signal.lower() not in ['buy', 'sell']:
            return {"error": "Invalid signal. Must be 'buy' or 'sell'"}
        
        # Create signal record
        signal_record = {
            "timestamp": timestamp,
            "signal": signal.lower(),
            "price": price,
            "coin": coin.upper(),
            "regime": regime,
            "processed": True
        }
        
        # Process the signal logic
        result = self._execute_signal_logic(signal_record)
        
        # Store the signal
        self.trading_signals.append(signal_record)
        
        # Update current positions
        self._update_positions(signal_record)
        
        return result
    
    def _execute_signal_logic(self, signal_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the trading signal logic"""
        signal = signal_record['signal']
        coin = signal_record['coin']
        price = signal_record['price']
        regime = signal_record['regime']
        
        if signal == 'buy':
            return {
                "action": "BUY_EXECUTED",
                "coin": coin,
                "price": price,
                "regime": regime,
                "message": f"Buy signal processed for {coin} at ${price} in {regime} regime",
                "timestamp": signal_record['timestamp']
            }
        elif signal == 'sell':
            return {
                "action": "SELL_EXECUTED",
                "coin": coin,
                "price": price,
                "regime": regime,
                "message": f"Sell signal processed for {coin} at ${price} in {regime} regime",
                "timestamp": signal_record['timestamp']
            }
    
    def _update_positions(self, signal_record: Dict[str, Any]):
        """Update current positions based on the signal"""
        coin = signal_record['coin']
        signal = signal_record['signal']
        price = signal_record['price']
        
        if coin not in self.current_positions:
            self.current_positions[coin] = {
                "position": "none",
                "entry_price": None,
                "last_signal": None,
                "last_update": None
            }
        
        self.current_positions[coin].update({
            "position": "long" if signal == "buy" else "short" if signal == "sell" else "none",
            "entry_price": price,
            "last_signal": signal,
            "last_update": signal_record['timestamp']
        })
    
    def get_signal_history(self, coin: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Get signal history, optionally filtered by coin and limited"""
        signals = self.trading_signals
        
        if coin:
            signals = [s for s in signals if s['coin'] == coin.upper()]
        
        if limit:
            signals = signals[-limit:]
        
        return signals
    
    def get_current_position(self, coin: str) -> Dict[str, Any]:
        """Get current position for a specific coin"""
        return self.current_positions.get(coin.upper(), {
            "position": "none",
            "entry_price": None,
            "last_signal": None,
            "last_update": None
        })
    
    def clear_history(self):
        """Clear all signal history"""
        self.trading_signals.clear()
        self.current_positions.clear()
    
    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Run the Flask application"""
        self.app.run(host=host, port=port, debug=debug)

# Example usage:
if __name__ == "__main__":
    # Initialize backend
    backend = Backend()
    
    # Example of processing signals programmatically
    result1 = backend.process_trading_signal("buy", 45000.50, "BTC", "bull_market")
    print(f"Signal 1 result: {result1}")
    
    result2 = backend.process_trading_signal("sell", 3200.75, "ETH", "bear_market")
    print(f"Signal 2 result: {result2}")
    
    # Print current positions
    print(f"Current positions: {backend.current_positions}")
    
    # Run the Flask server (uncomment to start server)
    # backend.run()