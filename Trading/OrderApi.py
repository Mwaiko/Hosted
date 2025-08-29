import hashlib
import hmac
import time
import requests
import json
from typing import Dict, Optional, Union
from enum import Enum

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

class PositionSide(Enum):
    BOTH = "BOTH"
    LONG = "LONG"
    SHORT = "SHORT"

class OrderAPI:
    """
    Binance Futures Trading API Class for executing buy/sell signals
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize the OrderAPI
        
        Args:
            api_key (str): Binance API Key
            api_secret (str): Binance API Secret
            testnet (bool): Use testnet if True, mainnet if False
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
        self.headers = {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
    def _generate_signature(self, params: str) -> str:
        """Generate HMAC SHA256 signature"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = True) -> Dict:
        """
        Make HTTP request to Binance API
        
        Args:
            method (str): HTTP method (GET, POST, DELETE)
            endpoint (str): API endpoint
            params (Dict): Request parameters
            signed (bool): Whether request requires signature
            
        Returns:
            Dict: API response
        """
        if params is None:
            params = {}
            
        url = f"{self.base_url}{endpoint}"
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = self._generate_signature(query_string)
            params['signature'] = signature
        
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, params=params)
            elif method == "DELETE":
                response = requests.delete(url, headers=self.headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            raise
    
    def get_account_info(self) -> Dict:
        """Get futures account information"""
        return self._make_request("GET", "/fapi/v2/account")
    
    def get_position_info(self, symbol: str = None) -> Dict:
        """
        Get position information
        
        Args:
            symbol (str): Trading symbol (optional, gets all if None)
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request("GET", "/fapi/v2/positionRisk", params)
    
    def get_balance(self) -> Dict:
        """Get futures account balance"""
        return self._make_request("GET", "/fapi/v2/balance")
    
    def place_market_order(self, symbol: str, side: OrderSide, quantity: float, 
                          position_side: PositionSide = PositionSide.BOTH) -> Dict:
        """
        Place a market order (buy/sell signal execution)
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            side (OrderSide): BUY or SELL
            quantity (float): Order quantity
            position_side (PositionSide): Position side for hedge mode
            
        Returns:
            Dict: Order response
        """
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': OrderType.MARKET.value,
            'quantity': quantity,
            'positionSide': position_side.value
        }
        
        return self._make_request("POST", "/fapi/v1/order", params)
    
    def place_limit_order(self, symbol: str, side: OrderSide, quantity: float, 
                         price: float, position_side: PositionSide = PositionSide.BOTH,
                         time_in_force: str = "GTC") -> Dict:
        """
        Place a limit order
        
        Args:
            symbol (str): Trading symbol
            side (OrderSide): BUY or SELL
            quantity (float): Order quantity
            price (float): Order price
            position_side (PositionSide): Position side
            time_in_force (str): Time in force (GTC, IOC, FOK)
        """
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': OrderType.LIMIT.value,
            'quantity': quantity,
            'price': price,
            'timeInForce': time_in_force,
            'positionSide': position_side.value
        }
        
        return self._make_request("POST", "/fapi/v1/order", params)
    
    def place_stop_loss_order(self, symbol: str, side: OrderSide, quantity: float,
                             stop_price: float, position_side: PositionSide = PositionSide.BOTH) -> Dict:
        """
        Place a stop loss order
        
        Args:
            symbol (str): Trading symbol
            side (OrderSide): BUY or SELL
            quantity (float): Order quantity
            stop_price (float): Stop price
            position_side (PositionSide): Position side
        """
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': OrderType.STOP_MARKET.value,
            'quantity': quantity,
            'stopPrice': stop_price,
            'positionSide': position_side.value
        }
        
        return self._make_request("POST", "/fapi/v1/order", params)
    
    def place_take_profit_order(self, symbol: str, side: OrderSide, quantity: float,
                               stop_price: float, position_side: PositionSide = PositionSide.BOTH) -> Dict:
        """
        Place a take profit order
        
        Args:
            symbol (str): Trading symbol
            side (OrderSide): BUY or SELL
            quantity (float): Order quantity
            stop_price (float): Stop price for take profit
            position_side (PositionSide): Position side
        """
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': OrderType.TAKE_PROFIT_MARKET.value,
            'quantity': quantity,
            'stopPrice': stop_price,
            'positionSide': position_side.value
        }
        
        return self._make_request("POST", "/fapi/v1/order", params)
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """
        Cancel an order
        
        Args:
            symbol (str): Trading symbol
            order_id (int): Order ID to cancel
        """
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        return self._make_request("DELETE", "/fapi/v1/order", params)
    
    def cancel_all_orders(self, symbol: str) -> Dict:
        """
        Cancel all open orders for a symbol
        
        Args:
            symbol (str): Trading symbol
        """
        params = {'symbol': symbol}
        return self._make_request("DELETE", "/fapi/v1/allOpenOrders", params)
    
    def get_open_orders(self, symbol: str = None) -> Dict:
        """
        Get open orders
        
        Args:
            symbol (str): Trading symbol (optional)
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request("GET", "/fapi/v1/openOrders", params)
    
    def close_position(self, symbol: str, position_side: PositionSide = PositionSide.BOTH) -> Dict:
        """
        Close an open position by placing opposite market order
        
        Args:
            symbol (str): Trading symbol
            position_side (PositionSide): Position side to close
        """
        # Get current position
        positions = self.get_position_info(symbol)
        
        for position in positions:
            if position['symbol'] == symbol and position['positionSide'] == position_side.value:
                position_amt = float(position['positionAmt'])
                
                if position_amt == 0:
                    return {"message": "No open position to close"}
                
                # Determine the side to close position
                close_side = OrderSide.SELL if position_amt > 0 else OrderSide.BUY
                close_quantity = abs(position_amt)
                
                return self.place_market_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=close_quantity,
                    position_side=position_side
                )
        
        return {"message": "Position not found"}
    
    def send_buy_signal(self, symbol: str, quantity: float, order_type: str = "MARKET",
                       price: float = None, stop_loss: float = None, 
                       take_profit: float = None) -> Dict:
        """
        Execute a BUY signal
        
        Args:
            symbol (str): Trading symbol
            quantity (float): Order quantity
            order_type (str): Order type (MARKET or LIMIT)
            price (float): Limit price (required for LIMIT orders)
            stop_loss (float): Stop loss price (optional)
            take_profit (float): Take profit price (optional)
        """
        try:
            # Place main buy order
            if order_type.upper() == "MARKET":
                order_response = self.place_market_order(symbol, OrderSide.BUY, quantity)
            elif order_type.upper() == "LIMIT":
                if price is None:
                    raise ValueError("Price is required for LIMIT orders")
                order_response = self.place_limit_order(symbol, OrderSide.BUY, quantity, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # Place stop loss if specified
            if stop_loss:
                self.place_stop_loss_order(symbol, OrderSide.SELL, quantity, stop_loss)
            
            # Place take profit if specified
            if take_profit:
                self.place_take_profit_order(symbol, OrderSide.SELL, quantity, take_profit)
            
            return order_response
            
        except Exception as e:
            print(f"Error executing BUY signal: {e}")
            return {"error": str(e)}
    
    def send_sell_signal(self, symbol: str, quantity: float, order_type: str = "MARKET",
                        price: float = None, stop_loss: float = None,
                        take_profit: float = None) -> Dict:
        """
        Execute a SELL signal
        
        Args:
            symbol (str): Trading symbol
            quantity (float): Order quantity
            order_type (str): Order type (MARKET or LIMIT)
            price (float): Limit price (required for LIMIT orders)
            stop_loss (float): Stop loss price (optional)
            take_profit (float): Take profit price (optional)
        """
        try:
            # Place main sell order
            if order_type.upper() == "MARKET":
                order_response = self.place_market_order(symbol, OrderSide.SELL, quantity)
            elif order_type.upper() == "LIMIT":
                if price is None:
                    raise ValueError("Price is required for LIMIT orders")
                order_response = self.place_limit_order(symbol, OrderSide.SELL, quantity, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # Place stop loss if specified
            if stop_loss:
                self.place_stop_loss_order(symbol, OrderSide.BUY, quantity, stop_loss)
            
            # Place take profit if specified
            if take_profit:
                self.place_take_profit_order(symbol, OrderSide.BUY, quantity, take_profit)
            
            return order_response
            
        except Exception as e:
            print(f"Error executing SELL signal: {e}")
            return {"error": str(e)}

