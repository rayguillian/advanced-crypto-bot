from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta
import pandas as pd
import logging
from typing import Dict, List, Optional, Union
import json
import time

class ExchangeHandler:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = False):
        """Initialize the exchange handler"""
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.logger = logging.getLogger(__name__)
        self._initialize_exchange_info()
        
    def _initialize_exchange_info(self):
        """Initialize exchange information and trading rules"""
        try:
            info = self.client.get_exchange_info()
            self.symbols_info = {}
            
            for symbol_data in info['symbols']:
                if symbol_data['status'] == 'TRADING':
                    self.symbols_info[symbol_data['symbol']] = {
                        'baseAsset': symbol_data['baseAsset'],
                        'quoteAsset': symbol_data['quoteAsset'],
                        'filters': {f['filterType']: f for f in symbol_data['filters']},
                        'orderTypes': symbol_data['orderTypes'],
                        'permissions': symbol_data['permissions']
                    }
            
            self.logger.info(f"Initialized {len(self.symbols_info)} trading pairs")
        except Exception as e:
            self.logger.error(f"Error initializing exchange info: {e}")
            raise
    
    def get_historical_data(self, symbol: str, interval: str,
                          start_time: Union[str, datetime],
                          end_time: Optional[Union[str, datetime]] = None,
                          limit: int = 1000) -> pd.DataFrame:
        """Fetch historical klines/candlestick data"""
        try:
            if isinstance(start_time, datetime):
                start_time = int(start_time.timestamp() * 1000)
            if isinstance(end_time, datetime):
                end_time = int(end_time.timestamp() * 1000)
            
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=str(start_time),
                end_str=str(end_time) if end_time else None,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            
            return df.set_index('timestamp')
            
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise
    
    def place_order(self, symbol: str, side: str, order_type: str,
                   quantity: Optional[float] = None, price: Optional[float] = None,
                   stop_price: Optional[float] = None, time_in_force: str = 'GTC',
                   quote_quantity: Optional[float] = None,
                   client_order_id: Optional[str] = None) -> Dict:
        """Place a trading order"""
        try:
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': order_type.upper(),
            }
            
            if quantity:
                params['quantity'] = self._format_quantity(symbol, quantity)
            if quote_quantity and order_type.upper() == 'MARKET':
                params['quoteOrderQty'] = self._format_price(symbol, quote_quantity)
            if price:
                params['price'] = self._format_price(symbol, price)
            if stop_price:
                params['stopPrice'] = self._format_price(symbol, stop_price)
            if order_type.upper() != 'MARKET':
                params['timeInForce'] = time_in_force
            if client_order_id:
                params['newClientOrderId'] = client_order_id
            
            order = self.client.create_order(**params)
            self.logger.info(f"Order placed successfully: {json.dumps(order, indent=2)}")
            return order
            
        except BinanceAPIException as e:
            self.logger.error(f"Error placing order for {symbol}: {e}")
            raise
    
    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity according to symbol's precision requirements"""
        step_size = float(self.symbols_info[symbol]['filters']['LOT_SIZE']['stepSize'])
        precision = len(str(step_size).rstrip('0').split('.')[-1])
        return f"{quantity:.{precision}f}"
    
    def _format_price(self, symbol: str, price: float) -> str:
        """Format price according to symbol's precision requirements"""
        tick_size = float(self.symbols_info[symbol]['filters']['PRICE_FILTER']['tickSize'])
        precision = len(str(tick_size).rstrip('0').split('.')[-1])
        return f"{price:.{precision}f}"
    
    def get_account_balance(self, asset: Optional[str] = None) -> Union[Dict, float]:
        """Get account balance for specific asset or all assets"""
        try:
            if asset:
                balance = self.client.get_asset_balance(asset=asset)
                return float(balance['free'])
            else:
                account = self.client.get_account()
                return {
                    b['asset']: {
                        'free': float(b['free']),
                        'locked': float(b['locked'])
                    }
                    for b in account['balances']
                    if float(b['free']) > 0 or float(b['locked']) > 0
                }
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching balance: {e}")
            raise
    
    def get_ticker_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching price for {symbol}: {e}")
            raise
    
    def cancel_order(self, symbol: str, order_id: Optional[int] = None, 
                    client_order_id: Optional[str] = None) -> Dict:
        """Cancel an open order"""
        try:
            params = {'symbol': symbol}
            if order_id:
                params['orderId'] = order_id
            if client_order_id:
                params['origClientOrderId'] = client_order_id
                
            return self.client.cancel_order(**params)
        except BinanceAPIException as e:
            self.logger.error(f"Error canceling order for {symbol}: {e}")
            raise
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders for a symbol or all symbols"""
        try:
            if symbol:
                return self.client.get_open_orders(symbol=symbol)
            return self.client.get_open_orders()
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching open orders: {e}")
            raise