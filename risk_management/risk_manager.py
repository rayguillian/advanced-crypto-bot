import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

class RiskManager:
    def __init__(self, parameters: Dict = None):
        self.logger = logging.getLogger(__name__)
        
        # Default risk parameters
        self.params = {
            'max_position_size': 0.20,      # Maximum 20% of portfolio per position
            'base_risk_per_trade': 0.02,    # Risk 2% of portfolio per trade
            'max_risk_multiplier': 2.0,     # Maximum risk multiplier based on conviction
            'min_reward_risk_ratio': 2.0,   # Minimum reward:risk ratio
            'max_correlation_trades': 3,     # Maximum number of correlated trades
            'max_drawdown': 0.15,           # Maximum drawdown before reducing position sizes
            'vol_lookback_period': 20,      # Periods for volatility calculation
            'vol_risk_adjustment': True,     # Adjust position size based on volatility
            'portfolio_heat': 0.5,          # Maximum portfolio heat (total risk exposure)
            'atr_periods': 14,              # Periods for ATR calculation
            'atr_stop_multiplier': 2.5      # Multiplier for ATR-based stops
        }
        
        if parameters:
            self.params.update(parameters)
        
        self.positions = {}
        self.portfolio_value = 0
        self.current_drawdown = 0
    
    def calculate_position_size(self, symbol: str, signal: Dict, price: float, portfolio_value: float) -> Dict:
        try:
            base_size = portfolio_value * self.params['base_risk_per_trade']
            conviction_multiplier = min(signal.get('strength', 0.5), self.params['max_risk_multiplier'])
            
            if self.params['vol_risk_adjustment'] and 'volatility' in signal:
                vol_multiplier = 1 / (1 + signal['volatility'])
            else:
                vol_multiplier = 1.0
            
            if self.current_drawdown > self.params['max_drawdown']:
                drawdown_factor = 1 - (self.current_drawdown / 2)
            else:
                drawdown_factor = 1.0
            
            position_size = base_size * conviction_multiplier * vol_multiplier * drawdown_factor
            max_size = portfolio_value * self.params['max_position_size']
            position_size = min(position_size, max_size)
            
            quantity = position_size / price
            atr = signal.get('atr', price * 0.02)
            stop_distance = atr * self.params['atr_stop_multiplier']
            
            stop_loss = price - stop_distance if signal['signal'] > 0 else price + stop_distance
            min_take_profit_distance = stop_distance * self.params['min_reward_risk_ratio']
            take_profit = price + min_take_profit_distance if signal['signal'] > 0 else price - min_take_profit_distance
            
            return {
                'quantity': quantity,
                'position_value': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': position_size * (stop_distance / price),
                'reward_risk_ratio': self.params['min_reward_risk_ratio']
            }
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            raise