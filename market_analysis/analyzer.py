import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import math

class MarketAnalyzer:
    def __init__(self, exchange_handler, parameters: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange_handler
        
        # Default parameters
        self.params = {
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'volume_profile_bins': 24,
            'orderbook_depth': 20,
            'support_resistance_lookback': 100,
            'volume_significance_threshold': 1.5,
            'price_precision': 2
        }
        
        if parameters:
            self.params.update(parameters)

    def analyze_order_book(self, symbol: str, depth: int = None) -> Dict:
        try:
            depth = depth or self.params['orderbook_depth']
            orderbook = self.exchange.client.get_order_book(symbol=symbol, limit=depth)
            
            bids = np.array([[float(price), float(qty)] for price, qty in orderbook['bids']])
            asks = np.array([[float(price), float(qty)] for price, qty in orderbook['asks']])
            
            bid_cumsum = np.cumsum(bids[:, 1])
            ask_cumsum = np.cumsum(asks[:, 1])
            
            bid_clusters = self._cluster_price_levels(bids)
            ask_clusters = self._cluster_price_levels(asks)
            
            total_bid_volume = bid_cumsum[-1]
            total_ask_volume = ask_cumsum[-1]
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            bid_walls = self._detect_walls(bids)
            ask_walls = self._detect_walls(asks)
            
            return {
                'imbalance': imbalance,
                'support_levels': bid_clusters,
                'resistance_levels': ask_clusters,
                'bid_walls': bid_walls,
                'ask_walls': ask_walls,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'bid_ask_ratio': total_bid_volume / total_ask_volume if total_ask_volume > 0 else float('inf')
            }
        
        except Exception as e:
            self.logger.error(f"Error analyzing order book: {e}")
            raise

    def _cluster_price_levels(self, orders: np.ndarray) -> List[Dict]:
        clusters = []
        current_cluster = {
            'price_start': orders[0][0],
            'price_end': orders[0][0],
            'total_volume': orders[0][1]
        }
        
        price_threshold = orders[0][0] * 0.001
        
        for i in range(1, len(orders)):
            if abs(orders[i][0] - orders[i-1][0]) <= price_threshold:
                current_cluster['price_end'] = orders[i][0]
                current_cluster['total_volume'] += orders[i][1]
            else:
                if current_cluster['total_volume'] > orders[:, 1].mean() * self.params['volume_significance_threshold']:
                    clusters.append(current_cluster)
                current_cluster = {
                    'price_start': orders[i][0],
                    'price_end': orders[i][0],
                    'total_volume': orders[i][1]
                }
        
        if current_cluster['total_volume'] > orders[:, 1].mean() * self.params['volume_significance_threshold']:
            clusters.append(current_cluster)
        
        return clusters

    def _detect_walls(self, orders: np.ndarray) -> List[Dict]:
        mean_volume = orders[:, 1].mean()
        std_volume = orders[:, 1].std()
        threshold = mean_volume + (2 * std_volume)
        
        walls = []
        for price, volume in orders:
            if volume > threshold:
                walls.append({
                    'price': price,
                    'volume': volume,
                    'strength': (volume - mean_volume) / std_volume
                })
        
        return walls

    def analyze_multiple_timeframes(self, symbol: str, base_timeframe: str) -> Dict:
        try:
            analyses = {}
            
            for tf in self.params['timeframes']:
                if self._get_timeframe_minutes(tf) < self._get_timeframe_minutes(base_timeframe):
                    continue
                
                lookback_days = max(1, min(30, self._get_timeframe_minutes(tf) / 1440 * 10))
                start_time = datetime.now() - timedelta(days=lookback_days)
                
                df = self.exchange.fetch_historical_data(
                    symbol=symbol,
                    interval=tf,
                    start_time=start_time
                )
                
                analyses[tf] = self._analyze_timeframe(df)
            
            consensus = self._generate_timeframe_consensus(analyses)
            
            return {
                'timeframes': analyses,
                'consensus': consensus
            }
        
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis: {e}")
            raise

    def _analyze_timeframe(self, df: pd.DataFrame) -> Dict:
        sma20 = df['close'].rolling(window=20).mean()
        sma50 = df['close'].rolling(window=50).mean()
        sma200 = df['close'].rolling(window=200).mean()
        
        current_price = df['close'].iloc[-1]
        
        short_trend = 1 if current_price > sma20.iloc[-1] else -1
        medium_trend = 1 if current_price > sma50.iloc[-1] else -1
        long_trend = 1 if current_price > sma200.iloc[-1] else -1
        
        volume_profile = self._calculate_volume_profile(df)
        levels = self._find_support_resistance(df)
        
        return {
            'trends': {
                'short': short_trend,
                'medium': medium_trend,
                'long': long_trend
            },
            'volume_profile': volume_profile,
            'support_resistance': levels,
            'momentum': self._calculate_momentum(df),
            'volatility': self._calculate_volatility(df)
        }

    def _generate_timeframe_consensus(self, analyses: Dict) -> Dict:
        trend_votes = {
            'short': 0,
            'medium': 0,
            'long': 0
        }
        
        weights = {
            '1m': 0.5,
            '5m': 0.7,
            '15m': 0.8,
            '1h': 1.0,
            '4h': 1.2,
            '1d': 1.5
        }
        
        for timeframe, analysis in analyses.items():
            weight = weights.get(timeframe, 1.0)
            for trend_type, trend in analysis['trends'].items():
                trend_votes[trend_type] += trend * weight
        
        total_weight = sum(weights.values())
        for trend_type in trend_votes:
            trend_votes[trend_type] = trend_votes[trend_type] / total_weight
        
        return {
            'trend_consensus': trend_votes,
            'strength': abs(sum(trend_votes.values()) / 3),
            'direction': np.sign(sum(trend_votes.values()))
        }

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / self.params['volume_profile_bins']
        
        profile = []
        for i in range(self.params['volume_profile_bins']):
            price_level = df['low'].min() + (i * bin_size)
            mask = (df['low'] <= price_level + bin_size) & (df['high'] >= price_level)
            volume = df.loc[mask, 'volume'].sum()
            
            if volume > 0:
                profile.append({
                    'price_level': price_level,
                    'volume': volume
                })
        
        return {
            'profile': profile,
            'value_area_high': self._find_value_area_high(profile),
            'value_area_low': self._find_value_area_low(profile)
        }