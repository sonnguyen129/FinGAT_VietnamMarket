"""
Feature Engineering for Vietnamese Stock Market
Creates technical indicators and features for FinGAT model
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Feature engineering for stock data"""
    
    def __init__(self, 
                 window_sizes: List[int] = [5, 10, 15, 20, 25, 30],
                 normalize: bool = True):
        """
        Initialize Feature Engineer
        
        Args:
            window_sizes: List of moving average window sizes
            normalize: Whether to normalize features
        """
        self.window_sizes = window_sizes
        self.normalize = normalize
        self.scalers = {}
        
    def create_price_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price ratio features
        
        Args:
            df: Stock dataframe with OHLC data
        
        Returns:
            DataFrame with added price ratio features
        """
        # Open/Close ratio
        df['c_open'] = (df['Open'] / df['Close']) - 1
        
        # High/Close ratio
        df['c_high'] = (df['High'] / df['Close']) - 1
        
        # Low/Close ratio  
        df['c_low'] = (df['Low'] / df['Close']) - 1
        
        # Volume ratio (current vs average)
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20, min_periods=1).mean()
        
        # Price range (High-Low)/Close
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        
        return df
    
    def create_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create moving average features
        
        Args:
            df: Stock dataframe with price data
        
        Returns:
            DataFrame with moving average features
        """
        for window in self.window_sizes:
            # Simple moving average
            ma_col = f'MA_{window}'
            df[ma_col] = df['Close'].rolling(window=window, min_periods=1).mean()
            
            # MA ratio (Close/MA - 1)
            df[f'{window}-days'] = (df[ma_col] / df['Close']) - 1
            
            # Drop the MA column (keep only ratio)
            df = df.drop(columns=[ma_col])
        
        return df
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional technical indicators
        
        Args:
            df: Stock dataframe
        
        Returns:
            DataFrame with technical indicators
        """
        # RSI (Relative Strength Index)
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Bollinger Bands
        df = self._calculate_bollinger_bands(df)
        
        # MACD
        df = self._calculate_macd(df)
        
        # Volume weighted average price (VWAP)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['VWAP_ratio'] = (df['Close'] / df['VWAP']) - 1
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # Fill NaN with neutral value
        
        # Normalize to [-1, 1]
        return (rsi - 50) / 50
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, num_std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        rolling_mean = df['Close'].rolling(window=period, min_periods=1).mean()
        rolling_std = df['Close'].rolling(window=period, min_periods=1).std()
        
        df['BB_upper'] = rolling_mean + (rolling_std * num_std)
        df['BB_lower'] = rolling_mean - (rolling_std * num_std)
        
        # Position within bands (normalized to [-1, 1])
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        df['BB_position'] = df['BB_position'].clip(0, 1) * 2 - 1
        
        # Band width (volatility indicator)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / rolling_mean
        
        # Drop intermediate columns
        df = df.drop(columns=['BB_upper', 'BB_lower'])
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicator"""
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        # MACD histogram (normalized by price)
        df['MACD_hist'] = (macd - signal_line) / df['Close']
        
        return df
    
    def create_all_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Create all features for a stock
        
        Args:
            df: Stock dataframe
            symbol: Stock symbol (for scaler tracking)
        
        Returns:
            DataFrame with all features
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure we have return
        if 'Return' not in df.columns:
            df['Return'] = df['Close'].pct_change().fillna(0)
        
        # Create features
        df = self.create_price_ratios(df)
        df = self.create_moving_averages(df)
        df = self.create_technical_indicators(df)
        
        # Drop rows with NaN (from indicators)
        df = df.fillna(method='ffill').fillna(0)
        
        # Normalize if requested
        if self.normalize and symbol:
            df = self._normalize_features(df, symbol)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding original OHLCV data)
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of feature column names
        """
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        return [col for col in df.columns if col not in exclude_cols]
    
    def _normalize_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize features using StandardScaler"""
        # Features to normalize (exclude date, prices, volume)
        feature_cols = [col for col in df.columns 
                       if col not in ['Date', 'Open', 'High', 'Low', 'Close', 
                                     'Adj Close', 'Volume', 'Return', 'LogReturn']]
        
        if symbol not in self.scalers:
            self.scalers[symbol] = StandardScaler()
            df[feature_cols] = self.scalers[symbol].fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scalers[symbol].transform(df[feature_cols])
        
        return df
    
    def create_sliding_windows(self, 
                              df: pd.DataFrame,
                              window_size: int = 15,
                              target_size: int = 1,
                              stride: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding windows for model input
        
        Args:
            df: Feature dataframe
            window_size: Number of days for input (15 for FinGAT)
            target_size: Number of days to predict (1 for next day)
            stride: Step size for sliding window
        
        Returns:
            Tuple of (features, returns, movements)
        """
        # Select feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['Date', 'Open', 'High', 'Low', 'Close', 
                                     'Adj Close', 'Volume']]
        
        features = []
        returns = []
        movements = []
        
        for i in range(0, len(df) - window_size - target_size + 1, stride):
            # Input window
            window_features = df[feature_cols].iloc[i:i+window_size].values
            features.append(window_features)
            
            # Target (next day return and movement)
            target_return = df['Return'].iloc[i+window_size:i+window_size+target_size].values
            returns.append(target_return)
            
            # Movement (1 if positive, 0 if negative)
            movement = (target_return >= 0).astype(int)
            movements.append(movement)
        
        return np.array(features), np.array(returns), np.array(movements)
    
    def create_week_grouping(self, features: np.ndarray, days_per_week: int = 5) -> List[np.ndarray]:
        """
        Group daily features into weekly format for FinGAT
        
        Args:
            features: Array of shape (samples, days, features)
            days_per_week: Trading days per week
        
        Returns:
            List of weekly feature arrays
        """
        samples, days, num_features = features.shape
        num_weeks = days // days_per_week
        
        weekly_features = []
        for week_idx in range(num_weeks):
            start_idx = week_idx * days_per_week
            end_idx = start_idx + days_per_week
            week_data = features[:, start_idx:end_idx, :]
            weekly_features.append(week_data)
        
        # Handle remaining days (if days % days_per_week != 0)
        remaining_days = days % days_per_week
        if remaining_days > 0:
            # Pad the last week with zeros or repeat last day
            last_week = features[:, -days_per_week:, :]
            weekly_features.append(last_week)
        
        return weekly_features


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    sample_df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.randn(len(dates)) * 10 + 100,
        'High': np.random.randn(len(dates)) * 10 + 105,
        'Low': np.random.randn(len(dates)) * 10 + 95,
        'Close': np.random.randn(len(dates)) * 10 + 100,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Create features
    feature_df = fe.create_all_features(sample_df, symbol='TEST')
    
    print("Feature columns created:")
    print(feature_df.columns.tolist())
    
    # Create sliding windows
    X, y_return, y_movement = fe.create_sliding_windows(feature_df)
    print(f"\nSliding windows shape: {X.shape}")
    print(f"Return targets shape: {y_return.shape}")
    print(f"Movement targets shape: {y_movement.shape}")
    
    # Group into weeks
    weekly_features = fe.create_week_grouping(X)
    print(f"\nNumber of weeks: {len(weekly_features)}")
    if weekly_features:
        print(f"Week feature shape: {weekly_features[0].shape}")