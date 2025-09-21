"""
Data Processing Module for FinGAT Vietnam Project

This module handles loading, preprocessing, and feature engineering for Vietnamese stock market data.
Implements the data processing pipeline as described in the FinGAT paper.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings

from src.utils.seed_utils import set_seed
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)


class VietnamDataProcessor:
    def generate_sliding_windows(self, window_size: int = 16, week_size: int = 5, target_shift: int = 1):
        """
        Sinh các cửa sổ trượt (sliding window) cho từng mã cổ phiếu.
        Mỗi cửa sổ gồm 15 ngày (3 tuần × 5 ngày) + 1 ngày target.
        Gán nhãn y_return (return ngày tiếp theo), y_move (up/down movement).
        Returns:
            Dict[symbol, List[Dict]]: Mỗi symbol là 1 list các dict chứa window, label, v.v.
        """
        all_windows = {}
        for symbol, df in self.stock_data.items():
            df = df.sort_values('Date').reset_index(drop=True)
            windows = []
            for i in range(len(df) - window_size):
                window_df = df.iloc[i:i+window_size]
                # 3 tuần × 5 ngày: lấy 15 ngày đầu làm input, ngày thứ 16 làm target
                input_window = window_df.iloc[:window_size-1]
                target_row = window_df.iloc[window_size-1]
                # y_return: return của ngày target (so với close ngày cuối window)
                prev_close = input_window['Close'].values[-1]
                target_close = target_row['Close']
                y_return = (target_close - prev_close) / prev_close if prev_close != 0 else 0.0
                # y_move: 1 nếu y_return > 0, 0 nếu y_return <= 0
                y_move = int(y_return > 0)
                windows.append({
                    'symbol': symbol,
                    'start_date': input_window['Date'].values[0],
                    'end_date': input_window['Date'].values[-1],
                    'input_window': input_window,
                    'target_date': target_row['Date'],
                    'y_return': y_return,
                    'y_move': y_move
                })
            all_windows[symbol] = windows
        logger.info(f"Generated sliding windows for {len(all_windows)} stocks.")
        return all_windows
    def build_common_trading_calendar(self):
        """
        Build a common trading calendar (intersection of all trading days).
        Returns:
            DatetimeIndex of common trading days
        """
        all_dates = []
        if not self.stock_data:
            logger.error("No stock data loaded. Please check your data loading step.")
            return pd.DatetimeIndex([])
        for symbol, df in self.stock_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                logger.warning(f"Stock {symbol} has empty DataFrame, skipping.")
                continue
            if 'Date' not in df.columns:
                logger.warning(f"Stock {symbol} missing 'Date' column, skipping.")
                continue
            all_dates.append(pd.Series(df['Date'].dropna().unique()))
        if not all_dates:
            logger.error("No trading dates found in any stock data. Please check your CSV files and cleaning logic.")
            return pd.DatetimeIndex([])
        # Find intersection of all trading dates
        common_dates = set(all_dates[0])
        for dates in all_dates[1:]:
            common_dates = common_dates & set(dates)
        if not common_dates:
            logger.error("No common trading dates found across stocks. Check for inconsistent or missing dates.")
            return pd.DatetimeIndex([])
        common_calendar = pd.DatetimeIndex(sorted(common_dates))
        logger.info(f"Common trading calendar has {len(common_calendar)} dates.")
        return common_calendar

    def sync_all_stocks_to_calendar(self, calendar=None):
        """
        Reindex all stock DataFrames to the common trading calendar.
        Args:
            calendar: DatetimeIndex of trading days (if None, build automatically)
        """
        if calendar is None:
            calendar = self.build_common_trading_calendar()
        for symbol, df in self.stock_data.items():
            if 'Date' not in df.columns:
                continue
            df = df.set_index('Date').reindex(calendar)
            df['Symbol'] = symbol
            # Optionally, fill missing values (forward fill then back fill)
            df = df.fillna(method='ffill').fillna(method='bfill')
            df = df.reset_index().rename(columns={'index': 'Date'})
            self.stock_data[symbol] = df
        logger.info("All stocks synchronized to common trading calendar.")
    """
    Data processor for Vietnam stock market data.

    Handles loading CSV files, data cleaning, feature engineering,
    and preparation for FinGAT model training.
    """

    def __init__(self, config: Dict):
        """
        Initialize the data processor.

        Args:
            config: Configuration dictionary containing data processing parameters
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.feature_config = config.get('feature_engineering', {})

        # Paths
        self.raw_data_path = Path(config.get('paths', {}).get('raw_data', 'data/raw'))
        self.processed_data_path = Path(config.get('paths', {}).get('processed_data', 'data/processed'))

        # Ensure processed data directory exists
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.stock_data = {}
        self.sector_mapping = {}
        self.features_df = None

        logger.info(f"VietnamDataProcessor initialized with config: {self.data_config.get('market', 'vietnam')}")

    def load_sector_mapping(self, sector_file: str = None) -> Dict[str, str]:
        """
        Load sector mapping from CSV file.

        Args:
            sector_file: Path to sector mapping CSV file

        Returns:
            Dictionary mapping stock symbols to sectors
        """
        if sector_file is None:
            sector_file = Path(self.data_config.get('sector_mapping', {}).get('file', 'datasets/VN_Companies.csv'))

        try:
            sector_df = pd.read_csv(sector_file)
            stock_col = self.data_config.get('sector_mapping', {}).get('stock_column', 'Symbol')
            sector_col = self.data_config.get('sector_mapping', {}).get('sector_column', 'Sector')

            self.sector_mapping = dict(zip(sector_df[stock_col], sector_df[sector_col]))

            logger.info(f"Loaded sector mapping for {len(self.sector_mapping)} stocks")
            return self.sector_mapping

        except Exception as e:
            logger.error(f"Error loading sector mapping: {e}")
            return {}

    def load_stock_data(self, stock_files: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load stock data from CSV files.

        Args:
            stock_files: List of stock CSV file paths. If None, loads all CSV files from config data_sources

        Returns:
            Dictionary mapping stock symbols to their DataFrames
        """
        if stock_files is None:
            # Lấy thông tin từ config nếu có
            stock_files_config = self.config.get('data_sources', {}).get('stock_files', {})
            stock_path = stock_files_config.get('path', str(self.raw_data_path))
            stock_pattern = stock_files_config.get('pattern', '*.csv')
            stock_files = list(Path(stock_path).glob(stock_pattern))
            # Filter out sector mapping file
            stock_files = [f for f in stock_files if 'companies' not in f.name.lower()]

        loaded_count = 0

        for file_path in stock_files:
            try:
                # Extract stock symbol from filename
                symbol = Path(file_path).stem.upper()

                # Load data
                df = pd.read_csv(file_path)

                # Clean and validate data
                df_cleaned = self._clean_stock_data(df, symbol)

                if df_cleaned is not None and len(df_cleaned) > 0:
                    self.stock_data[symbol] = df_cleaned
                    loaded_count += 1

            except Exception as e:
                logger.warning(f"Failed to load data for {file_path}: {e}")

        logger.info(f"Successfully loaded data for {loaded_count} stocks")
        return self.stock_data

    def _clean_stock_data(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """
        Clean and validate stock data.

        Args:
            df: Raw stock DataFrame
            symbol: Stock symbol

        Returns:
            Cleaned DataFrame or None if data is invalid
        """
        try:
            # Copy dataframe
            df = df.copy()

            # Handle column mapping for Vietnamese data
            column_mapping = self.data_config.get('processing', {}).get('column_mapping', {})
            # Ensure Adj Close is mapped if present
            if 'Adj Close' not in column_mapping and 'Adj Close' in df.columns:
                column_mapping['Adj Close'] = 'Adj Close'
            df = df.rename(columns=column_mapping)

            # Check required columns (add Adj Close if present in any file)
            required_cols = self.data_config.get('processing', {}).get('required_columns', [])
            if 'Adj Close' not in required_cols and 'Adj Close' in df.columns:
                required_cols = required_cols + ['Adj Close']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.warning(f"Missing columns for {symbol}: {missing_cols}")
                return None

            # Convert date column
            date_col = self.data_config.get('processing', {}).get('date_column', 'Date')
            date_format = self.data_config.get('processing', {}).get('date_format', '%Y-%m-%d')

            df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)

            # Clean price and volume data
            price_cols = ['Open', 'High', 'Low', 'Close']
            volume_cols = ['Volume']

            # Remove zero/negative prices and volumes
            cleaning_config = self.data_config.get('processing', {}).get('cleaning', {})

            if cleaning_config.get('remove_zero_price', True):
                for col in price_cols:
                    if col in df.columns:
                        df = df[df[col] > 0]

            if cleaning_config.get('remove_zero_volume', True):
                for col in volume_cols:
                    if col in df.columns:
                        df = df[df[col] > 0]

            # Check price consistency (High >= Low, Close between High/Low)
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                df = df[df['High'] >= df['Low']]
                df = df[(df['Close'] >= df['Low']) & (df['Close'] <= df['High'])]

            # Remove outliers if configured
            if cleaning_config.get('outlier_detection', False):
                df = self._remove_outliers(df, symbol, cleaning_config.get('outlier_threshold', 3))

            # Check minimum data requirements
            min_days = self.data_config.get('processing', {}).get('min_trading_days', 252)
            if len(df) < min_days:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} days (minimum: {min_days})")
                return None

            # Add stock symbol column
            df['Symbol'] = symbol

            logger.debug(f"Cleaned data for {symbol}: {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error cleaning data for {symbol}: {e}")
            return None

    def _remove_outliers(self, df: pd.DataFrame, symbol: str, threshold: float = 3) -> pd.DataFrame:
        """
        Remove outliers using z-score method.

        Args:
            df: Stock DataFrame
            symbol: Stock symbol
            threshold: Z-score threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['Date']]

        for col in numeric_cols:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > threshold

                if outliers.sum() > 0:
                    logger.debug(f"Removed {outliers.sum()} outliers from {symbol}.{col}")
                    df = df[~outliers]

        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a stock DataFrame.

        Args:
            df: Stock DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicators added
        """
        df = df.copy()

        try:
            # Simple Moving Averages
            sma_periods = self.feature_config.get('technical_indicators', {}).get('sma_periods', [5, 10, 20])
            for period in sma_periods:
                df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)

            # Exponential Moving Averages
            ema_periods = self.feature_config.get('technical_indicators', {}).get('ema_periods', [12, 26])
            for period in ema_periods:
                df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)

            # RSI
            rsi_period = self.feature_config.get('technical_indicators', {}).get('rsi_period', 14)
            df['RSI_14'] = ta.momentum.rsi(df['Close'], window=rsi_period)

            # MACD
            macd_config = self.feature_config.get('technical_indicators', {}).get('macd', {})
            macd_fast = macd_config.get('fast_period', 12)
            macd_slow = macd_config.get('slow_period', 26)
            macd_signal = macd_config.get('signal_period', 9)

            df['MACD'] = ta.trend.macd_diff(df['Close'], window_slow=macd_slow, window_fast=macd_fast)
            df['MACD_signal'] = ta.trend.macd_signal(df['Close'], window_slow=macd_slow, 
                                                   window_fast=macd_fast, window_sign=macd_signal)
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']

            # Bollinger Bands
            bb_config = self.feature_config.get('technical_indicators', {}).get('bollinger_bands', {})
            bb_period = bb_config.get('period', 20)
            bb_std = bb_config.get('std_multiplier', 2)

            df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], window=bb_period, window_dev=bb_std)
            df['BB_middle'] = ta.volatility.bollinger_mavg(df['Close'], window=bb_period)
            df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], window=bb_period, window_dev=bb_std)

            # Volume indicators
            vol_sma_period = self.feature_config.get('technical_indicators', {}).get('volume_sma_period', 20)
            df['volume_sma'] = ta.volume.volume_sma(df['Close'], df['Volume'], window=vol_sma_period)
            df['price_volume_trend'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])

            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df

    def calculate_returns_and_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns and volatility features.

        Args:
            df: Stock DataFrame

        Returns:
            DataFrame with return and volatility features added
        """
        df = df.copy()

        try:
            # Return periods
            return_periods = self.feature_config.get('returns', {}).get('periods', [1, 5, 10])

            for period in return_periods:
                df[f'return_{period}d'] = df['Close'].pct_change(periods=period)

            # Volatility periods
            vol_periods = self.feature_config.get('volatility', {}).get('periods', [10, 20])
            vol_method = self.feature_config.get('volatility', {}).get('method', 'std')

            for period in vol_periods:
                if vol_method == 'std':
                    df[f'volatility_{period}d'] = df['return_1d'].rolling(window=period).std()
                # Add other volatility methods (Parkinson, Garman-Klass) here if needed

            return df

        except Exception as e:
            logger.error(f"Error calculating returns and volatility: {e}")
            return df

    def normalize_features(self, df: pd.DataFrame, method: str = 'z_score', 
                          window: int = 252) -> pd.DataFrame:
        """
        Normalize features using specified method.

        Args:
            df: DataFrame with features
            method: Normalization method ('z_score', 'min_max', 'robust')
            window: Rolling window for normalization

        Returns:
            DataFrame with normalized features
        """
        df = df.copy()

        # Get feature columns (exclude date, symbol, and raw OHLCV)
        exclude_cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        try:
            if method == 'z_score':
                for col in feature_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        rolling_mean = df[col].rolling(window=window, min_periods=30).mean()
                        rolling_std = df[col].rolling(window=window, min_periods=30).std()
                        df[f'{col}_norm'] = (df[col] - rolling_mean) / rolling_std

            elif method == 'min_max':
                for col in feature_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        rolling_min = df[col].rolling(window=window, min_periods=30).min()
                        rolling_max = df[col].rolling(window=window, min_periods=30).max()
                        df[f'{col}_norm'] = (df[col] - rolling_min) / (rolling_max - rolling_min)

            # Add other normalization methods as needed

            return df

        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return df

    def process_all_stocks(self) -> pd.DataFrame:
        """
        Process all loaded stock data and create feature matrix.

        Returns:
            Combined DataFrame with all processed features
        """
        logger.info("Starting feature engineering for all stocks...")

        # Đồng bộ lịch giao dịch chung trước khi xử lý features
        self.sync_all_stocks_to_calendar()

        processed_dfs = []

        for symbol, df in self.stock_data.items():
            try:
                # Calculate technical indicators
                df_with_indicators = self.calculate_technical_indicators(df)

                # Calculate returns and volatility
                df_with_returns = self.calculate_returns_and_volatility(df_with_indicators)

                # Normalize features
                normalization_config = self.feature_config.get('normalization', {})
                method = normalization_config.get('method', 'z_score')
                window = normalization_config.get('rolling_window', 252)

                df_normalized = self.normalize_features(df_with_returns, method, window)

                # Add sector information
                if symbol in self.sector_mapping:
                    df_normalized['Sector'] = self.sector_mapping[symbol]
                else:
                    df_normalized['Sector'] = 'Unknown'
                    logger.warning(f"No sector mapping found for {symbol}")

                processed_dfs.append(df_normalized)
                logger.debug(f"Processed features for {symbol}: {df_normalized.shape}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        if processed_dfs:
            self.features_df = pd.concat(processed_dfs, ignore_index=True)
            logger.info(f"Feature engineering completed. Total shape: {self.features_df.shape}")

            # Save processed data
            output_file = self.processed_data_path / "processed_features.csv"
            self.features_df.to_csv(output_file, index=False)
            logger.info(f"Processed features saved to {output_file}")

            return self.features_df
        else:
            logger.error("No data was successfully processed")
            return pd.DataFrame()

    def get_feature_summary(self) -> Dict:
        """
        Get summary statistics of processed features.

        Returns:
            Dictionary with feature summary information
        """
        if self.features_df is None or self.features_df.empty:
            return {}

        summary = {
            'total_records': len(self.features_df),
            'unique_stocks': self.features_df['Symbol'].nunique(),
            'date_range': {
                'start': self.features_df['Date'].min(),
                'end': self.features_df['Date'].max()
            },
            'sectors': self.features_df['Sector'].value_counts().to_dict(),
            'feature_columns': [col for col in self.features_df.columns 
                              if col not in ['Date', 'Symbol', 'Sector']],
            'missing_values': self.features_df.isnull().sum().to_dict()
        }

        return summary


# Example usage
if __name__ == "__main__":
    # Example: Load and merge config.yaml & data_config.yaml for VietnamDataProcessor
    import yaml
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)
    with open('configs/data_config.yaml', 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    # Merge data_config vào main_config (ưu tiên data_sources, processing, feature_engineering...)
    merged_config = main_config.copy()
    merged_config['data_sources'] = data_config.get('data_sources', {})
    merged_config['processing'] = data_config.get('processing', {})
    merged_config['feature_engineering'] = data_config.get('feature_engineering', {})
    merged_config['paths'] = main_config.get('paths', {})
    # Nếu cần merge thêm các key khác, bổ sung tại đây

    # Initialize processor
    processor = VietnamDataProcessor(merged_config)

    # Load data
    processor.load_sector_mapping()
    processor.load_stock_data()

    # Process all stocks
    features_df = processor.process_all_stocks()

    # Get summary
    summary = processor.get_feature_summary()
    print(f"Processing completed: {summary}")