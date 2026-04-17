"""
Vietnamese Stock Market Data Loader
Loads and processes stock data from VN_datasets folder
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class VNDataLoader:
    """Data loader for Vietnamese stock market"""
    
    def __init__(self, 
                 data_path: str = 'datasets/VN_datasets/',
                 companies_file: str = 'datasets/VN_Companies.csv',
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 min_history_days: int = 365):
        """
        Initialize VN Data Loader
        
        Args:
            data_path: Path to VN_datasets folder
            companies_file: Path to companies info CSV
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            min_history_days: Minimum required history days
        """
        self.data_path = data_path
        self.companies_file = companies_file
        self.start_date = start_date
        self.end_date = end_date
        self.min_history_days = min_history_days
        
        # Load company sector information
        self.companies_df = self._load_companies_info()
        
        # Storage for loaded data
        self.stock_data = {}
        self.valid_stocks = []
        self.sector_mapping = {}
        
    def _load_companies_info(self) -> pd.DataFrame:
        """Load company sector information"""
        try:
            df = pd.read_csv(self.companies_file)
            print(f"Loaded {len(df)} companies with sector information")
            return df
        except Exception as e:
            print(f"Error loading companies file: {e}")
            return pd.DataFrame()
    
    def load_all_stocks(self, limit: Optional[int] = None) -> Dict:
        """
        Load all stock data from CSV files
        
        Args:
            limit: Limit number of stocks to load (for testing)
        
        Returns:
            Dictionary with stock data
        """
        stock_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        
        if limit:
            stock_files = stock_files[:limit]
        
        print(f"Loading {len(stock_files)} stock files...")
        
        for file in stock_files:
            symbol = file.replace('.csv', '')
            
            # Check if symbol exists in companies info
            if symbol not in self.companies_df['Symbol'].values:
                continue
            
            try:
                # Load stock data
                df = pd.read_csv(os.path.join(self.data_path, file))

                # Convert date column
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Filter by date range if specified
                if self.start_date:
                    df = df[df['Date'] >= pd.to_datetime(self.start_date)]
                if self.end_date:
                    df = df[df['Date'] <= pd.to_datetime(self.end_date)]
                
                # Check minimum history requirement
                if len(df) < self.min_history_days:
                    continue
                
                # Sort by date
                df = df.sort_values('Date').reset_index(drop=True)
                
                # Get sector information
                sector = self.companies_df[
                    self.companies_df['Symbol'] == symbol
                ]['Sector'].iloc[0]
                
                # Store data
                self.stock_data[symbol] = {
                    'data': df,
                    'sector': sector,
                    'name': self.companies_df[
                        self.companies_df['Symbol'] == symbol
                    ]['Name'].iloc[0]
                }
                
                self.valid_stocks.append(symbol)
                
                # Update sector mapping
                if sector not in self.sector_mapping:
                    self.sector_mapping[sector] = []
                self.sector_mapping[sector].append(symbol)
                
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.valid_stocks)} stocks")
        print(f"Sectors found: {list(self.sector_mapping.keys())}")
        
        return self.stock_data
    
    def align_dates(self) -> None:
        """Align all stocks to have the same trading dates"""
        if not self.stock_data:
            print("No stock data loaded. Run load_all_stocks() first.")
            return
        
        # Find common dates across all stocks
        all_dates = []
        for symbol in self.valid_stocks:
            dates = set(self.stock_data[symbol]['data']['Date'].values)
            all_dates.append(dates)
        
        # Get intersection of all dates
        common_dates = set.intersection(*all_dates)
        common_dates = sorted(list(common_dates))
        
        print(f"Found {len(common_dates)} common trading dates")
        
        # Filter each stock to only include common dates
        for symbol in self.valid_stocks:
            df = self.stock_data[symbol]['data']
            df = df[df['Date'].isin(common_dates)]
            df = df.sort_values('Date').reset_index(drop=True)
            self.stock_data[symbol]['data'] = df
    
    def calculate_returns(self) -> None:
        """Calculate daily returns for all stocks"""
        for symbol in self.valid_stocks:
            df = self.stock_data[symbol]['data']
            
            # Calculate daily returns
            df['Return'] = df['Close'].pct_change()
            
            # Calculate log returns
            df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Fill first row NaN with 0
            df['Return'] = df['Return'].fillna(0)
            df['LogReturn'] = df['LogReturn'].fillna(0)
            
            self.stock_data[symbol]['data'] = df
    
    def get_price_matrix(self, price_type: str = 'Close') -> pd.DataFrame:
        """
        Get price matrix for all stocks
        
        Args:
            price_type: Type of price (Open, High, Low, Close, Adj Close)
        
        Returns:
            DataFrame with dates as index and stocks as columns
        """
        if not self.stock_data:
            print("No data loaded")
            return pd.DataFrame()
        
        # Get first stock to initialize dataframe
        first_symbol = self.valid_stocks[0]
        dates = self.stock_data[first_symbol]['data']['Date']
        
        # Create price matrix
        price_matrix = pd.DataFrame(index=dates)
        
        for symbol in self.valid_stocks:
            price_matrix[symbol] = self.stock_data[symbol]['data'][price_type].values
        
        return price_matrix
    
    def get_return_matrix(self) -> pd.DataFrame:
        """Get return matrix for all stocks"""
        if 'Return' not in self.stock_data[self.valid_stocks[0]]['data'].columns:
            self.calculate_returns()
        
        return self.get_price_matrix('Return')
    
    def get_sector_stocks(self, sector: str) -> List[str]:
        """Get list of stocks in a specific sector"""
        return self.sector_mapping.get(sector, [])
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get information for a specific stock"""
        if symbol not in self.stock_data:
            return {}
        return {
            'symbol': symbol,
            'name': self.stock_data[symbol]['name'],
            'sector': self.stock_data[symbol]['sector'],
            'data_points': len(self.stock_data[symbol]['data']),
            'date_range': (
                self.stock_data[symbol]['data']['Date'].min(),
                self.stock_data[symbol]['data']['Date'].max()
            )
        }
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for all loaded stocks"""
        stats = []
        
        for symbol in self.valid_stocks:
            df = self.stock_data[symbol]['data']
            
            if 'Return' in df.columns:
                stats.append({
                    'Symbol': symbol,
                    'Sector': self.stock_data[symbol]['sector'],
                    'Data Points': len(df),
                    'Mean Return': df['Return'].mean(),
                    'Std Return': df['Return'].std(),
                    'Min Price': df['Close'].min(),
                    'Max Price': df['Close'].max(),
                    'Avg Volume': df['Volume'].mean()
                })
        
        return pd.DataFrame(stats)
    
    def get_stock_to_sector_mapping(self) -> Dict[str, str]:
        """
        Get stock-to-sector mapping for GraphConstructor
        
        Returns:
            Dictionary mapping stock symbol to sector name
        """
        stock_to_sector = {}
        for sector, stocks in self.sector_mapping.items():
            for stock in stocks:
                stock_to_sector[stock] = sector
        return stock_to_sector


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = VNDataLoader(
        data_path='./datasets/VN_datasets/',
        companies_file='./datasets/VN_Companies.csv',
        start_date='2019-01-01',
        end_date='2023-12-31',
        min_history_days=250
    )
    
    # Load stocks (limit to 50 for testing)
    stock_data = loader.load_all_stocks(limit=50)
    
    # Align dates
    loader.align_dates()
    
    # Calculate returns
    loader.calculate_returns()
    
    # Get summary
    summary = loader.get_summary_statistics()
    print("\nSummary Statistics:")
    print(summary.head(10))
    
    # Get sector information
    print("\nSectors and stock counts:")
    for sector, stocks in loader.sector_mapping.items():
        print(f"{sector}: {len(stocks)} stocks")