"""
Backtesting module for FinGAT model
Implements Vietnamese market-specific trading simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FinGATBacktester:
    """
    Backtesting engine with Vietnamese market constraints
    - T+2 settlement
    - Transaction costs: 0.15% commission + 0.1% sell tax
    - Lot size: 100 shares
    - Price limits: ±7% daily
    """
    
    def __init__(self,
                 initial_capital: float = 1_000_000_000,  # 1 billion VND
                 transaction_cost: float = 0.0015,  # 0.15% commission
                 sell_tax: float = 0.001,  # 0.1% sell tax
                 min_lot_size: int = 100,  # Minimum 100 shares per trade
                 max_position_size: float = 0.2,  # Max 20% portfolio per stock
                 settlement_days: int = 2,  # T+2 settlement
                 price_limit: float = 0.07):  # ±7% daily price limit
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital in VND
            transaction_cost: Commission rate for buy/sell
            sell_tax: Additional tax on sell orders
            min_lot_size: Minimum shares per transaction
            max_position_size: Maximum position size as fraction of portfolio
            settlement_days: Settlement period (T+N)
            price_limit: Daily price change limit
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.sell_tax = sell_tax
        self.min_lot_size = min_lot_size
        self.max_position_size = max_position_size
        self.settlement_days = settlement_days
        self.price_limit = price_limit
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}  # {symbol: {'shares': n, 'price': p, 'date': d}}
        self.pending_orders = []  # Orders waiting for settlement
        self.portfolio_value = initial_capital
        
        # History tracking
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0
        self.total_tax = 0
        
    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.pending_orders = []
        self.portfolio_value = self.initial_capital
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0
        self.total_tax = 0
    
    def update(self, date: pd.Timestamp, signals: Dict[str, float], 
               current_prices: Dict[str, float], actual_returns: Dict[str, float]):
        """
        Update portfolio based on signals (for compatibility with evaluation script)
        
        Args:
            date: Current date
            signals: Dictionary of {symbol: weight} for target portfolio
            current_prices: Current prices for all stocks
            actual_returns: Actual returns for the day
        """
        # Convert to Series for internal use
        prices_series = pd.Series(current_prices)
        
        # Update portfolio value first
        self._update_portfolio_value(prices_series, date)
        
        # Apply returns to existing positions
        for symbol in list(self.positions.keys()):
            if symbol in actual_returns:
                ret = actual_returns[symbol]
                # Update position value based on return
                if symbol in self.positions:
                    old_value = self.positions[symbol]['shares'] * self.positions[symbol]['price']
                    new_value = old_value * (1 + ret)
                    # Update price to reflect return
                    if self.positions[symbol]['shares'] > 0:
                        self.positions[symbol]['price'] = new_value / self.positions[symbol]['shares']
        
        # Store current state
        self.current_date = date
        self.current_signals = signals
        self.current_prices = current_prices
    
    def get_results(self) -> Dict:
        """Get backtesting results"""
        return self._calculate_metrics()
    
    def _update_portfolio_value(self,
                                current_prices: pd.Series,
                                date: pd.Timestamp):
        """Update current portfolio value"""
        # Calculate positions value
        positions_value = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices.index:
                price = current_prices[symbol]
                if not pd.isna(price):
                    positions_value += position['shares'] * price
        
        # Total portfolio value
        self.portfolio_value = self.cash + positions_value
        
        # Record history
        self.portfolio_history.append({
            'date': date,
            'cash': self.cash,
            'positions_value': positions_value,
            'value': self.portfolio_value,
            'num_positions': len(self.positions)
        })
    
    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics"""
        if not self.portfolio_history:
            return {}
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        # Calculate returns
        portfolio_df['daily_return'] = portfolio_df['value'].pct_change()
        portfolio_df['cumulative_return'] = (1 + portfolio_df['daily_return']).cumprod() - 1
        
        # Performance metrics
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # Annualized return (assuming 252 trading days)
        n_days = len(portfolio_df)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        
        # Sharpe ratio
        if len(portfolio_df['daily_return']) > 1:
            daily_returns = portfolio_df['daily_return'].dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_df['daily_return']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'final_capital': self.portfolio_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_commission': self.total_commission,
            'total_tax': self.total_tax,
            'total_transaction_costs': self.total_commission + self.total_tax
        }