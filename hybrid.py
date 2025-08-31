import pandas as pd
import numpy as np
import os
import glob
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

class HybridPutCallStrategy:
    def __init__(self, symbol: str, target_delta: float = 0.3, initial_capital: float = 100000):
        """
        Hybrid Put/Covered Call Strategy - Switches between puts and covered calls
        
        Strategy Logic:
        1. Start with cash secured puts
        2. If put assigned (buy stock) -> switch to covered calls
        3. If call assigned (sell stock) -> switch back to puts
        4. Roll weekly on Fridays
        
        Args:
            symbol: Stock ticker symbol
            target_delta: Target delta for option selection (e.g., 0.3 for 30 delta)
            initial_capital: Starting capital for the strategy
        """
        self.symbol = symbol.upper()
        self.target_delta = target_delta
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Strategy state
        self.stock_shares = 0        # Shares of stock owned
        self.stock_basis = 0         # Average cost basis of stock
        self.strategy_mode = "PUT"   # "PUT" or "CALL" mode
        self.positions = []          # List of open positions
        self.trades = []             # History of all trades
        self.daily_pnl = []          # Daily P&L tracking
        self.weekly_summary = []     # Weekly P&L and position tracking
        
        # Options data
        self.options_data = {}       # Date -> DataFrame mapping
        self.data_sources = {}       # Date -> source file mapping
        
        # Stock price data
        self.stock_prices = {}       # Date -> adjusted close price mapping
        
    def load_options_data(self, data_dir: str, start_date: str = None, end_date: str = None) -> None:
        """Load all options data files for the symbol"""
        print(f"Loading options data for {self.symbol}...")
        
        # Determine which years to load based on date range
        years_to_load = ['2024']  # default
        if start_date and end_date:
            start_year = int(start_date.split('-')[0])
            end_year = int(end_date.split('-')[0])
            years_to_load = [str(year) for year in range(start_year, end_year + 1)]
        
        files = []
        for year in years_to_load:
            year_pattern = os.path.join(data_dir, self.symbol, year, '*.csv')
            year_files = glob.glob(year_pattern)
            files.extend(year_files)
            if year_files:
                print(f"  Found {len(year_files)} files for {year}")
        
        if not files:
            print(f"  Checking all available years...")
            # Fallback: load from all available years
            symbol_dir = os.path.join(data_dir, self.symbol)
            if os.path.exists(symbol_dir):
                for item in os.listdir(symbol_dir):
                    if item.isdigit():  # Year folder
                        year_pattern = os.path.join(symbol_dir, item, '*.csv')
                        year_files = glob.glob(year_pattern)
                        files.extend(year_files)
                        if year_files:
                            print(f"  Found {len(year_files)} files for {item}")
        
        if not files:
            raise ValueError(f"No options data found for {self.symbol} in {data_dir}")
        
        for file in files:
            # Extract date from filename: NVDA_2024-01-01_options.csv
            filename = os.path.basename(file)
            date_str = filename.split('_')[1]  # Get 2024-01-01 part
            
            try:
                df = pd.read_csv(file)
                # Basic data cleaning and filtering
                if not df.empty and 'strike' in df.columns.str.lower():
                    # Standardize column names
                    df.columns = df.columns.str.lower().str.replace(' ', '_')
                    
                    # Handle different column name formats
                    if 'last' in df.columns and 'lastprice' not in df.columns:
                        df['lastprice'] = df['last']
                    if 'mark' in df.columns and 'lastprice' not in df.columns:
                        df['lastprice'] = df['mark']
                    
                    # Ensure we have required columns
                    required_cols = ['strike', 'bid', 'ask']
                    if all(col in df.columns for col in required_cols):
                        self.options_data[date_str] = df
                        self.data_sources[date_str] = os.path.basename(file)
                        
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        print(f"Loaded data for {len(self.options_data)} trading days")
        
        # Load daily adjusted stock prices
        self.load_stock_prices(data_dir)
        
        # Apply split adjustments to options data
        self.apply_split_adjustments()
        
    def get_trading_fridays(self, start_date: str, end_date: str) -> List[str]:
        """Get all available Fridays between start and end date from actual data"""
        # Get all available dates from loaded data
        available_dates = list(self.options_data.keys())
        available_dates.sort()
        
        # Filter for dates in range
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        fridays = []
        for date_str in available_dates:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            if start <= date_obj <= end and date_obj.weekday() == 4:  # Friday = 4
                fridays.append(date_str)
        
        # If no Fridays found, use all available dates (for testing)
        if not fridays:
            print("No Fridays found, using all available trading days...")
            trading_days = []
            for date_str in available_dates:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                if start <= date_obj <= end:
                    trading_days.append(date_str)
            # Take every 5th day to simulate weekly trading
            fridays = trading_days[::5] if trading_days else []
        
        return fridays
    
    def find_next_friday_expiry(self, current_date: str) -> Optional[str]:
        """Find the next Friday expiry date (7 days from current Friday)"""
        current = datetime.strptime(current_date, '%Y-%m-%d')
        next_friday = current + timedelta(days=7)
        return next_friday.strftime('%Y-%m-%d')
    
    def select_put_by_delta(self, df: pd.DataFrame, target_delta: float, 
                           spot_price: float, target_expiry: str = None) -> Optional[Dict]:
        """Select put option closest to target delta"""
        # Handle different ways puts are identified in the data
        if 'type' in df.columns:
            puts = df[df['type'].str.upper() == 'PUT'].copy()
        else:
            # Fallback: assume puts are identified in contractID or symbol
            puts = df[df.get('contractid', df.get('contractID', '')).str.contains('P', na=False)].copy()
        
        # Filter by target expiration date if specified (find nearest expiry within 3 days)
        if target_expiry and 'expiration' in puts.columns:
            target_date = datetime.strptime(target_expiry, '%Y-%m-%d')
            available_expiries = puts['expiration'].unique()
            
            # Find the closest expiry within 3 days of target
            best_expiry = None
            min_diff = 4  # More than 3 days
            
            for expiry_str in available_expiries:
                if expiry_str == 'expiration':  # Skip header
                    continue
                try:
                    expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
                    diff = abs((expiry_date - target_date).days)
                    if diff < min_diff:
                        min_diff = diff
                        best_expiry = expiry_str
                except:
                    continue
            
            if best_expiry:
                puts = puts[puts['expiration'] == best_expiry]
            else:
                # Fallback: use first available weekly expiry (within 14 days)
                puts = puts[puts['expiration'] != 'expiration']  # Remove header if present
        
        puts = puts[puts['strike'] < spot_price * 1.1]  # Within 10% of spot
        
        if puts.empty:
            return None
        
        if 'delta' in puts.columns:
            puts['abs_delta'] = abs(puts['delta'])
            puts['delta_diff'] = abs(puts['abs_delta'] - target_delta)
            selected = puts.loc[puts['delta_diff'].idxmin()]
        else:
            # Estimate delta based on moneyness
            puts['moneyness'] = puts['strike'] / spot_price
            puts['estimated_delta'] = np.maximum(0.1, np.minimum(0.5, 
                                               0.5 * (1.1 - puts['moneyness'])))
            puts['delta_diff'] = abs(puts['estimated_delta'] - target_delta)
            selected = puts.loc[puts['delta_diff'].idxmin()]
        
        bid = selected.get('bid', 0)
        ask = selected.get('ask', selected.get('lastprice', 0))
        
        if bid <= 0 or ask <= 0:
            return None
            
        return {
            'strike': selected['strike'],
            'bid': bid,
            'ask': ask,
            'mid_price': (bid + ask) / 2,
            'delta': selected.get('delta', selected.get('estimated_delta', target_delta)),
            'type': 'PUT',
            'source_line': selected.name + 2  # +2 because pandas index is 0-based and CSV has header row
        }
    
    def select_call_by_delta(self, df: pd.DataFrame, target_delta: float, 
                            spot_price: float, target_expiry: str = None) -> Optional[Dict]:
        """Select call option closest to target delta"""
        # Handle different ways calls are identified in the data
        if 'type' in df.columns:
            calls = df[df['type'].str.upper() == 'CALL'].copy()
        else:
            # Fallback: assume calls are identified in contractID or symbol
            calls = df[df.get('contractid', df.get('contractID', '')).str.contains('C', na=False)].copy()
            
        # Filter by target expiration date if specified (find nearest expiry within 3 days)
        if target_expiry and 'expiration' in calls.columns:
            target_date = datetime.strptime(target_expiry, '%Y-%m-%d')
            available_expiries = calls['expiration'].unique()
            
            # Find the closest expiry within 3 days of target
            best_expiry = None
            min_diff = 4  # More than 3 days
            
            for expiry_str in available_expiries:
                if expiry_str == 'expiration':  # Skip header
                    continue
                try:
                    expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
                    diff = abs((expiry_date - target_date).days)
                    if diff < min_diff:
                        min_diff = diff
                        best_expiry = expiry_str
                except:
                    continue
            
            if best_expiry:
                calls = calls[calls['expiration'] == best_expiry]
            else:
                # Fallback: use first available weekly expiry (within 14 days)
                calls = calls[calls['expiration'] != 'expiration']  # Remove header if present
            
        calls = calls[calls['strike'] > spot_price * 0.95]  # At least 5% OTM
        
        if calls.empty:
            return None
        
        if 'delta' in calls.columns:
            calls['abs_delta'] = abs(calls['delta'])
            calls['delta_diff'] = abs(calls['abs_delta'] - target_delta)
            selected = calls.loc[calls['delta_diff'].idxmin()]
        else:
            # Estimate delta based on moneyness
            calls['moneyness'] = calls['strike'] / spot_price
            calls['estimated_delta'] = np.maximum(0.1, np.minimum(0.8, 
                                                 0.8 - 0.3 * (calls['moneyness'] - 1)))
            calls['delta_diff'] = abs(calls['estimated_delta'] - target_delta)
            selected = calls.loc[calls['delta_diff'].idxmin()]
        
        bid = selected.get('bid', 0)
        ask = selected.get('ask', selected.get('lastprice', 0))
        
        if bid <= 0 or ask <= 0:
            return None
            
        return {
            'strike': selected['strike'],
            'bid': bid,
            'ask': ask,
            'mid_price': (bid + ask) / 2,
            'delta': selected.get('delta', selected.get('estimated_delta', target_delta)),
            'type': 'CALL',
            'source_line': selected.name + 2  # +2 because pandas index is 0-based and CSV has header row
        }
    
    def load_stock_prices(self, data_dir: str) -> None:
        """Load daily adjusted stock prices"""
        stock_file = os.path.join(data_dir, self.symbol, f"{self.symbol}_daily_adjusted.csv")
        
        if os.path.exists(stock_file):
            print(f"Loading stock prices from {stock_file}")
            df = pd.read_csv(stock_file)
            
            # Convert timestamp to date and adjusted_close to float
            df['date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
            df['adjusted_close'] = pd.to_numeric(df['adjusted_close'], errors='coerce')
            
            # Create date -> price mapping
            for _, row in df.iterrows():
                self.stock_prices[row['date']] = row['adjusted_close']
            
            print(f"Loaded {len(self.stock_prices)} daily stock prices")
        else:
            print(f"Warning: No daily adjusted stock data found at {stock_file}")
            print("Will estimate spot price from options data as fallback")
    
    def apply_split_adjustments(self) -> None:
        """Apply split adjustments to options data to match adjusted stock prices"""
        if not self.stock_prices or not self.options_data:
            return
        
        print("Checking for split adjustments needed...")
        
        # Find a common date to compare
        common_dates = set(self.stock_prices.keys()) & set(self.options_data.keys())
        if not common_dates:
            return
        
        sample_date = list(common_dates)[0]
        stock_price = self.stock_prices[sample_date]
        options_df = self.options_data[sample_date]
        
        # Compare median strike to stock price to detect split ratio
        median_strike = options_df['strike'].median()
        ratio = median_strike / stock_price
        
        # If ratio > 2, likely a stock split occurred
        if ratio > 2:
            split_factor = round(ratio)
            print(f"Detected {split_factor}:1 stock split - adjusting options data")
            
            # Apply split adjustment to all options data
            for date, df in self.options_data.items():
                df['strike'] = df['strike'] / split_factor
                df['bid'] = df['bid'] / split_factor
                df['ask'] = df['ask'] / split_factor
                if 'lastprice' in df.columns:
                    df['lastprice'] = df['lastprice'] / split_factor
                if 'mark' in df.columns:
                    df['mark'] = df['mark'] / split_factor
            
            print(f"Applied {split_factor}:1 split adjustment to {len(self.options_data)} days of options data")
        else:
            print("No split adjustment needed")
    
    def get_spot_price(self, date: str) -> float:
        """Get spot price from daily adjusted stock data, fallback to options estimation"""
        # First try to get actual adjusted close price
        if date in self.stock_prices:
            return self.stock_prices[date]
        
        # Fallback: estimate from options data
        if date not in self.options_data:
            return None
            
        df = self.options_data[date]
        
        # Simple estimation: use the strike price closest to where bid-ask spread is tightest
        df['spread'] = df['ask'] - df['bid']
        df['spread_ratio'] = df['spread'] / df['ask']
        
        reasonable_spreads = df[df['spread_ratio'] < 0.1]  # Less than 10% spread
        
        if reasonable_spreads.empty:
            reasonable_spreads = df
        
        mid_idx = len(reasonable_spreads) // 2
        return reasonable_spreads.iloc[mid_idx]['strike']
    
    def buy_stock(self, date: str, price: float, shares: int) -> None:
        """Buy shares of stock (from put assignment)"""
        cost = price * shares
        
        # Update position
        if self.stock_shares > 0:
            # Calculate new weighted average cost basis
            total_cost = self.stock_basis * self.stock_shares + cost
            self.stock_shares += shares
            self.stock_basis = total_cost / self.stock_shares
        else:
            self.stock_shares = shares
            self.stock_basis = price
            
        print(f"{date}: ASSIGNED {shares} shares @ ${price:.2f} (basis: ${self.stock_basis:.2f})")
        
        # Switch to covered call mode
        self.strategy_mode = "CALL"
        print(f"  -> Switching to COVERED CALL mode")
    
    def sell_stock(self, date: str, price: float, shares: int) -> None:
        """Sell shares of stock (from call assignment)"""
        if shares > self.stock_shares:
            shares = self.stock_shares
            
        if shares > 0:
            realized_pnl = (price - self.stock_basis) * shares
            
            self.stock_shares -= shares
            
            print(f"{date}: CALLED AWAY {shares} shares @ ${price:.2f} (P&L: ${realized_pnl:.0f})")
            
            # Reset basis if no shares left
            if self.stock_shares == 0:
                self.stock_basis = 0
                # Switch to cash secured put mode
                self.strategy_mode = "PUT"
                print(f"  -> Switching to CASH SECURED PUT mode")
    
    def execute_trade(self, date: str, action: str, option_type: str, strike: float, 
                     price: float, expiry: str, quantity: int = 1, source_line: int = None) -> None:
        """Record a trade execution"""
        trade = {
            'date': date,
            'action': action,  # 'sell_open', 'buy_close'
            'type': option_type,  # 'PUT', 'CALL'
            'strike': strike,
            'price': price,
            'expiry': expiry,
            'quantity': quantity,
            'premium': price * quantity * 100,  # Options are per 100 shares
            'source_line': source_line
        }
        
        self.trades.append(trade)
        
        if action == 'sell_open':
            # Add to positions
            self.positions.append({
                'type': option_type,
                'strike': strike,
                'expiry': expiry,
                'quantity': quantity,
                'entry_price': price,
                'entry_date': date,
                'source_line': source_line
            })
            # Collect premium
            self.current_capital += trade['premium']
            print(f"{date}: SELL {quantity} {strike}{option_type[0]} exp {expiry} @ ${price:.2f} "
                  f"(+${trade['premium']:.0f} premium)")
            
        elif action == 'buy_close':
            # Remove from positions and pay to close
            self.current_capital -= trade['premium']
            # Find and remove the position
            for i, pos in enumerate(self.positions):
                if (pos['strike'] == strike and pos['expiry'] == expiry and 
                    pos['type'] == option_type):
                    entry_premium = pos['entry_price'] * pos['quantity'] * 100
                    pnl = entry_premium - trade['premium']
                    print(f"{date}: BUY {quantity} {strike}{option_type[0]} exp {expiry} @ ${price:.2f} "
                          f"(-${trade['premium']:.0f}, P&L: ${pnl:.0f})")
                    self.positions.pop(i)
                    break
    
    def run_strategy(self, start_date: str = '2024-01-01', 
                    end_date: str = '2024-12-31') -> Dict:
        """
        Run the hybrid put/covered call strategy
        
        Returns:
            Dictionary with strategy performance metrics
        """
        print(f"Running Hybrid Put/Call Strategy for {self.symbol}")
        print(f"Target Delta: {self.target_delta}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Starting Mode: {self.strategy_mode}")
        print("-" * 60)
        
        # Get all Fridays in the period
        fridays = self.get_trading_fridays(start_date, end_date)
        
        if not fridays:
            raise ValueError("No trading days found with options data")
        
        for friday in fridays:
            spot_price = self.get_spot_price(friday)
            if not spot_price:
                continue
            
            # Track weekly data at the start of each Friday
            week_start_capital = self.current_capital
            week_start_stock_value = self.stock_shares * spot_price if self.stock_shares > 0 else 0
            week_start_total = week_start_capital + week_start_stock_value
                
            print(f"\n{friday} (Friday) - {self.symbol} @ ${spot_price:.2f} [{self.strategy_mode} mode]")
            
            # Handle expiring positions (positions expiring today)
            expiring_positions = [pos for pos in self.positions 
                                if pos['expiry'] == friday]
            
            for pos in expiring_positions:
                if pos['type'] == 'PUT':
                    # Cash secured put expiring
                    if spot_price > pos['strike']:
                        print(f"  PUT {pos['strike']}P expires worthless (profit)")
                        # Remove position without cost
                        self.positions = [p for p in self.positions if not 
                                        (p['strike'] == pos['strike'] and p['expiry'] == pos['expiry'] and p['type'] == pos['type'])]
                    else:
                        # Put assigned - buy stock at strike price
                        shares_assigned = pos['quantity'] * 100
                        self.buy_stock(friday, pos['strike'], shares_assigned)
                        # Remove position
                        self.positions = [p for p in self.positions if not 
                                        (p['strike'] == pos['strike'] and p['expiry'] == pos['expiry'] and p['type'] == pos['type'])]
                
                elif pos['type'] == 'CALL':
                    # Covered call expiring
                    if spot_price <= pos['strike']:
                        print(f"  CALL {pos['strike']}C expires worthless (profit)")
                        # Remove position without cost
                        self.positions = [p for p in self.positions if not 
                                        (p['strike'] == pos['strike'] and p['expiry'] == pos['expiry'] and p['type'] == pos['type'])]
                    else:
                        # Call assigned - stock called away
                        shares_called = pos['quantity'] * 100
                        if shares_called <= self.stock_shares:
                            self.sell_stock(friday, pos['strike'], shares_called)
                        # Remove position
                        self.positions = [p for p in self.positions if not 
                                        (p['strike'] == pos['strike'] and p['expiry'] == pos['expiry'] and p['type'] == pos['type'])]
            
            # Roll existing positions (close current, open new)
            positions_to_roll = [pos for pos in self.positions 
                               if pos['expiry'] not in [friday]]  # Not expiring today
            
            for pos in positions_to_roll:
                # Close current position
                df = self.options_data[friday]
                if pos['type'] == 'PUT':
                    if 'type' in df.columns:
                        close_options = df[(df['strike'] == pos['strike']) & (df['type'].str.upper() == 'PUT')]
                    else:
                        close_options = df[(df['strike'] == pos['strike']) & 
                                         (df.get('contractid', df.get('contractID', '')).str.contains('P', na=False))]
                else:
                    if 'type' in df.columns:
                        close_options = df[(df['strike'] == pos['strike']) & (df['type'].str.upper() == 'CALL')]
                    else:
                        close_options = df[(df['strike'] == pos['strike']) & 
                                         (df.get('contractid', df.get('contractID', '')).str.contains('C', na=False))]
                    
                if not close_options.empty:
                    close_price = close_options.iloc[0]['ask']  # Pay ask to close
                    self.execute_trade(friday, 'buy_close', pos['type'], pos['strike'], 
                                     close_price, pos['expiry'])
            
            # Open new weekly position based on current mode
            next_friday = self.find_next_friday_expiry(friday)
            if next_friday:
                df = self.options_data[friday]
                
                if self.strategy_mode == "PUT":
                    # Sell cash secured put
                    selected_option = self.select_put_by_delta(df, self.target_delta, spot_price, next_friday)
                    
                    if selected_option:
                        # Calculate maximum contracts based on available capital
                        capital_per_contract = selected_option['strike'] * 100
                        max_contracts = int(self.current_capital / capital_per_contract)
                        
                        if max_contracts > 0:
                            self.execute_trade(friday, 'sell_open', 'PUT', selected_option['strike'],
                                             selected_option['bid'], next_friday, max_contracts, selected_option.get('source_line'))
                        else:
                            print(f"  Insufficient capital for put (need ${capital_per_contract:.0f})")
                
                elif self.strategy_mode == "CALL" and self.stock_shares >= 100:
                    # Sell covered call
                    selected_option = self.select_call_by_delta(df, self.target_delta, spot_price, next_friday)
                    
                    if selected_option:
                        # Calculate maximum contracts based on stock holdings
                        max_contracts = self.stock_shares // 100  # 1 contract per 100 shares
                        if max_contracts > 0:
                            self.execute_trade(friday, 'sell_open', 'CALL', selected_option['strike'],
                                             selected_option['bid'], next_friday, max_contracts, selected_option.get('source_line'))
            
            # Calculate weekly P&L and summary
            week_end_capital = self.current_capital
            week_end_stock_value = self.stock_shares * spot_price if self.stock_shares > 0 else 0
            week_end_total = week_end_capital + week_end_stock_value
            week_pnl = week_end_total - week_start_total
            
            # Track weekly trades for this period
            week_trades = [t for t in self.trades if t['date'] == friday]
            
            # Determine current position and capital allocated
            current_positions = []
            total_capital_allocated = 0
            
            for pos in self.positions:
                pos_desc = f"{pos['quantity']}x {pos['strike']:.0f}{pos['type'][0]} @ ${pos['entry_price']:.2f}"
                current_positions.append(pos_desc)
                
                # Calculate capital allocated (cash secured for puts, stock value for calls)
                if pos['type'] == 'PUT':
                    capital_allocated = pos['strike'] * pos['quantity'] * 100
                else:  # CALL
                    capital_allocated = spot_price * pos['quantity'] * 100  # Value of covered stock
                total_capital_allocated += capital_allocated
            
            position_desc = ", ".join(current_positions) if current_positions else "None"
            
            # Get expiration price and option closing prices for positions traded this week
            expiry_date = None
            expiry_price = None
            option_closing_prices = []
            weekly_option_pnl = 0
            
            # Check positions that were opened this week (from trades)
            week_positions = []
            week_source_info = []  # Track source info for this week
            
            # Add positions that were opened this week (from trades)
            for trade in week_trades:
                if trade['action'] == 'sell_open':
                    position = {
                        'strike': trade['strike'],
                        'expiry': trade['expiry'],
                        'quantity': trade['quantity'],
                        'entry_price': trade['price'],
                        'entry_date': friday,
                        'type': trade['type'],
                        'source_line': trade.get('source_line', 'unknown')
                    }
                    week_positions.append(position)
                    # Track source info for current positions (opening line only)
                    open_line = trade.get('source_line', 'unknown')
                    base_filename = self.data_sources.get(friday, 'unknown')
                    source_info = f"{base_filename}:{open_line}"
                    week_source_info.append(source_info)
            
            # Check if positions expired this week (before they were removed from self.positions)
            if expiring_positions:
                # Use the expiry date from expiring positions
                first_expiring_pos = expiring_positions[0]
                expiry_date = first_expiring_pos['expiry']
                if expiry_date in self.stock_prices:
                    expiry_price = self.stock_prices[expiry_date]
            elif week_positions:
                # Use the expiry date from the first position to get stock closing price
                first_pos_expiry = week_positions[0]['expiry']
                if first_pos_expiry:
                    expiry_date = first_pos_expiry
                    if first_pos_expiry in self.stock_prices:
                        expiry_price = self.stock_prices[first_pos_expiry]
            
            # Add ALL expired positions (regardless of when they were opened)
            for pos in expiring_positions:
                week_positions.append(pos)
                
            if week_positions:
                # Calculate option P&L for positions opened this week
                for pos in week_positions:
                    if pos.get('entry_date') == friday and pos['expiry'] in self.options_data:
                        # Get option closing price at expiration
                        expiry_df = self.options_data[pos['expiry']]
                        
                        # Find the same option at expiration
                        option_type = pos.get('type', 'PUT')  # Default to PUT for older position tracking
                        if option_type == 'PUT':
                            if 'type' in expiry_df.columns:
                                closing_options = expiry_df[(expiry_df['strike'] == pos['strike']) & (expiry_df['type'].str.upper() == 'PUT')]
                            else:
                                closing_options = expiry_df[(expiry_df['strike'] == pos['strike']) & 
                                                           (expiry_df.get('contractid', expiry_df.get('contractID', '')).str.contains('P', na=False))]
                        else:  # CALL
                            if 'type' in expiry_df.columns:
                                closing_options = expiry_df[(expiry_df['strike'] == pos['strike']) & (expiry_df['type'].str.upper() == 'CALL')]
                            else:
                                closing_options = expiry_df[(expiry_df['strike'] == pos['strike']) & 
                                                           (expiry_df.get('contractid', expiry_df.get('contractID', '')).str.contains('C', na=False))]
                        
                        if not closing_options.empty:
                            # At expiration, options are worth exactly their intrinsic value
                            if pos['expiry'] in self.stock_prices:
                                stock_price_at_expiry = self.stock_prices[pos['expiry']]
                                if option_type == 'PUT':
                                    closing_price = max(0, pos['strike'] - stock_price_at_expiry)
                                else:  # CALL
                                    closing_price = max(0, stock_price_at_expiry - pos['strike'])
                            else:
                                # Fallback to market price if no stock data
                                closing_price = (closing_options.iloc[0]['bid'] + closing_options.iloc[0]['ask']) / 2
                            
                            option_closing_prices.append(f"${closing_price:.2f}")
                            # P&L = entry_price - closing_price (since we sold the option)
                            option_pnl = (pos['entry_price'] - closing_price) * pos['quantity'] * 100
                            weekly_option_pnl += option_pnl
                            
                            # Track source info: opening line + closing line
                            open_line = pos.get('source_line', 'unknown')
                            close_line = closing_options.iloc[0].name + 2  # +2 for 0-based index and CSV header
                            base_filename = self.data_sources.get(pos['expiry'], 'unknown')
                            source_info = f"{base_filename}:{open_line}-{close_line}"
                            week_source_info.append(source_info)
                        else:
                            option_closing_prices.append("N/A")
                            # Track source info even if no closing data found
                            open_line = pos.get('source_line', 'unknown')
                            base_filename = self.data_sources.get(pos['expiry'], 'unknown')
                            source_info = f"{base_filename}:{open_line}-N/A"
                            week_source_info.append(source_info)
            
            option_close_str = ", ".join(option_closing_prices) if option_closing_prices else "N/A"
            
            # Weekly summary entry
            weekly_entry = {
                'date': friday,
                'spot_price': spot_price,
                'expiry_date': expiry_date,
                'expiry_price': expiry_price,
                'option_close_price': option_close_str,
                'option_week_pnl': weekly_option_pnl,
                'strategy_mode': self.strategy_mode,
                'positions': position_desc,
                'trades_count': len(week_trades),
                'week_pnl': week_pnl,
                'cumulative_pnl': week_end_total - self.initial_capital,
                'total_value': week_end_total,
                'cash_balance': week_end_capital,
                'stock_value': week_end_stock_value,
                'stock_shares': self.stock_shares,
                'capital_allocated': total_capital_allocated,
                'source': self._get_source_info(friday, week_source_info)
            }
            self.weekly_summary.append(weekly_entry)
        
        # Calculate final performance
        return self.calculate_performance()
    
    def calculate_performance(self) -> Dict:
        """Calculate strategy performance metrics"""
        # Calculate stock position value at final price (using last available price)
        final_stock_value = 0
        if self.stock_shares > 0:
            last_date = max(self.options_data.keys())
            final_price = self.get_spot_price(last_date)
            final_stock_value = self.stock_shares * final_price
        
        total_premium_collected = sum(t['premium'] for t in self.trades 
                                    if t['action'] == 'sell_open')
        total_premium_paid = sum(t['premium'] for t in self.trades 
                               if t['action'] == 'buy_close')
        
        net_premium = total_premium_collected - total_premium_paid
        final_capital = self.current_capital + final_stock_value
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # Count trades by type
        put_trades = sum(1 for t in self.trades if t['type'] == 'PUT')
        call_trades = sum(1 for t in self.trades if t['type'] == 'CALL')
        
        performance = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'cash_balance': self.current_capital,
            'stock_value': final_stock_value,
            'stock_shares': self.stock_shares,
            'stock_basis': self.stock_basis,
            'final_mode': self.strategy_mode,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'net_premium': net_premium,
            'total_trades': len(self.trades),
            'put_trades': put_trades,
            'call_trades': call_trades,
            'premium_collected': total_premium_collected,
            'premium_paid': total_premium_paid
        }
        
        return performance
    
    def _get_source_info(self, date: str, week_source_info: list = None) -> str:
        """Get source information with line numbers for both open and close"""
        base_filename = self.data_sources.get(date, 'unknown')
        
        # If we have detailed source info from position tracking, use it
        if week_source_info:
            # Use the first source info entry (or combine if multiple)
            return week_source_info[0] if week_source_info else f"{base_filename}:data"
        
        # If we have active positions, use the opening line number only
        if self.positions:
            first_position = self.positions[0]
            line_num = first_position.get('source_line')
            if line_num:
                return f"{base_filename}:{line_num}"
        
        # Fallback to generic data reference
        return f"{base_filename}:data"
    
    def print_weekly_summary(self) -> None:
        """Print weekly position and P&L summary table"""
        if not self.weekly_summary:
            return
        
        print("\n" + "="*255)
        print("WEEKLY POSITION AND P&L SUMMARY")
        print("="*255)
        print(f"{'Date':<12} {'Spot $':<8} {'Expiry':<12} {'Exp $':<8} {'Mode':<6} {'Positions':<25} {'Opt Close':<12} {'Opt P&L':<10} {'Week P&L':<10} {'Cum P&L':<10} {'Total Value':<12} {'Cash':<10} {'Capital Alloc':<13} {'Source':<40}")
        print("-"*255)
        
        for entry in self.weekly_summary:
            expiry_date_str = f"{entry['expiry_date']:<12}" if entry['expiry_date'] else f"{'N/A':<12}"
            expiry_str = f"{entry['expiry_price']:<8.0f}" if entry['expiry_price'] else f"{'N/A':<8}"
            opt_close_str = f"{entry['option_close_price'][:11]:<12}" if entry['option_close_price'] else f"{'N/A':<12}"
            print(f"{entry['date']:<12} "
                  f"{entry['spot_price']:<8.0f} "
                  f"{expiry_date_str} "
                  f"{expiry_str} "
                  f"{entry['strategy_mode']:<6} "
                  f"{entry['positions'][:24]:<25} "
                  f"{opt_close_str} "
                  f"${entry['option_week_pnl']:<9,.0f} "
                  f"${entry['week_pnl']:<9,.0f} "
                  f"${entry['cumulative_pnl']:<9,.0f} "
                  f"${entry['total_value']:<11,.0f} "
                  f"${entry['cash_balance']:<9,.0f} "
                  f"${entry['capital_allocated']:<12,.0f} "
                  f"{entry['source'][:39]:<40}")
        
        print("-"*255)

def main():
    parser = argparse.ArgumentParser(description='Hybrid Put/Call Strategy Backtest')
    parser.add_argument('symbol', type=str, help='Stock ticker symbol (e.g., NVDA)')
    parser.add_argument('--delta', type=float, default=0.3, help='Target delta (default: 0.3)')
    parser.add_argument('--deltas', type=str, help='Comma-separated list of deltas to test (e.g., "0.2,0.3,0.4")')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing options data')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed trade output for multi-delta runs')
    parser.add_argument('--weekly', action='store_true', help='Show week-by-week position and P&L summary')
    
    args = parser.parse_args()
    
    # Determine deltas to test
    if args.deltas:
        deltas_to_test = [float(d.strip()) for d in args.deltas.split(',')]
        quiet_mode = True  # Always use quiet mode for multiple deltas
    else:
        deltas_to_test = [args.delta]
        quiet_mode = args.quiet
    
    # Store results for all deltas
    all_results = []
    strategy_instances = []  # Store strategy instances for weekly summaries
    
    try:
        # Test each delta
        for delta in deltas_to_test:
            # Initialize strategy
            strategy = HybridPutCallStrategy(args.symbol, delta, args.capital)
            
            # Suppress print statements for quiet mode
            if quiet_mode and len(deltas_to_test) > 1:
                # Temporarily redirect stdout for strategy execution
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
            
            try:
                # Load options data
                strategy.load_options_data(args.data_dir, args.start_date, args.end_date)
                
                # Run strategy
                performance = strategy.run_strategy(args.start_date, args.end_date)
                performance['delta'] = delta
                all_results.append(performance)
                strategy_instances.append(strategy)  # Store strategy instance
                
            finally:
                # Restore stdout if it was redirected
                if quiet_mode and len(deltas_to_test) > 1:
                    sys.stdout = old_stdout
        
        # Display results
        if len(deltas_to_test) > 1:
            # Table format for multiple deltas
            print("\n" + "="*100)
            print("HYBRID PUT/CALL STRATEGY - DELTA COMPARISON")
            print("="*100)
            print(f"Symbol: {args.symbol}")
            print(f"Period: {args.start_date} to {args.end_date}")
            print(f"Initial Capital: ${args.capital:,.0f}")
            print("-"*100)
            print(f"{'Delta':<8} {'Return %':<10} {'Final Capital':<15} {'Net Premium':<12} {'Total Trades':<12} {'Put Trades':<10} {'Call Trades':<11} {'Final Mode':<10}")
            print("-"*100)
            
            for result in all_results:
                print(f"{result['delta']:<8.1f} {result['total_return_pct']:<10.1f} ${result['final_capital']:<14,.0f} ${result['net_premium']:<11,.0f} {result['total_trades']:<12d} {result['put_trades']:<10d} {result['call_trades']:<11d} {result['final_mode']:<10}")
            
            print("-"*100)
            
            # Find best performing delta
            best_result = max(all_results, key=lambda x: x['total_return_pct'])
            print(f"Best Delta: {best_result['delta']:.1f} with {best_result['total_return_pct']:.1f}% return")
            
        else:
            # Single delta - detailed format
            performance = all_results[0]
            print("\n" + "="*60)
            print("STRATEGY PERFORMANCE SUMMARY")
            print("="*60)
            print(f"Symbol: {args.symbol}")
            print(f"Target Delta: {performance['delta']}")
            print(f"Initial Capital: ${performance['initial_capital']:,.0f}")
            print(f"Final Capital: ${performance['final_capital']:,.0f}")
            print(f"Cash Balance: ${performance['cash_balance']:,.0f}")
            print(f"Stock Value: ${performance['stock_value']:,.0f}")
            print(f"Stock Shares: {performance['stock_shares']}")
            if performance['stock_shares'] > 0:
                print(f"Stock Basis: ${performance['stock_basis']:,.2f}")
            print(f"Final Mode: {performance['final_mode']}")
            print(f"Total Return: {performance['total_return_pct']:.1f}%")
            print(f"Net Premium: ${performance['net_premium']:,.0f}")
            print(f"Total Trades: {performance['total_trades']} ({performance['put_trades']} puts, {performance['call_trades']} calls)")
            print(f"Premium Collected: ${performance['premium_collected']:,.0f}")
            print(f"Premium Paid: ${performance['premium_paid']:,.0f}")
            
            # Show weekly summary if requested
            if args.weekly and len(deltas_to_test) == 1:
                strategy_instances[0].print_weekly_summary()
        
        # Show weekly summary for multi-delta runs if requested (only for best delta)
        if args.weekly and len(deltas_to_test) > 1:
            best_idx = all_results.index(max(all_results, key=lambda x: x['total_return_pct']))
            print(f"\nWeekly summary for best performing delta ({all_results[best_idx]['delta']:.1f}):")
            strategy_instances[best_idx].print_weekly_summary()
        
    except Exception as e:
        print(f"Error running strategy: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())