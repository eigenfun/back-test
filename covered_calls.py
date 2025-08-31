import pandas as pd
import numpy as np
import os
import glob
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

class CoveredCallStrategy:
    def __init__(self, symbol: str, target_delta: float = 0.3, initial_capital: float = 100000):
        """
        Covered Call Strategy - Weekly options rolled on Fridays
        
        Args:
            symbol: Stock ticker symbol
            target_delta: Target delta for call selection (e.g., 0.3 for 30 delta)
            initial_capital: Starting capital for the strategy
        """
        self.symbol = symbol.upper()
        self.target_delta = target_delta
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Strategy state
        self.stock_shares = 0        # Shares of stock owned
        self.stock_basis = 0         # Average cost basis of stock
        self.positions = []          # List of open call positions
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
                    
                    # Filter for calls only
                    if 'type' in df.columns:
                        df = df[df['type'].str.upper() == 'CALL']
                    elif 'contractname' in df.columns:
                        df = df[df['contractname'].str.contains('C')]
                    else:
                        # Fallback: assume calls are identified in contractID or symbol
                        df = df[df.get('contractid', df.get('contractID', '')).str.contains('C', na=False)]
                    
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
    
    def select_call_by_delta(self, df: pd.DataFrame, target_delta: float, 
                            spot_price: float, target_expiry: str = None) -> Optional[Dict]:
        """
        Select call option closest to target delta
        
        Args:
            df: Options data for the day
            target_delta: Target delta (absolute value, e.g., 0.3)
            spot_price: Current stock price
        
        Returns:
            Dictionary with selected option details
        """
        if df.empty:
            return None
            
        # Filter by target expiration date if specified (find nearest expiry within 3 days)
        if target_expiry and 'expiration' in df.columns:
            target_date = datetime.strptime(target_expiry, '%Y-%m-%d')
            available_expiries = df['expiration'].unique()
            
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
                df = df[df['expiration'] == best_expiry]
            else:
                # Fallback: use first available weekly expiry (within 14 days)
                df = df[df['expiration'] != 'expiration']  # Remove header if present
        
        # Filter for call options with strikes above current price (OTM calls)
        calls = df[df['strike'] > spot_price * 0.95].copy()  # At least 5% OTM
        
        if calls.empty:
            return None
        
        # If delta column exists, use it directly
        if 'delta' in calls.columns:
            calls['abs_delta'] = abs(calls['delta'])
            calls['delta_diff'] = abs(calls['abs_delta'] - target_delta)
            selected = calls.loc[calls['delta_diff'].idxmin()]
        else:
            # Estimate delta based on moneyness (simplified approach)
            calls['moneyness'] = calls['strike'] / spot_price
            # Rough approximation: delta ≈ max(0.1, 0.8 - 0.3 * (moneyness - 1))
            calls['estimated_delta'] = np.maximum(0.1, np.minimum(0.8, 
                                                 0.8 - 0.3 * (calls['moneyness'] - 1)))
            calls['delta_diff'] = abs(calls['estimated_delta'] - target_delta)
            selected = calls.loc[calls['delta_diff'].idxmin()]
        
        # Ensure we have bid/ask prices
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
    
    def _get_expiry_stock_price(self, expiry_date: str) -> Optional[float]:
        """Get stock price on expiration date with fallback to nearest available date"""
        if not expiry_date:
            return None
            
        # First try exact match
        if expiry_date in self.stock_prices:
            return self.stock_prices[expiry_date]
        
        # Try to find the closest date (within ±3 business days)
        import datetime
        try:
            target_date = datetime.datetime.strptime(expiry_date, '%Y-%m-%d')
        except ValueError:
            return None
            
        # Look for nearby dates
        for days_offset in range(1, 4):  # Check ±1-3 days
            # Try earlier dates first (market closed on expiry might have price from previous day)
            for direction in [-1, 1]:
                check_date = target_date + datetime.timedelta(days=direction * days_offset)
                check_date_str = check_date.strftime('%Y-%m-%d')
                if check_date_str in self.stock_prices:
                    print(f"  Warning: Using stock price from {check_date_str} for expiry {expiry_date}")
                    return self.stock_prices[check_date_str]
        
        print(f"  Warning: No stock price found for expiry date {expiry_date} or nearby dates")
        return None
    
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
        
        # Find strikes with reasonable spreads
        reasonable_spreads = df[df['spread_ratio'] < 0.1]  # Less than 10% spread
        
        if reasonable_spreads.empty:
            reasonable_spreads = df
        
        # Estimate spot as average of strikes weighted by volume or use middle strikes
        mid_idx = len(reasonable_spreads) // 2
        return reasonable_spreads.iloc[mid_idx]['strike']
    
    def buy_stock(self, date: str, price: float, shares: int) -> None:
        """Buy shares of stock"""
        cost = price * shares
        if cost > self.current_capital:
            # Can't afford full purchase, buy what we can
            shares = int(self.current_capital / price)
            cost = price * shares
            
        if shares > 0:
            # Update position
            if self.stock_shares > 0:
                # Calculate new weighted average cost basis
                total_cost = self.stock_basis * self.stock_shares + cost
                self.stock_shares += shares
                self.stock_basis = total_cost / self.stock_shares
            else:
                self.stock_shares = shares
                self.stock_basis = price
                
            self.current_capital -= cost
            print(f"{date}: BUY {shares} shares @ ${price:.2f} (-${cost:.0f}, basis: ${self.stock_basis:.2f})")
    
    def sell_stock(self, date: str, price: float, shares: int) -> None:
        """Sell shares of stock"""
        if shares > self.stock_shares:
            shares = self.stock_shares
            
        if shares > 0:
            proceeds = price * shares
            realized_pnl = (price - self.stock_basis) * shares
            
            self.stock_shares -= shares
            self.current_capital += proceeds
            
            print(f"{date}: SELL {shares} shares @ ${price:.2f} (+${proceeds:.0f}, P&L: ${realized_pnl:.0f})")
            
            # Reset basis if no shares left
            if self.stock_shares == 0:
                self.stock_basis = 0
    
    def execute_trade(self, date: str, action: str, strike: float, price: float, 
                     expiry: str, quantity: int = 1, source_line: int = None) -> None:
        """Record an options trade execution"""
        trade = {
            'date': date,
            'action': action,  # 'sell_open', 'buy_close'
            'strike': strike,
            'price': price,
            'expiry': expiry,
            'quantity': quantity,
            'premium': price * quantity * 100,  # Options are per 100 shares
        }
        
        self.trades.append(trade)
        
        if action == 'sell_open':
            # Add to positions
            self.positions.append({
                'strike': strike,
                'expiry': expiry,
                'quantity': quantity,
                'entry_price': price,
                'entry_date': date,
                'source_line': source_line
            })
            # Collect premium
            self.current_capital += trade['premium']
            print(f"{date}: SELL {quantity} {strike}C exp {expiry} @ ${price:.2f} "
                  f"(+${trade['premium']:.0f} premium)")
            
        elif action == 'buy_close':
            # Remove from positions and pay to close
            self.current_capital -= trade['premium']
            # Find and remove the position
            for i, pos in enumerate(self.positions):
                if pos['strike'] == strike and pos['expiry'] == expiry:
                    entry_premium = pos['entry_price'] * pos['quantity'] * 100
                    pnl = entry_premium - trade['premium']
                    print(f"{date}: BUY {quantity} {strike}C exp {expiry} @ ${price:.2f} "
                          f"(-${trade['premium']:.0f}, P&L: ${pnl:.0f})")
                    self.positions.pop(i)
                    break
    
    def run_strategy(self, start_date: str = '2024-01-01', 
                    end_date: str = '2024-12-31') -> Dict:
        """
        Run the covered call strategy
        
        Returns:
            Dictionary with strategy performance metrics
        """
        print(f"Running Covered Call Strategy for {self.symbol}")
        print(f"Target Delta: {self.target_delta}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print("-" * 60)
        
        # Get all Fridays in the period
        print(f"Finding trading Fridays between {start_date} and {end_date}...")
        fridays = self.get_trading_fridays(start_date, end_date)
        
        if not fridays:
            raise ValueError("No trading days found with options data")
        
        print(f"Found {len(fridays)} trading Fridays to process: {fridays[0]} to {fridays[-1]}")
        print(f"Processing weekly strategy execution...")
        
        last_friday = None
        
        for friday in fridays:
            spot_price = self.get_spot_price(friday)
            if not spot_price:
                continue
            
            # Track weekly data at the start of each Friday
            week_start_capital = self.current_capital
            week_start_stock_value = 0
            if self.stock_shares > 0:
                week_start_stock_value = self.stock_shares * spot_price
            week_start_total = week_start_capital + week_start_stock_value
                
            print(f"\n{friday} (Friday) - {self.symbol} @ ${spot_price:.2f}")
            print(f"  Week {fridays.index(friday) + 1} of {len(fridays)} | Capital: ${week_start_capital:,.0f} | Stock Shares: {self.stock_shares} | Active Positions: {len(self.positions)}")
            
            # First Friday: buy maximum initial stock position
            if friday == fridays[0]:
                # Buy maximum shares we can afford (rounded down to 100-share lots)
                max_shares = int(self.current_capital / spot_price)
                shares_to_buy = (max_shares // 100) * 100  # Round down to nearest 100
                if shares_to_buy >= 100:  # Need at least 100 shares for covered calls
                    self.buy_stock(friday, spot_price, shares_to_buy)
                else:
                    print(f"  Insufficient capital to buy 100 shares (need ${spot_price * 100:.0f})")
                    continue
            
            # Find positions that expired since last processing date (or today if expiry == today)
            current_date = datetime.strptime(friday, '%Y-%m-%d')
            if last_friday:
                last_date = datetime.strptime(last_friday, '%Y-%m-%d')
                # Check for positions that expired between last_friday (exclusive) and current friday (inclusive)
                expiring_positions = [pos for pos in self.positions 
                                    if last_date < datetime.strptime(pos['expiry'], '%Y-%m-%d') <= current_date]
                print(f"  Checking for positions that expired between {last_friday} and {friday}...")
            else:
                # First processing date - only check positions expiring today
                expiring_positions = [pos for pos in self.positions 
                                    if pos['expiry'] == friday]
                print(f"  Checking positions expiring on {friday}...")
            
            if expiring_positions:
                print(f"  Found {len(expiring_positions)} expiring position(s)")
            else:
                print(f"  No positions expiring this period")
            
            # Determine positions that were active during this week (before processing/removal)
            # Include both existing positions and newly opened positions
            week_active_positions = []
            total_capital_allocated = 0
            
            # Add all current positions (including those about to expire)
            for pos in self.positions:
                pos_desc = f"{pos['quantity']}x {pos['strike']:.0f}C @ ${pos['entry_price']:.2f}"
                week_active_positions.append(pos_desc)
                # Calculate capital allocated (stock value for covered calls)
                capital_allocated = spot_price * pos['quantity'] * 100  # Value of covered stock
                total_capital_allocated += capital_allocated
            
            # Add newly opened positions this week from trades
            for trade in week_trades:
                if trade['action'] == 'sell_open':
                    # Check if this position is not already in our list
                    trade_desc = f"{trade['quantity']}x {trade['strike']:.0f}C @ ${trade['price']:.2f}"
                    # Only add if we don't already have it (to avoid duplicates)
                    position_exists = False
                    for existing_desc in week_active_positions:
                        if trade_desc in existing_desc:
                            position_exists = True
                            break
                    if not position_exists:
                        week_active_positions.append(trade_desc)
                        capital_allocated = spot_price * trade['quantity'] * 100
                        total_capital_allocated += capital_allocated
            
            position_desc = ", ".join(week_active_positions) if week_active_positions else "None"
            
            for pos in expiring_positions:
                # Get the actual stock price on the expiration date
                expiry_stock_price = self._get_expiry_stock_price(pos['expiry'])
                if expiry_stock_price is None:
                    print(f"  Warning: No expiry price found for {pos['expiry']}, using current spot price as fallback")
                    expiry_stock_price = spot_price
                
                # Calculate the intrinsic value at expiration
                closing_price = max(0, expiry_stock_price - pos['strike'])
                # P&L = entry_price - closing_price (since we sold the option)
                position_pnl = (pos['entry_price'] - closing_price) * pos['quantity'] * 100
                
                if position_pnl >= 0:
                    # Position is profitable or breakeven - let it expire
                    if expiry_stock_price <= pos['strike']:
                        print(f"  CALL {pos['strike']}C expires worthless (profit: ${position_pnl:.0f})")
                    else:
                        print(f"  CALL {pos['strike']}C expires ITM but profitable (P&L: ${position_pnl:.0f})")
                        # Stock called away but we keep the profit
                        shares_called = pos['quantity'] * 100
                        if shares_called <= self.stock_shares:
                            self.sell_stock(friday, pos['strike'], shares_called)
                            print(f"    Stock called away at ${pos['strike']:.2f}")
                else:
                    # Position has negative P&L - close it to avoid further losses
                    close_cost = closing_price * pos['quantity'] * 100
                    self.current_capital -= close_cost
                    
                    print(f"  CALL {pos['strike']}C closed at ${closing_price:.2f} (loss: ${-position_pnl:.0f})")
                
                # Remove the position regardless of outcome
                self.positions = [p for p in self.positions if not 
                                (p['strike'] == pos['strike'] and p['expiry'] == pos['expiry'])]
            
            # Roll existing positions (close current, open new)
            positions_to_roll = [pos for pos in self.positions 
                               if pos['expiry'] not in [friday]]  # Not expiring today
            
            if positions_to_roll:
                print(f"  Rolling {len(positions_to_roll)} existing position(s)...")
                for pos in positions_to_roll:
                    # Close current position
                    df = self.options_data[friday]
                    close_options = df[df['strike'] == pos['strike']]
                    if not close_options.empty:
                        close_price = close_options.iloc[0]['ask']  # Pay ask to close
                        self.execute_trade(friday, 'buy_close', pos['strike'], 
                                         close_price, pos['expiry'])
            
            # Open new weekly position if we own stock
            if self.stock_shares >= 100:
                next_friday = self.find_next_friday_expiry(friday)
                print(f"  Looking for new call options expiring {next_friday}...")
                if next_friday:
                    df = self.options_data[friday]
                    selected_call = self.select_call_by_delta(df, self.target_delta, spot_price, next_friday)
                    
                    if selected_call:
                        # Calculate maximum position size based on stock owned
                        max_contracts = self.stock_shares // 100  # 1 contract per 100 shares
                        
                        if max_contracts > 0:
                            print(f"  Selected ${selected_call['strike']:.2f}C (delta: {selected_call['delta']:.2f}) for {max_contracts} contracts")
                            self.execute_trade(friday, 'sell_open', selected_call['strike'],
                                             selected_call['bid'], next_friday, max_contracts, selected_call.get('source_line'))
                        else:
                            print(f"  No stock available to cover calls")
                    else:
                        print(f"  No suitable call options found for target delta {self.target_delta}")
                else:
                    print(f"  No expiry date found for new calls")
            elif len(self.positions) == 0 and self.current_capital > 0:
                # No stock, no positions - buy maximum stock we can afford
                max_shares = int(self.current_capital / spot_price)
                # Round down to nearest 100 shares for covered calls
                shares_to_buy = (max_shares // 100) * 100
                if shares_to_buy >= 100:
                    self.buy_stock(friday, spot_price, shares_to_buy)
            
            # Calculate weekly P&L and summary
            week_end_capital = self.current_capital
            week_end_stock_value = 0
            if self.stock_shares > 0:
                week_end_stock_value = self.stock_shares * spot_price
            week_end_total = week_end_capital + week_end_stock_value
            week_pnl = week_end_total - week_start_total
            
            # Track weekly trades for this period
            week_trades = [t for t in self.trades if t['date'] == friday]
            
            # Get expiration price and option closing prices for current positions
            expiry_date = None
            expiry_price = None
            option_closing_prices = []
            weekly_option_pnl = 0
            
            # For positions that expired this week, show their opening price as spot and expiry price as exp
            position_open_price = spot_price  # Default to current Friday price
            expiry_date = None
            expiry_price = None
            
            if expiring_positions:
                # Use the expiry date from expiring positions
                first_expiring_pos = expiring_positions[0]
                expiry_date = first_expiring_pos['expiry']
                # Get the actual stock closing price on the expiration date
                expiry_price = self._get_expiry_stock_price(expiry_date)
                
                # Get the stock price from when this position was opened (for spot price)
                open_date = first_expiring_pos.get('entry_date')
                if open_date and open_date in self.stock_prices:
                    position_open_price = self.stock_prices[open_date]
            else:
                # No positions expired this week - don't show expiry info
                expiry_date = None
                expiry_price = None
                
            # Calculate option closing prices only for expired positions
            week_source_info = []  # Track source info for this week
            
            # Only process expired positions for closing prices
            for pos in expiring_positions:
                if pos['expiry'] in self.options_data:
                    # Get option closing price at expiration
                    expiry_df = self.options_data[pos['expiry']]
                    
                    # Find the same call option at expiration
                    closing_options = expiry_df[expiry_df['strike'] == pos['strike']]
                    
                    if not closing_options.empty:
                        # At expiration, options are worth exactly their intrinsic value
                        stock_price_at_expiry = self._get_expiry_stock_price(pos['expiry'])
                        if stock_price_at_expiry is not None:
                            closing_price = max(0, stock_price_at_expiry - pos['strike'])
                        else:
                            # Fallback to market price if no stock data
                            closing_price = (closing_options.iloc[0]['bid'] + closing_options.iloc[0]['ask']) / 2
                            print(f"  Warning: Using market price for closing {pos['strike']}C (no expiry stock price)")
                        
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
            
            # Add source info for current positions opened this week (opening line only)
            for pos in self.positions:
                if pos.get('entry_date') == friday:
                    open_line = pos.get('source_line', 'unknown')
                    base_filename = self.data_sources.get(friday, 'unknown')
                    source_info = f"{base_filename}:{open_line}"
                    week_source_info.append(source_info)
            
            option_close_str = ", ".join(option_closing_prices) if option_closing_prices else "N/A"
            
            # Weekly summary entry
            weekly_entry = {
                'date': friday,
                'spot_price': position_open_price,
                'expiry_date': expiry_date,
                'expiry_price': expiry_price,
                'option_close_price': option_close_str,
                'option_week_pnl': weekly_option_pnl,
                'positions': position_desc,
                'trades_count': len(week_trades),
                'week_pnl': week_pnl,
                'cumulative_pnl': week_end_total - self.initial_capital,
                'total_value': week_end_total,
                'stock_value': week_end_stock_value,
                'capital_allocated': total_capital_allocated,
                'source': self._get_source_info(friday, week_source_info)
            }
            self.weekly_summary.append(weekly_entry)
            
            # Print week summary
            print(f"  Week Summary: P&L ${week_pnl:,.0f} | Total Value: ${week_end_total:,.0f} | Stock: {self.stock_shares} shares | Positions: {len(self.positions)}")
            
            # Update last_friday for next iteration
            last_friday = friday
        
        # Calculate final performance
        print(f"\nStrategy execution completed! Processed {len(fridays)} weeks.")
        print(f"Final Total Value: ${week_end_total:,.0f} | Stock Shares: {self.stock_shares} | Total Trades: {len(self.trades)}")
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
        
        performance = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'cash_balance': self.current_capital,
            'stock_value': final_stock_value,
            'stock_shares': self.stock_shares,
            'stock_basis': self.stock_basis,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'net_premium': net_premium,
            'total_trades': len(self.trades),
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
        
        print("\n" + "="*245)
        print("WEEKLY POSITION AND P&L SUMMARY")
        print("="*245)
        print(f"{'Date':<12} {'Spot $':<8} {'Expiry':<12} {'Exp $':<8} {'Positions':<25} {'Opt Close':<12} {'Opt P&L':<10} {'Week P&L':<10} {'Cum P&L':<10} {'Total Value':<12} {'Capital Alloc':<13} {'Source':<40}")
        print("-"*245)
        
        for entry in self.weekly_summary:
            expiry_date_str = f"{entry['expiry_date']:<12}" if entry['expiry_date'] else f"{'N/A':<12}"
            expiry_str = f"{entry['expiry_price']:<8.0f}" if entry['expiry_price'] else f"{'N/A':<8}"
            opt_close_str = f"{entry['option_close_price'][:11]:<12}" if entry['option_close_price'] else f"{'N/A':<12}"
            print(f"{entry['date']:<12} "
                  f"{entry['spot_price']:<8.0f} "
                  f"{expiry_date_str} "
                  f"{expiry_str} "
                  f"{entry['positions'][:24]:<25} "
                  f"{opt_close_str} "
                  f"${entry['option_week_pnl']:<9,.0f} "
                  f"${entry['week_pnl']:<9,.0f} "
                  f"${entry['cumulative_pnl']:<9,.0f} "
                  f"${entry['total_value']:<11,.0f} "
                  f"${entry['capital_allocated']:<12,.0f} "
                  f"{entry['source'][:39]:<40}")
        
        print("-"*245)

def main():
    parser = argparse.ArgumentParser(description='Covered Call Strategy Backtest')
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
            strategy = CoveredCallStrategy(args.symbol, delta, args.capital)
            
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
            print("COVERED CALL STRATEGY - DELTA COMPARISON")
            print("="*100)
            print(f"Symbol: {args.symbol}")
            print(f"Period: {args.start_date} to {args.end_date}")
            print(f"Initial Capital: ${args.capital:,.0f}")
            print("-"*100)
            print(f"{'Delta':<8} {'Return %':<10} {'Final Capital':<15} {'Stock Value':<12} {'Stock Shares':<12} {'Total Trades':<12} {'Net Premium':<12}")
            print("-"*100)
            
            for result in all_results:
                print(f"{result['delta']:<8.1f} {result['total_return_pct']:<10.1f} ${result['final_capital']:<14,.0f} ${result['stock_value']:<11,.0f} {result['stock_shares']:<12d} {result['total_trades']:<12d} ${result['net_premium']:<11,.0f}")
            
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
            print(f"Total Return: {performance['total_return_pct']:.1f}%")
            print(f"Net Premium: ${performance['net_premium']:,.0f}")
            print(f"Total Trades: {performance['total_trades']}")
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