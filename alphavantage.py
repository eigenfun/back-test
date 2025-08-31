import requests
import pandas as pd
import os
import argparse
import datetime
import time
from typing import Optional

def download_daily_adjusted_data(symbol: str, api_key: str, datatype: str = 'json') -> dict:
    """
    Download daily adjusted stock data from AlphaVantage API.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'IBM')
        api_key: AlphaVantage API key
        datatype: Response format ('json' or 'csv')
    
    Returns:
        Dictionary containing the API response
    """
    base_url = "https://www.alphavantage.co/query"
    
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol.upper(),
        'apikey': api_key,
        'datatype': datatype,
        'outputsize': 'full'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        if datatype == 'json':
            return response.json()
        else:
            return {'csv_content': response.text}
    
    except requests.RequestException as e:
        print(f"Error downloading daily adjusted data: {e}")
        return None

def download_options_data(symbol: str, api_key: str, date: Optional[str] = None, 
                         datatype: str = 'json') -> dict:
    """
    Download historical options data from AlphaVantage API for a specific symbol and date.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'IBM')
        api_key: AlphaVantage API key
        date: Date in YYYY-MM-DD format (optional, defaults to previous trading session)
        datatype: Response format ('json' or 'csv')
    
    Returns:
        Dictionary containing the API response
    """
    base_url = "https://www.alphavantage.co/query"
    
    params = {
        'function': 'HISTORICAL_OPTIONS',
        'symbol': symbol.upper(),
        'apikey': api_key,
        'datatype': datatype
    }
    
    if date:
        params['date'] = date
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        if datatype == 'json':
            return response.json()
        else:
            return {'csv_content': response.text}
    
    except requests.RequestException as e:
        print(f"Error downloading data: {e}")
        return None

def save_daily_adjusted_data(data: dict, symbol: str) -> str:
    """
    Save daily adjusted stock data to CSV file.
    
    Args:
        data: Daily adjusted data from API
        symbol: Stock ticker symbol
    
    Returns:
        Path to saved file
    """
    # Create data directory
    data_dir = os.path.join('data', symbol.upper())
    os.makedirs(data_dir, exist_ok=True)
    
    filename = f"{symbol.upper()}_daily_adjusted.csv"
    filepath = os.path.join(data_dir, filename)
    
    if 'csv_content' in data:
        # Save CSV content directly
        with open(filepath, 'w') as f:
            f.write(data['csv_content'])
    else:
        # Convert JSON to DataFrame and save
        if 'Time Series (Daily)' in data:
            # Convert time series data to DataFrame
            time_series = data['Time Series (Daily)']
            df_data = []
            
            for date, values in time_series.items():
                row = {'date': date}
                row.update(values)
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            # Sort by date
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df.to_csv(filepath, index=False)
        else:
            print(f"Unexpected data format: {list(data.keys())}")
            return None
    
    print(f"Daily adjusted data saved to: {filepath}")
    return filepath

def save_options_data(data: dict, symbol: str, date: str = None) -> str:
    """
    Save options data to CSV file.
    
    Args:
        data: Options data from API
        symbol: Stock ticker symbol
        date: Date string for filename
    
    Returns:
        Path to saved file
    """
    # Create data directory with year subfolder
    if date:
        year = date.split('-')[0]
        data_dir = os.path.join('data', symbol.upper(), year)
        filename = f"{symbol.upper()}_{date}_options.csv"
    else:
        # For latest data, use current year
        current_year = str(datetime.datetime.now().year)
        data_dir = os.path.join('data', symbol.upper(), current_year)
        filename = f"{symbol.upper()}_latest_options.csv"
    
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    if 'csv_content' in data:
        # Save CSV content directly
        with open(filepath, 'w') as f:
            f.write(data['csv_content'])
    else:
        # Convert JSON to DataFrame and save
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            df.to_csv(filepath, index=False)
        else:
            print(f"Unexpected data format: {list(data.keys())}")
            return None
    
    print(f"Data saved to: {filepath}")
    return filepath

def download_date_range(symbol: str, api_key: str, start_date: str, 
                       end_date: str, delay: float = 0.5, skip_weekends: bool = True) -> None:
    """
    Download options data for a range of dates.
    
    Args:
        symbol: Stock ticker symbol
        api_key: AlphaVantage API key
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        delay: Delay between API calls in seconds (free tier: 5 calls/min)
        skip_weekends: Skip Saturday and Sunday (default: True)
    """
    start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    current_date = start
    while current_date <= end:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Skip weekends if enabled (Saturday=5, Sunday=6)
        if skip_weekends and current_date.weekday() in [5, 6]:
            weekday_name = current_date.strftime('%A')
            print(f"\nSkipping {weekday_name} {date_str} (weekend)")
            current_date += datetime.timedelta(days=1)
            continue
            
        print(f"\nDownloading options data for {symbol} on {date_str}")
        
        # Download data for current date
        data = download_options_data(symbol, api_key, date_str, datatype='csv')
        
        if data and 'csv_content' in data:
            # Check if data contains actual options data (not just headers)
            lines = data['csv_content'].strip().split('\n')
            if len(lines) > 1:  # More than just header
                save_options_data(data, symbol, date_str)
                print(f"✓ Found options data for {date_str}")
            else:
                print(f"✗ No options data available for {date_str}")
        else:
            print(f"✗ Failed to download data for {date_str}")
        
        # Move to next date
        current_date += datetime.timedelta(days=1)
        
        # Add delay to respect API rate limits (free tier: 5 calls per minute)
        if current_date <= end:
            print(f"Waiting {delay} seconds (API rate limit)...")
            time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description='Download historical options and stock data from AlphaVantage API')
    parser.add_argument('symbol', type=str, help='Stock ticker symbol (e.g., IBM)')
    parser.add_argument('api_key', type=str, help='AlphaVantage API key')
    parser.add_argument('--date', type=str, help='Specific date (YYYY-MM-DD). If not provided, gets latest data')
    parser.add_argument('--start-date', type=str, help='Start date for range download (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for range download (YYYY-MM-DD)')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between API calls in seconds (default: 0.5)')
    parser.add_argument('--skip-weekends', action='store_true', default=True, help='Skip Saturday and Sunday (default: True)')
    parser.add_argument('--include-weekends', dest='skip_weekends', action='store_false', help='Include weekends in download')
    parser.add_argument('--datatype', type=str, choices=['json', 'csv'], default='csv', help='Response format')
    parser.add_argument('--daily-adjusted', action='store_true', help='Download daily adjusted stock data instead of options')
    
    args = parser.parse_args()
    
    # Validate date range arguments
    if (args.start_date or args.end_date) and not (args.start_date and args.end_date):
        print("Error: Both --start-date and --end-date must be provided for range download")
        return
    
    if args.date and (args.start_date or args.end_date):
        print("Error: Cannot use --date with date range options")
        return
    
    # Validate date formats
    try:
        if args.date:
            datetime.datetime.strptime(args.date, '%Y-%m-%d')
        if args.start_date:
            datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
        if args.end_date:
            datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD. {e}")
        return
    
    if args.daily_adjusted:
        print(f"Starting AlphaVantage daily adjusted stock data download for {args.symbol}")
        
        # Download daily adjusted data (full historical dataset)
        data = download_daily_adjusted_data(args.symbol, args.api_key, args.datatype)
        
        if data:
            saved_path = save_daily_adjusted_data(data, args.symbol)
            if saved_path:
                print(f"✓ Successfully downloaded and saved daily adjusted stock data")
            else:
                print("✗ Failed to save data")
        else:
            print("✗ Failed to download data")
    else:
        print(f"Starting AlphaVantage options data download for {args.symbol}")
        
        if args.start_date and args.end_date:
            # Download date range
            skip_msg = "(skipping weekends)" if args.skip_weekends else "(including weekends)"
            print(f"Downloading data from {args.start_date} to {args.end_date} {skip_msg}")
            download_date_range(args.symbol, args.api_key, args.start_date, args.end_date, args.delay, args.skip_weekends)
        else:
            # Download single date or latest
            data = download_options_data(args.symbol, args.api_key, args.date, args.datatype)
            
            if data:
                saved_path = save_options_data(data, args.symbol, args.date)
                if saved_path:
                    print(f"✓ Successfully downloaded and saved options data")
                else:
                    print("✗ Failed to save data")
            else:
                print("✗ Failed to download data")

if __name__ == '__main__':
    main()