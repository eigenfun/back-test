#!/usr/bin/env python3

import pandas as pd
import os
import glob
from datetime import datetime

def test_data_loading(symbol, year):
    """Test data loading for any symbol/year"""
    print(f"Testing {symbol} {year} data...")
    
    # Load data
    pattern = os.path.join('data', symbol, str(year), '*.csv')
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} files")
    if not files:
        return
    
    # Check date range
    dates = []
    fridays = []
    for file in files:
        filename = os.path.basename(file)
        date_str = filename.split('_')[1]
        dates.append(date_str)
        
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        if date_obj.weekday() == 4:  # Friday
            fridays.append(date_str)
    
    dates.sort()
    fridays.sort()
    
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Found {len(fridays)} Fridays")
    print(f"Sample Fridays: {fridays[:5]}")
    
    # Test file structure
    sample_file = files[0]
    print(f"\nSample file: {os.path.basename(sample_file)}")
    df = pd.read_csv(sample_file)
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")
    
    # Check for puts/calls
    if 'type' in df.columns:
        puts = len(df[df['type'].str.upper() == 'PUT'])
        calls = len(df[df['type'].str.upper() == 'CALL']) 
        print(f"Puts: {puts}, Calls: {calls}")
    else:
        contractids = df.get('contractID', df.get('contractid', pd.Series()))
        if not contractids.empty:
            puts = contractids.str.contains('P', na=False).sum()
            calls = contractids.str.contains('C', na=False).sum()
            print(f"Puts (from contractID): {puts}, Calls: {calls}")

if __name__ == '__main__':
    # Test available data
    symbols = ['AAPL', 'NVDA', 'TSLA', 'MSFT']
    years = [2023, 2024]
    
    for symbol in symbols:
        for year in years:
            if os.path.exists(f'data/{symbol}/{year}'):
                test_data_loading(symbol, year)
                print("-" * 50)