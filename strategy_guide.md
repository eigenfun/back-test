# Strategy Definition and Testing Guide

This guide explains how to define a custom trading strategy, run a backtest, and analyze the results within the options backtesting framework.

## 1. Defining a Strategy

To define a trading strategy, you need to create a Python class that inherits from the base `Strategy` class. This class will contain the logic for your trading strategy.

For this guide, we will use the example of a **Covered Call** strategy. A covered call is an options strategy where an investor holds a long position in an asset and writes (sells) call options on that same asset.

Here is the basic structure of a strategy class:

```python
from backtesting import Strategy

class CoveredCall(Strategy):
    def initialize(self):
        # Initialization logic goes here
        pass

    def on_data(self, data):
        # Trading logic goes here
        pass
```

### 1.1. The `initialize` Method

The `initialize` method is called once at the beginning of a backtest. You can use this method to set up any parameters or variables that your strategy will use.

For our Covered Call strategy, we might want to define the number of shares to hold and the desired delta for the call option we sell.

```python
class CoveredCall(Strategy):
    def initialize(self, shares=100, delta=0.3):
        self.shares = shares
        self.delta = delta
```

### 1.2. The `on_data` Method

The `on_data` method is the core of your strategy. It is called for each new data point in the historical data. This is where you will implement your trading logic.

The `data` object passed to this method will give you access to the current market data, including the price of the underlying asset and the options chain.

Here is a simplified implementation of the `on_data` method for our Covered Call strategy:

```python
class CoveredCall(Strategy):
    # ... (initialize method) ...

    def on_data(self, data):
        # Check if we already have a position
        if not self.portfolio.has_position(self.ticker):
            # Buy the underlying stock
            self.buy(self.ticker, self.shares)

        # Check if we have an open call option
        if not self.portfolio.has_open_option(self.ticker, 'call'):
            # Find the call option with the closest delta to our target
            options_chain = data.get_options_chain(self.ticker)
            target_option = options_chain.find_closest_delta('call', self.delta)

            # Sell the call option
            if target_option:
                self.sell_option(target_option)
```

## 2. Running a Backtest

Once you have defined your strategy, you can run a backtest to see how it would have performed on historical data.

To do this, you will need to:

1.  Instantiate the backtesting engine.
2.  Pass your strategy class to the engine.
3.  Specify the ticker symbol and the time period for the backtest.

Here is an example of how to run a backtest:

```python
from backtesting import Backtest
from strategies import CoveredCall

# Instantiate the backtest engine
backtest = Backtest(
    ticker='AAPL',
    strategy=CoveredCall,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Run the backtest
results = backtest.run()

# Print the results
print(results)
```

## 3. Analyzing the Results

The `run` method of the backtest engine will return a `results` object that contains a summary of the strategy's performance.

This object will include a variety of key performance metrics (KPIs), such as:

*   **Total Return:** The overall profit or loss of the strategy.
*   **Sharpe Ratio:** A measure of risk-adjusted return.
*   **Max Drawdown:** The largest peak-to-trough decline in the portfolio's value.

The backtesting framework will also generate a series of plots and charts to help you visualize the strategy's performance, including:

*   An equity curve, showing the growth of the portfolio over time.
*   A drawdown plot, showing the periods where the portfolio was losing value.

By analyzing these metrics and visualizations, you can gain a deeper understanding of your strategy's strengths and weaknesses, and make informed decisions about whether to deploy it in a live market.

## 4. Advanced Strategy Concepts

For more complex strategies, you will need to manage your positions more actively and potentially adjust your strategy on a daily basis. This section will cover some of these advanced concepts using an **Iron Condor** strategy as an example.

An Iron Condor is a neutral strategy that involves selling a call spread and a put spread. It profits when the underlying asset stays within a certain price range.

### 4.1. Managing Positions

In a real-world scenario, you will want to close your positions based on certain criteria, such as:

*   **Profit Target:** Close the position when it has reached a certain profit.
*   **Stop Loss:** Close the position to limit your losses if the trade moves against you.
*   **Expiration:** Close the position as it nears expiration to avoid assignment.

Here is an example of how you might implement this logic in the `on_data` method:

```python
class IronCondor(Strategy):
    # ... (initialize method) ...

    def on_data(self, data):
        # Check if we have an open Iron Condor position
        if self.portfolio.has_position('IronCondor'):
            position = self.portfolio.get_position('IronCondor')

            # Check for profit target or stop loss
            if position.pnl_percent >= self.profit_target or position.pnl_percent <= self.stop_loss:
                self.close_position(position)
                return

            # Check if it's time to adjust the position
            if self.should_adjust(position, data):
                self.adjust_position(position, data)

        # If we don't have a position, look for an opportunity to open one
        else:
            if self.is_good_time_to_open_position(data):
                self.open_iron_condor(data)
```

### 4.2. Daily Rebalancing and Adjustments

Many strategies require daily monitoring and adjustments. For example, with an Iron Condor, you might want to adjust the position if the price of the underlying asset moves too close to one of your short strikes.

The `on_data` method is called for each new data point (which can be daily or even intraday, depending on your data). You can use this method to implement your daily rebalancing logic.

In the example above, the `should_adjust` and `adjust_position` methods would contain the logic for when and how to adjust the Iron Condor. This could involve closing the existing position and opening a new one with different strike prices.

### 4.3. A Note on Complexity

As your strategies become more complex, it is important to keep your code organized and well-documented. You may want to create helper methods for tasks such as:

*   Finding suitable options contracts.
*   Calculating position sizes.
*   Managing the state of your strategy.

By breaking down your strategy into smaller, more manageable pieces, you can make it easier to develop, test, and debug.
