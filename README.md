# Options Backtesting Framework

This project provides a backtesting framework for options trading strategies.

## Setup and Usage

These instructions will guide you through setting up a virtual environment using `uv` and running the scraper.

### 1. Install uv

If you don't have `uv` installed, you can install it with this command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Google Chrome and ChromeDriver

This scraper uses Selenium to control a Chrome browser. You will need to have Google Chrome installed.

You will also need to install the ChromeDriver. The easiest way to do this is with Homebrew:

```bash
brew install chromedriver
```

If you are not on macOS, or you don't use Homebrew, you can download the ChromeDriver from the [official website](https://chromedriver.chromium.org/downloads).

### 3. Create the Virtual Environment

Navigate to the `backtest` directory and create a virtual environment:

```bash
cd /Users/nash/dev/backtest
uv venv
```

This will create a virtual environment named `.venv` in the current directory.

### 4. Activate the Virtual Environment

Activate the virtual environment:

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

Install the required Python packages:

```bash
uv pip install -r requirements.txt
```

### 6. Run the Scraper

Now you can run the scraper to download options data. For example, to download options data for AAPL, run:

```bash
python yahoo_scraper.py AAPL
```

You can also specify a date range:

```bash
python yahoo_scraper.py AAPL --start-date 2025-09-01 --end-date 2025-12-31
```