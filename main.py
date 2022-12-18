from quantconnect.algorithm import QCAlgorithm
from quantconnect.data import Market
from quantconnect.python import (
    HistoryRequest,
    PortfolioTarget,
    OrderDirection,
    OrderType
)
import numpy as np
from datetime import datetime

import train_model
import strategies
import feedback

class TradingStrategy(QCAlgorithm):
    def Initialize(self):
        # Set frequency of portfolio adjustments (e.g. daily, weekly, monthly)
        frequency = 'daily'
        
        # Set start and end dates for backtesting
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 12, 31)
        
        # Set risk-free rate and threshold parameters
        self.risk_free_rate = 0.05
        self.thresholds = {
            'risk_factor': 2.0,
            'return_factor': 1.5,
            'diversification_factor': 0.5,
            'short_window': 20,
            'long_window': 50
        }

        # Train machine learning model to identify market condition
        self.model = train_model.train()
        # Set up event handler for new data
        self.SetWarmUp(self.Time.DaysInPeriod(frequency, start_date))
        self.Schedule.On(self.DateRules.EveryDay(self.Symbols), self.TimeRules.AfterMarketOpen(self.Symbols), self.rebalance_portfolio)
        # Set initial portfolio
        self.portfolio = np.ones((self.Securities.Count,)) / self.Securities.Count
        
        # Set frequency for portfolio rebalancing
        self.frequency = 1
        
        # Set flag to indicate first iteration
        self.first_iteration = True

        # Subscribe to data for all securities
        for symbol in self.SymbolCache:
            self.AddData(symbol.ID.Symbol, symbol.ID.SecurityType, symbol.ID.Resolution)
        
    def OnData(self, data):
        if self.first_iteration:
            # Get historical data for all securities
            history = self.History([HistoryRequest(symbol.ID, self.Time - self.frequency, self.Time, Resolution.Daily) for symbol in self.SymbolCache])
            self.first_iteration = False
        else:
            # Update portfolio
            market_condition = train_model.identify_market_condition(data)
            self.portfolio = strategies.update(self.portfolio, data, market_condition)

            # Check if it is time to rebalance portfolio
            if self.Time.day % self.frequency == 0:
                # Rebalance portfolio
                self.portfolio = feedback.adjust(self.portfolio, data, self.risk_free_rate, self.thresholds)
                
                # Execute trades to adjust portfolio
                for i, symbol in enumerate(self.SymbolCache):
                    asset_weight = self.portfolio[i]  # weight of asset in portfolio
                    asset_holdings = self.Portfolio[symbol].Quantity  # current holdings of asset

                    # Calculate number of shares to buy or sell
                    shares = int(asset_weight * self.Portfolio.TotalPortfolioValue / data[symbol].Price) - asset_holdings

                    # Check if we need to buy or sell
                    if shares > 0:
                        # Buy shares
                        self.Buy(symbol, shares)
                    elif shares < 0:
                        # Sell shares
                        self.Sell(symbol, -shares)
                    if shares == 0:
                        continue  # skip to next asset

                    if not self.Securities[symbol].IsTradable:
                        self.Log(f'{symbol} is not tradable')
                        continue  # skip to next asset
                    try:
                        # Buy shares
                        self.Buy(symbol, shares)
                    except Exception as e:
                        self.Log(f'Error executing trade for {symbol}: {e}')
                    # Log trade details
                    self.Log(f'Traded {shares} shares of {symbol} at {data[symbol].Price:.2f}')
                    self.Log(f'Updated portfolio weights: {self.portfolio}')
                    # Update portfolio value and cash balance
                    self.Portfolio.SetCash(self.Portfolio.Cash + shares * data[symbol].Price)
                    self.Portfolio.SetHoldings(symbol, asset_weight, self.Portfolio.TotalPortfolioValue)



                        
