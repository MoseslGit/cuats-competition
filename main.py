from AlgorithmImports import *
from InsightWeightingPortfolioConstructionModel import InsightWeightingPortfolioConstructionModel

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

        self.SetBenchmark("SPY")
        
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 12, 31)

        self.SetCash(100000)
        
        # Estimated risk-free rate for calculations
        self.risk_free_rate = 0.05
        self.thresholds = {
            'risk_factor': 2.0,
            'return_factor': 1.5,
            'diversification_factor': 0.5,
            'short_window': 20,
            'long_window': 50
        }

        #Import trained market identification model
        self.model = train_model.load_model()

        # Add universe of 10 largest ETFs, 10 largest stocks, and 10 largest bonds
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

        # Set portfolio construction model
        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel())

        # Set execution model
        self.SetExecution(ImmediateExecutionModel())

        # full list of equities we consider here, including major indexes and ETFs
        full_equity_list = [
            'SPY',
            'QQQ',
            'IWM',
            'EEM',
            'TLT',
            'GLD',
            'VXX',
            'XLF',
            'XLE',
            'XLI',
            'XLP',
            'XLV',
        ]

        # full list of bonds used here
        full_bond_list = [
            'SHY',
            'IEF',
            'TLT',
            'LQD',
            'HYG',
            'EMB',
            'TIP',
        ]

        # Add data for equities considered
        for symbol in full_equity_list:
            self.AddEquity(symbol, Resolution.Daily)
        
        # Add data for bonds considered
        for symbol in full_bond_list:
            self.AddEquity(symbol, Resolution.Daily)

        # initialize portfolio
        self.SetPortfolioConstruction(ConfidenceWeightedOptimizationPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

        self.frequency = 5  # rebalance portfolio every 5 days
        self.SetWarmUp(100)
        self.Schedule.On(self.DateRules.EveryDay(self.Symbols), self.TimeRules.AfterMarketOpen(self.Symbols), self.rebalance_portfolio)

    def CoarseSelectionFunction(self, coarse):
        # Sort stocks by daily dollar volume and take the top 20
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        top20 = sortedByDollarVolume[:20]

        # Filter out stocks with fundamental data
        filtered = [x for x in top20 if x.HasFundamentalData]

        # Sort stocks by daily dollar volume and take the top 20
        sortedByDollarVolume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        top20 = sortedByDollarVolume[:20]

        return [x.Symbol for x in top20]
    
    def FineSelectionFunction(self, fine):
        # Filter out stocks with fundamental data
        filtered = [x for x in fine if x.HasFundamentalData]

        # Sort stocks by daily dollar volume and take the top 20
        sortedByDollarVolume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        top20 = sortedByDollarVolume[:20]

        return [x.Symbol for x in top20]

    def rebalance_portfolio(self):
        # Get insights
        insights = self.GetInsights()
        # Set portfolio targets
        self.SetHoldings(insights)

    def OnData(self, data):

        if self.IsWarmingUp:
            return

        # Get market condition
        market_condition = strategies.identify_market(self.model, data)

        # Update holdings based on market condition according to respective strategy in strategies.py
        

        '''else:
            # Update portfolio
            self.portfolio = strategies.update(self.portfolio, data, market_condition)

            # Check if it is time to rebalance portfolio
            if self.Time.day % self.frequency == 0:
                # Rebalance portfolio
                self.portfolio = feedback.adjust(self.portfolio, data, self.risk_free_rate, self.thresholds)
                
                # Execute trades to adjust portfolio
                for i, symbol in enumerate(self.portfolio):
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
                    self.Portfolio.SetHoldings(symbol, asset_weight, self.Portfolio.TotalPortfolioValue)''''''



                        
