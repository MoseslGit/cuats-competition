from AlgorithmImports import *

import numpy as np
from datetime import datetime

#import ML model, market condition strategies, and rebalancing functions
#import train_model
import strategies
import rebalance


class TradingStrategy(QCAlgorithm):
    def Initialize(self):

        self.SetBenchmark("SPY")
        
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2020, 12, 31)

        self.SetCash(100000)
        self.first_iteration = True
        self.market_condition = 1

        # Set estimated risk-free rate, as well as rebalancing thresholds for calculations
        self.risk_free_rate = 0.05
        self.thresholds = {
            'risk_factor': 2.0,
            'return_factor': 1.5,
            'diversification_factor': 0.5,
            'short_window': 7,
            'long_window': 14
        }
        

        # 5 etfs selected as proof of concept, SPY as the market, TQQQ as tech, XAGUSD as gold, UBT as bonds, UST as treasuries
        spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        tqqq = self.AddEquity("TQQQ", Resolution.Daily).Symbol
        xagusd = self.AddCfd("XAGUSD", Resolution.Daily).Symbol
        ubt = self.AddEquity("UBT", Resolution.Daily).Symbol
        ust = self.AddEquity("UST", Resolution.Daily).Symbol
        self.ticker = ["SPY", "TQQQ", "XAGUSD", "UBT", "UST"]
        self.historytickers = [spy, tqqq, xagusd, ubt, ust]
        
        # Set initial equal weights
        self.weightBySymbol = {"SPY" : 0.2, "TQQQ" : 0.2, "XAGUSD" : 0.2, "UBT" : 0.2, "UST" : 0.2}

        # schedule updating and rebalancing of portfolio weights
        self.Schedule.On(
            self.DateRules.MonthStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY"),
            self.Update)

        self.Schedule.On(
            self.DateRules.WeekStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 150),
            self.Rebalance)

        self.SetWarmUp(100)


    def Rebalance(self):
        # Rebalance every week depending on portfolio performance
        historical_data = self.History(self.historytickers, 14, Resolution.Daily)

        # Call rebalancing function from rebalance.py
        rebalanced_portfolio = rebalance.adjust(self.weightBySymbol, self.market_condition, historical_data, self.risk_free_rate, self.thresholds)
        
        # Liquidate any symbols that are no longer in the portfolio
        for symbol in self.Portfolio.Keys:
            if symbol not in rebalanced_portfolio:
                self.Liquidate(symbol)
        
        # Set holdings for rebalanced portfolio
        for symbol in rebalanced_portfolio:
            self.SetHoldings(symbol, rebalanced_portfolio[symbol])
        self.weightBySymbol = rebalanced_portfolio

    def Update(self):
        # Update portfolio weights every month depending on market conditions
        self.market_condition = 4
        historical_data = self.History(self.historytickers, 30, Resolution.Daily)

        # Call strategy from strategies.py based on market condition
        if self.market_condition == 1:
            updated_portfolio = strategies.crisis_strategy(historical_data, self.ticker)
        elif self.market_condition == 2:
            updated_portfolio = strategies.steady_state_strategy(historical_data, self.ticker)
        elif self.market_condition == 3:
            updated_portfolio = strategies.inflation_strategy(historical_data, self.ticker)
        else:
            updated_portfolio = strategies.woi_strategy(historical_data, self.ticker)

        # Liquidate any symbols that are no longer in the portfolio
        for symbol in self.Portfolio.Keys:
            if symbol not in updated_portfolio:
                self.Liquidate(symbol)

        # Set holdings for updated portfolio
        for symbol in updated_portfolio:
            self.SetHoldings(symbol, updated_portfolio[symbol])
        self.weightBySymbol = updated_portfolio
        

    def OnData(self, data):

        if self.IsWarmingUp:
            return

        # On first iteration, set initial portfolio weights as a baseline
        if self.first_iteration:
            for symbol in self.weightBySymbol:
                self.SetHoldings(symbol, self.weightBySymbol[symbol])
            self.first_iteration = False

        return
