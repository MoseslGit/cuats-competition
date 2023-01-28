from AlgorithmImports import *

import numpy as np
from datetime import datetime

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
        
        # Estimated risk-free rate for calculations
        self.risk_free_rate = 0.05
        self.thresholds = {
            'risk_factor': 2.0,
            'return_factor': 1.5,
            'diversification_factor': 0.5,
            'short_window': 7,
            'long_window': 14
        }
        

        #ETF universe selection
        spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        tqqq = self.AddEquity("TQQQ", Resolution.Daily).Symbol
        xagusd = self.AddCfd("XAGUSD", Resolution.Daily).Symbol
        ubt = self.AddEquity("UBT", Resolution.Daily).Symbol
        ust = self.AddEquity("UST", Resolution.Daily).Symbol
        self.ticker = ["SPY", "TQQQ", "XAGUSD", "UBT", "UST"]
        self.historytickers = [spy, tqqq, xagusd, ubt, ust]
        
        self.weightBySymbol = {"SPY" : 0.2, "TQQQ" : 0.2, "XAGUSD" : 0.2, "UBT" : 0.2, "UST" : 0.2}

        self.Schedule.On(
            self.DateRules.WeekStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 150),
            self.Rebalance)

        self.Schedule.On(
            self.DateRules.MonthStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY"),
            self.Update)

        self.SetWarmUp(100)


    def Rebalance(self):
        # Rebalance every week
        symbols = [symbol for symbol in self.Portfolio.Keys]
        historical_data = self.History(self.historytickers, 14, Resolution.Daily)

        rebalanced_portfolio = rebalance.adjust(self.weightBySymbol, historical_data, self.risk_free_rate, self.thresholds)
        for symbol in self.Portfolio.Keys:
            if symbol not in rebalanced_portfolio:
                self.Liquidate(symbol)
        for symbol in rebalanced_portfolio:
            if (rebalanced_portfolio[symbol] - self.weightBySymbol[symbol])/(rebalanced_portfolio[symbol] + self.weightBySymbol[symbol]) > 0.1:
                self.SetHoldings(symbol, rebalanced_portfolio[symbol])
        self.weightBySymbol = rebalanced_portfolio

    def Update(self):
        market_condition = 1
        symbols = [symbol for symbol in self.ticker]
        historical_data = self.History(self.historytickers, 30, Resolution.Daily)
        if market_condition == 1:
            updated_portfolio = strategies.crisis_strategy(historical_data, self.ticker)
        elif market_condition == 2:
            updated_portfolio = strategies.steady_state_strategy(historical_data, self.ticker)
        elif market_condition == 3:
            updated_portfolio = strategies.inflation_strategy(historical_data, self.ticker)
        else:
            updated_portfolio = strategies.woi_strategy(historical_data, self.ticker)
        for symbol in self.Portfolio.Keys:
            if symbol not in updated_portfolio:
                self.Liquidate(symbol)
        for symbol in updated_portfolio:
            self.SetHoldings(symbol, updated_portfolio[symbol])
        self.weightBySymbol = updated_portfolio
        

    def OnData(self, data):

        if self.IsWarmingUp:
            return

        if self.first_iteration:
            for symbol in self.weightBySymbol:
                self.SetHoldings(symbol, self.weightBySymbol[symbol])
            self.first_iteration = False

        return
