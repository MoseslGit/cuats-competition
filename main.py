from AlgorithmImports import *

import numpy as np
from datetime import datetime

import train_model
import strategies
import rebalance


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
        #Manual universe selection
        #tickers = ['TSLA', 'AAPL']
        #symbols = []
        # loop through the tickers list and create symbols for the universe
        #for i in range(len(tickers)):
        #    symbols.append(Symbol.Create(tickers[i], SecurityType.Equity, Market.USA))
        #    allocationPlot.AddSeries(Series(tickers[i], SeriesType.Line, ''))
        #self.AddChart(allocationPlot)
        #self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))

        #ETF universe selection
        #SPY ETF
        self.spy = self.AddEquity("SPY").Symbol
        self.AddUniverse(self.Universe.ETF(self.spy, self.UniverseSettings, self.ETFConstituentsFilter))
        #Gold ETF
        self.gld = self.AddEquity("GLD").Symbol
        self.AddUniverse(self.Universe.ETF(self.gld, self.UniverseSettings, self.ETFConstituentsFilter))
        #QQQ Tech ETF
        self.qqq = self.AddEquity("QQQ").Symbol
        self.AddUniverse(self.Universe.ETF(self.qqq, self.UniverseSettings, self.ETFConstituentsFilter))
        #Bond ETF
        self.agg = self.AddEquity("AGG").Symbol
        self.AddUniverse(self.Universe.ETF(self.agg, self.UniverseSettings, self.ETFConstituentsFilter))

        self.securities = []
        self.weightBySymbol = {}
        
        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.AfterMarketOpen(self.spy, 1),
            self.Rebalance)

        self.SetWarmUp(100)

    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
        for security in changes.AddedSecurities:
            security.SetLeverage(10)
            history = self.History(security.Symbol, 7, Resolution.Daily)
            self.Log(f'{security.Symbol.Value} added to the universe')
            

        for security in changes.RemovedSecurities:
            self.Log(f'{security.Symbol.Value} removed from to the universe')
            if security in self.securities:
                self.securities.remove(security)

        self.securities.extend(changes.AddedSecurities)


    def ETFConstituentsFilter(self, constituents):
        # Get the 10 securities with the largest weight in the index
        selected = sorted([c for c in constituents if c.Weight],
            key=lambda c: c.Weight, reverse=True)[:10]
        self.weightBySymbol = {c.Symbol: c.Weight for c in selected}
        
        return list(self.weightBySymbol.keys())


    def OnData(self, data):

        if self.IsWarmingUp:
            return

        # Every month check market condition and update portfolio
        if self.Time.day % 30 == 0:

            market_condition = strategies.identify_market(self.model, data)
            # Update portfolio
            updated_portfolio = strategies.update(self.Portfolio, data, market_condition)
            for symbol in self.Portfolio.Keys:
                if symbol not in rebalanced_portfolio:
                    self.Liquidate(symbol)
                self.SetHoldings(symbol, updated_portfolio[symbol])

        #Else every 2 weeks rebalance portfolio
        elif self.Time.day % 14 == 0:
            for i in range(len(self.Portfolio.Keys)):
                symbol = self.Portfolio.Keys[i]
                Close = data[symbol].Close
                currentweight = (self.Portfolio[symbol].Quantity * Close) /self.Portfolio.TotalPortfolioValue
            current_portfolio = {symbol: currentweight for symbol in self.Portfolio.Keys}
            rebalanced_portfolio = rebalance.adjust(current_portfolio, data, self.model, self.risk_free_rate, self.thresholds)
            for symbol in self.Portfolio.Keys:
                if symbol not in rebalanced_portfolio:
                    self.Liquidate(symbol)
                self.SetHoldings(symbol, rebalanced_portfolio[symbol])
