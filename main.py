from AlgorithmImports import *

import numpy as np
from datetime import datetime

#import train_model
import strategies
import rebalance


class TradingStrategy(QCAlgorithm):
    def Initialize(self):

        self.SetBenchmark("SPY")
        
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 12, 31)

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
        
        #Import trained market identification model
        #self.model = train_model.load_model()
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
        self.AddEquity("QQQ", Resolution.Daily)
        self.AddEquity("GLD", Resolution.Daily)
        self.AddEquity("UBT", Resolution.Daily)
        self.AddEquity("UST", Resolution.Daily)
        self.securities = ["QQQ", "GLD", "UBT", "UST"]
        
        self.weightBySymbol = {}

        self.Schedule.On(
            self.DateRules.WeekStart(self.spy),
            self.TimeRules.AfterMarketOpen(self.spy, 1),
            self.Rebalance)

        self.SetWarmUp(100)


    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
        for security in changes.AddedSecurities:
            security.SetLeverage(10)
            history = self.History(security.Symbol, 7, Resolution.Daily)
            self.securities.append(security)
            self.Log(f'{security.Symbol.Value} added to the universe')
            

        for security in changes.RemovedSecurities:
            self.Log(f'{security.Symbol.Value} removed from to the universe')
            if security in self.securities:
                self.securities.remove(security)
            self.Liquidate(security.Symbol)

        self.securities.extend(changes.AddedSecurities)


    def ETFConstituentsFilter(self, constituents):
        # Get the 10 securities with the largest weight in the index
        selected = sorted([c for c in constituents if c.Weight],
            key=lambda c: c.Weight, reverse=True)[:10]
        self.weightBySymbol = {c.Symbol: c.Weight for c in selected}
        self.securities = [c.Symbol for c in selected]
        return list(self.weightBySymbol.keys())

    def Rebalance(self):
        # Rebalance every week
        symbols = [symbol for symbol in self.Portfolio.Keys]
        historical_data = self.History(symbols, 30, Resolution.Daily)
        if not all(symbol in historical_data.index.levels[0] and historical_data.loc[symbol].shape[0] >= self.lookback for symbol in [self.spy, self.gld, self.agg, self.igov]): return
        rebalanced_portfolio = rebalance.adjust(self.weightBySymbol, historical_data, self.risk_free_rate, self.thresholds)
        for symbol in self.Portfolio.Keys:
            if symbol not in rebalanced_portfolio:
                self.Liquidate(symbol)
        for symbol in rebalanced_portfolio:
            if (rebalanced_portfolio[symbol] - self.weightBySymbol[symbol])/(rebalanced_portfolio[symbol] + self.weightBySymbol[symbol]) > 0.1:
                self.SetHoldings(symbol, rebalanced_portfolio[symbol])
        self.weightBySymbol[symbol] = rebalanced_portfolio[symbol]

    def OnData(self, data):

        if self.IsWarmingUp:
            return

        if self.first_iteration:
            for symbol in self.weightBySymbol:
                self.SetHoldings(symbol, self.weightBySymbol[symbol])
            self.first_iteration = False
            #add gld, igov, ivw, agg to self.weightBySymbol

            return
        #Every 30 days, check market condition and rebalance portfolio
        if self.Time.day % 30 == 0:

            #market_condition = strategies.identify_market(self.model, data)
            #random int between 1 and 4
            market_condition = np.random.randint(1, 5)
            symbols = [symbol for symbol in self.securities]
            historical_data = self.History(symbols, 60, Resolution.Daily)
            # Update portfolio
            if market_condition == 1:
                updated_portfolio = strategies.crisis_strategy(self.weightBySymbol, historical_data, self.securities)
            elif market_condition == 2:
                updated_portfolio = strategies.steady_state_strategy(self.weightBySymbol, historical_data, self.securities)
            elif market_condition == 3:
                updated_portfolio = strategies.inflation_strategy(self.weightBySymbol, historical_data, self.securities)
            elif market_condition == 4:
                updated_portfolio = strategies.woi_strategy(self.weightBySymbol, historical_data, self.securities)
            else:
                self.Log("Market condition not recognized")
            self.weightBySymbol = updated_portfolio
            for symbol in self.Portfolio.Keys:
                if symbol not in updated_portfolio:
                    self.Liquidate(symbol)
            for symbol in updated_portfolio:
                self.SetHoldings(symbol, updated_portfolio[symbol])
        '''#Else every 2 weeks rebalance portfolio
        elif self.Time.day % 30 == 0:
                symbols = [symbol for symbol in self.Portfolio.Keys]
                historical_data = self.History(symbols, 30, Resolution.Daily)
                rebalanced_portfolio = rebalance.adjust(self.weightBySymbol, historical_data, self.risk_free_rate, self.thresholds)
                for symbol in self.Portfolio.Keys:
                    if symbol not in rebalanced_portfolio:
                        self.Liquidate(symbol)
                for symbol in rebalanced_portfolio:
                    if (rebalanced_portfolio[symbol] - self.weightBySymbol[symbol])/(rebalanced_portfolio[symbol] + self.weightBySymbol[symbol]) > 0.1:
                        self.SetHoldings(symbol, rebalanced_portfolio[symbol])'''