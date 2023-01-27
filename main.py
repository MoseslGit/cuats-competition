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
            'short_window': 7,
            'long_window': 14
        }

        # #Import trained market identification model
        # self.model = train_model.load_model()
        
        # Train model once on initialisation
        self.Train(self.train_model)

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
        self.Schedule.On(self.DateRules.EveryDay(self.Symbols), self.TimeRules.AfterMarketOpen(self.Symbols), self.rebalance_portfolio)
        
        # Train model at the end of every month at midnight so that it's ready exactly at month start
        self.Train(self.DataRules.MonthEnd(0), self.TimeRules.Midnight, self.predict_model)
        

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

        if self.IsWarmingUp or self.model_is_training():
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
            for symbol in self.Portfolio.Keys:
                Close = data[symbol].Close
                currentweight = (self.Portfolio[symbol].Quantity * Close) /self.Portfolio.TotalPortfolioValue
            current_portfolio = {symbol: currentweight for symbol in self.Portfolio.Keys}
            symbols = [symbol for symbol in self.Portfolio.Keys]
            historical_data = self.History(symbols, 30, Resolution.Daily)
            rebalanced_portfolio = rebalance.adjust(current_portfolio, historical_data, self.risk_free_rate, self.thresholds)
            for symbol in self.Portfolio.Keys:
                if symbol not in rebalanced_portfolio:
                    self.Liquidate(symbol)
                self.SetHoldings(symbol, rebalanced_portfolio[symbol])

    def train_model(self, startyear, years):
        """Method to train model. Should only be called once at initialisation unless update needed.
        :param int startyear: Start year of training data in YYYY format
        :param int years: Number of years of training data to use
        """
        self.Log('Start training at {}'.format(self.Time))
        self.model_is_training = True
        
        # We want historic data grouped by month. Can do this by requesting data for one month at a time or slicing the whole history array using datetime indices. Here, we will request a month at a time.
        years = map(str, range(startyear, startyear+years+1))
        months = map(str, range(1, 13))
        
        # # Calculate input data for model
        # # For market as whole
        # #1. Monthly return of the market
        # market_history = self.History(self.spy, 30, Resolution.Daily)
        # market_return = market_history.Close.pct_change().dropna().mean()
        # market_return = market_return * 100
        # #2. Monthly volatility of the market
        # market_volatility = market_history.Close.pct_change().dropna().std()
        # market_volatility = market_volatility * 100
        
        # # For each security
        # for security in self.securities:
        #     history = self.History(security.Symbol, 30, Resolution.Daily)
        #     # Monthly return of each security
        #     monthly_return = history.Close.pct_change().dropna().mean()
        #     monthly_return = monthly_return * 100
        #     #2. Monthly volatility of each security
        #     monthly_volatility = history.Close.pct_change().dropna().std()
        #     monthly_volatility = monthly_volatility * 100
        #     #3. Covariance of monthly returns of each security and the market
        #     covariance = history.Close.pct_change().dropna().cov(market_history.Close.pct_change().dropna())
        #     #4. Beta of each security
        #     beta = covariance / market_volatility
        #     #5. Alpha of each security
        #     alpha = monthly_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))
        #     #6. Sharpe ratio of each security
        #     sharpe_ratio = (monthly_return - self.risk_free_rate) / monthly_volatility
        #     #9. Treynor ratio of each security
        #     treynor_ratio = (monthly_return - self.risk_free_rate) / beta
        #     #10. Information ratio of each security
        #     information_ratio = monthly_return / monthly_volatility
        #     #11. Sortino ratio of each security
        #     sortino_ratio = (monthly_return - self.risk_free_rate) / monthly_volatility
        #     #12. Jensen's alpha of each security
        #     jensens_alpha = (monthly_return - self.risk_free_rate) - (beta * (market_return - self.risk_free_rate))
        #     #13. Market capitalisation of each security
        #     market_cap = security.MarketCap
        #     #14. Price to earnings ratio of each security
        #     p_e_ratio = security.PriceToEarningsRatio
        #     #15. Price to book ratio of each security
        #     p_b_ratio = security.PriceToBookRatio
        
        returns = []
        volatilities = []
        # Use S&P as market data
        for year in years:
            for month in months:
                start_date_str = year + " " + month + " 01"
                start_date = Time.ParseDate(date_str)
                end_date_str = year + " " + month + " 30"
                end_date = Time.ParseDate(date_str)
                market_history = self.History(self.spy, start_date, end_date, Resolution.Daily)
                #1. Monthly return of the market
                market_return = market_history.Close.pct_change().dropna().mean()
                market_return = market_return * 100
                returns.append(market_return)
                #2. Monthly volatility of the market
                market_volatility = market_history.Close.pct_change().dropna().std()
                market_volatility = market_volatility * 100
                volatilities.append(market_volatility)
        
        data = tuple(zip(returns, volatilities))
        
        model = train_model.train(data)
        self.model_is_training = False
    
    def predict_model(self):
        """Predict on one datapoint averaged from data from one month"""
        predict_history = self.History(timedelta(days=30), Resolution.Daily)
