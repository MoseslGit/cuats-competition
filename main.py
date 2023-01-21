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
import MorningstarSectorCode


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

        # Add universe of 10 largest ETFs, 10 largest stocks, and 10 largest bonds
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

        self.securities = []
        self.symbol_data_by_symbol = {}

        self.portfolio = {}

        self.frequency = 5  # rebalance portfolio every 5 days
        self.first_iteration = True
        self.SetWarmUp(100)
        self.Schedule.On(self.DateRules.EveryDay(self.Symbols), self.TimeRules.AfterMarketOpen(self.Symbols), self.rebalance_portfolio)


    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:

        for security in changes.AddedSecurities:
            self.symbol_data_by_symbol[security.Symbol] = SymbolData() # You need to define this class

        for security in changes.RemovedSecurities:
            if security in self.securities:
                self.securities.remove(security)
            if security.Symbol in self.symbol_data_by_symbol:
                symbol_data = self.symbol_data_by_symbol.pop(security.Symbol, None)
                #get rid of unnecessary data
                if symbol_data:
                    symbol_data.dispose()
        self.securities.extend(changes.AddedSecurities)

    # get the top 10 ETFs in each category
    def CoarseSelectionFunction(self, coarse):
        # Filter out ETFs
        etfs = [x for x in coarse if x.HasFundamentalData and x.Price > 10]

        # Sort ETFs by daily dollar volume and take the top 10
        sortedByDollarVolume = sorted(etfs, key=lambda x: x.DollarVolume, reverse=True)
        top10 = sortedByDollarVolume[:10]

        return [x.Symbol for x in top10]

    # keep only the ETFs in tech, commodities, bonds, and real estate
    def FineSelectionFunction(self, fine):
        # Filter ETFs by category
        tech_etfs = [x for x in fine if x.AssetClassification.MorningstarSectorCode == MorningstarSectorCode.InformationTechnology]
        commodity_etfs = [x for x in fine if x.AssetClassification.MorningstarSectorCode == MorningstarSectorCode.Materials]
        bond_etfs = [x for x in fine if x.AssetClassification.MorningstarSectorCode == MorningstarSectorCode.Bonds]
        real_estate_etfs = [x for x in fine if x.AssetClassification.MorningstarSectorCode == MorningstarSectorCode.RealEstate]

        # Sort ETFs by market cap and take the top 10
        sortedByMarketCap = sorted(tech_etfs, key=lambda x: x.EarningReports.BasicAverageShares.ThreeMonths, reverse=True)
        top10_tech = sortedByMarketCap[:10]

        sortedByMarketCap = sorted(commodity_etfs, key=lambda x: x.EarningReports.BasicAverageShares.ThreeMonths, reverse=True)
        top10_commodity = sortedByMarketCap[:10]

        sortedByMarketCap = sorted(bond_etfs, key=lambda x: x.EarningReports.BasicAverageShares.ThreeMonths, reverse=True)
        top10_bond = sortedByMarketCap[:10]

        sortedByMarketCap = sorted(real_estate_etfs, key=lambda x: x.EarningReports.BasicAverageShares.ThreeMonths, reverse=True)
        top10_real_estate = sortedByMarketCap[:10]
    

        # Return the top 10 ETFs in each category
        return [x.Symbol for x in top10_tech] + [x.Symbol for x in top10_commodity] + [x.Symbol for x in top10_bond] + [x.Symbol for x in top10_real_estate]

    def GetInsights(self):
        insights = []
        for security in self.securities:
            symbol_data = self.symbol_data_by_symbol[security.Symbol]
            # Get market condition
            market_condition = strategies.identify_market(self.model, symbol_data.indicator.Current.Value)
            # Get insight
            insight = strategies.get_insight(self.model, symbol_data.indicator.Current.Value, market_condition)
            # Add insight to list
            insights.append(insight)
        return insights


    def rebalance_portfolio(self):
        # Get insights
        insights = self.GetInsights()
        # Set portfolio targets
        self.SetHoldings(insights)

    def OnData(self, data):

        if self.IsWarmingUp:
            return

        # If this is the first iteration, create initial portfolio based off historical data
        if self.first_iteration:
            self.portfolio = strategies.construct_portfolio(self.securities, data, self.risk_free_rate, self.thresholds)
            self.first_iteration = False

        else: 
            # Update indicators
            for security in self.securities:
                symbol_data = self.symbol_data_by_symbol[security.Symbol]
                symbol_data.indicator.Update(self.Time, data[security.Symbol].Close)

        # Every month check market condition and rebalance portfolio
        if self.Time.day % 30 == 0:

            market_condition = strategies.identify_market(self.model, symbol_data.indicator.Current.Value)
            # Update portfolio
            self.portfolio = strategies.update(self.portfolio, data, market_condition)

        #Else every week use feedback to rebalance portfolio
        elif self.Time.day % 7 == 0:
            rebalanced_portfolio = feedback.adjust(self.portfolio, data, self.model, self.risk_free_rate, self.thresholds)
            for symbol in self.portfolio.items():
                if symbol not in rebalanced_portfolio:
                    self.Liquidate(symbol, 'Not selected')
            self.SetHoldings(symbol, self.portfolio[symbol])