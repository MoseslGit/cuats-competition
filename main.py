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

        # Add universe of 10 largest ETFs, 10 largest stocks, and 10 largest bonds
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

        # Set portfolio construction model
        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel())

        self.securities = []
        self.symbol_data_by_symbol = {}

        self.frequency = 5  # rebalance portfolio every 5 days
        self.SetWarmUp(100)
        self.Schedule.On(self.DateRules.EveryDay(self.Symbols), self.TimeRules.AfterMarketOpen(self.Symbols), self.rebalance_portfolio)


    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:

        for security in changes.AddedSecurities:
            self.symbol_data_by_symbol[security.Symbol] = SymbolData() # You need to define this class

        for security in changes.RemovedSecurities:
            if security in self.securities:
                self.securities.remove(security)
                self.symbol_data_by_symbol.pop(security.Symbol, None)
            
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



                        
