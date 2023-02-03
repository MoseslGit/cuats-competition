from AlgorithmImports import *
import numpy as np
import strategies
import rebalance
from sklearn import mixture
import pandas as pd
import statistics

class TradingStrategy(QCAlgorithm):
    def Initialize(self):

        self.SetBenchmark("SPY")
        
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2022, 12, 31)

        self.SetCash(100000)
        self.first_iteration = True
        self.market_condition = 1
        self.previous_value = 100000

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
        # TODO I think this code needs refactoring: Cindy (using self.Securities.Keys)
        self.historytickers = [spy, tqqq, xagusd, ubt, ust] # List of securities to be used for history function

        # ============= For model training =================
        self.vix = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol
        self.fred = self.AddData(Fred, Fred.CommercialPaper.Three0DayAAAssetbackedCommercialPaperInterestRate, Resolution.Daily).Symbol # US federal reserve 30-day AA asset-backed commercial paper interest rate
        self.ustres = self.AddData(USTreasuryYieldCurveRate, "USTYCR").Symbol # US treasury yield

        # Set momentum indicator
        self.manual_mom = Momentum(30)
        self.manual_mom.Updated += (lambda sender, updated: self.mom_window.Add(updated))
        self.mom_window = RollingWindow[IndicatorDataPoint](30)

        # Set initial equal weights
        self.weightBySymbol = {"SPY" : 0.2, "TQQQ" : 0.2, "XAGUSD" : 0.2, "UBT" : 0.2, "UST" : 0.2}

        # Schedule updating and rebalancing of portfolio weights
        self.Schedule.On(
            self.DateRules.MonthStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY"),
            self.Update)

        self.Schedule.On(
            self.DateRules.WeekStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY"),
            self.Rebalance)
        
        self.SetWarmUp(100)

        # Set leverage, and lower cash buffer to reduce unfilled orders
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        self.Settings.FreePortfolioValuePercentage = 0.05

        # Model setup
        self.model_training = False
        self.model = self.TrainModel(2012, 8)
        
    # Set leverage to 5
    def CustomSecurityInitializer(self, security):
        security.SetLeverage(5)

    def Rebalance(self):
        self.portfolio_returns = float(self.Portfolio.TotalPortfolioValue - self.previous_value)
        self.previous_value = float(self.Portfolio.TotalPortfolioValue)
        # Rebalance every week depending on portfolio performance
        historical_data = self.History(self.historytickers, 14, Resolution.Daily)

        # Call rebalancing function from rebalance.py
        rebalanced_portfolio = rebalance.adjust(self.weightBySymbol, self.market_condition, historical_data, self.risk_free_rate, self.thresholds, self.portfolio_returns)
        
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
        self.market_condition = self.PredictModel()
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
        if self.IsWarmingUp or self.model_training:
            return

        # On first iteration, set initial portfolio weights as a baseline
        if self.first_iteration:
            for symbol in self.weightBySymbol:
                self.SetHoldings(symbol, self.weightBySymbol[symbol])
            self.first_iteration = False

    def IndicatorUpdate(self):
        """Update indicators."""
        
    def TrainModel(self, startyear, numyears):
        """Method to train model. Should only be called once at initialisation unless update needed.
        :param int startyear: Start year of training data in YYYY format
        :param int years: Number of years of training data to use
        """
        self.Debug(str(('Start training at {}'.format(self.Time))))
        self.model_training = True
        
        # We want historic data grouped by month. Can do this by requesting data for one month at a time or slicing the whole history array using datetime indices. Here, we will request a month at a time.
        years = map(str, range(startyear, startyear+numyears+1))
        months = map(str, range(1, 13))
        
        # Save data for training
        spy_returns = []
        tqqq_returns = []
        spy_volatilities = []
        vix_volatilities = []
        spy_sharpes = []
        spy_momentums = []
        us_treasury = []
        interest_rates = []

        # Use S&P as market data
        for year in years:
            for month in months:
                self.mom_window.Reset()
                # Month start
                start_date_str = year + " " + month + " 01"
                start_date = Time.ParseDate(start_date_str)
                # Month end
                end_date_str = year + " " + month + " 30"
                end_date = Time.ParseDate(end_date_str)
                # Get history of all securities
                history = history = self.History(self.Securities.Keys, start_date, end_date, Resolution.Daily)

                #1. Monthly return of the market in percent
                market_return = history.loc["spy"].close.pct_change().dropna().mean()
                market_return = market_return * 100
                spy_returns.append(market_return)
                #2. Monthly tech stock return - available from 2011
                tqqq_return = history.loc["tqqq"].close.pct_change().dropna().mean()
                tqqq_returns.append(tqqq_return * 100)
                #3. Monthly volatility of the market in percent
                market_volatility = history.loc["spy"].close.pct_change().dropna().std()
                market_volatility = market_volatility * 100
                spy_volatilities.append(market_volatility)
                #4. Monthly VIX average for volatility
                vix_volatilities.append(history.loc["vix"].close.dropna().mean())
                #5. Sharpe ratio of the market
                sharpe_ratio = (market_return - self.risk_free_rate) / market_volatility
                spy_sharpes.append(sharpe_ratio)
                #6. Monthly average day to day momentum of market
                for time, price in history.loc["spy"]["close"].items():
                    self.manual_mom.Update(time, price)
                momentum_list = [item.Value for item in self.mom_window]
                if momentum_list:
                    spy_momentums.append(statistics.fmean(momentum_list))
                else: # No data, act as if neutral
                    spy_momentums.append(0)
                #7. US Treasury price
                # We only want one datapoint for onemonth yield curve, so reduce history request period to 5 days (certain will have at least one non NaN value)
                # Available from 2012
                tres_start_date = Time.ParseDate(year + " " + month + " 25")
                ustres_history = self.History(self.ustres, tres_start_date, end_date, Resolution.Daily)
                us_treasury.append(ustres_history["onemonth"].dropna().mean())
                #8. US federal reserve 30-day AA asset-backed commercial paper interest rate
                fred_history = self.History(self.fred, start_date, end_date, Resolution.Daily)
                interest_rates.append(fred_history["value"].dropna().mean())

        # 2D array: years*12 x n where n is number of predictive variables
        data = list(zip(spy_returns, tqqq_returns, spy_volatilities, vix_volatilities, spy_sharpes, spy_momentums, us_treasury, interest_rates))
        
        model = mixture.BayesianGaussianMixture(n_components=4, covariance_type='full', random_state=0).fit(data)
        np.savetxt("means.csv", model.means_, delimiter=",")
        self.Debug(str(model.means_))
        self.Debug(str(model.covariances_[0]))
        np.savetxt("covs.csv", model.covariances_[0], delimiter=",")
        self.model_training = False
        return model

    def PredictModel(self):
        """Predict on one datapoint averaged from data from one month"""
        # Initialise test dataset - 1xn where n is number of predictive variables
        test_data = np.empty((1,8))

        self.mom_window.Reset()

        history = self.History(self.Securities.Keys, 30, Resolution.Daily)
        #1. Monthly return of SPY
        market_return = history.loc["spy"].close.pct_change().dropna().mean()
        market_return = market_return * 100
        test_data[0,0] = market_return
        #2. Monthly return of tech stocks
        tqqq_return = history.loc["tqqq"].close.pct_change().dropna().mean()
        test_data[0,1] = tqqq_return * 100
        #3. Monthly volatility of SPY
        market_volatility = history.loc["spy"].close.pct_change().dropna().std()
        market_volatility = market_volatility * 100
        test_data[0,2] = market_volatility
        #4. Monthly VIX average for volatility
        vix = history.loc["vix"].close.mean()
        test_data[0,3] = vix
        #5. Sharpe ratio of SPY
        sharpe_ratio = (market_return - self.risk_free_rate) / market_volatility
        test_data[0,4] = sharpe_ratio
        #6. Momentum of SPY
        for time, price in history.loc["spy"]["close"].items():
            self.manual_mom.Update(time, price)
        momentum_list = [item.Value for item in self.mom_window]
        if momentum_list:
            test_data[0,5] = statistics.fmean(momentum_list)
        else: # No data, act as if neutral
            test_data[0,5] = 0
        #7. US treasury yield
        ustres_history = self.History(self.ustres, 30, Resolution.Daily)
        test_data[0,6] = ustres_history["onemonth"].dropna().mean()
        #8. Interest rates
        fred_history = self.History(self.fred, 30, Resolution.Daily)
        test_data[0,7] = fred_history["value"].dropna().mean()

        # Debug predictive values
        self.Debug(str(self.model.predict(test_data)))

        return self.model.predict(test_data)[0]