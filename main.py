from AlgorithmImports import *
import numpy as np
import strategies
import rebalance
from sklearn import mixture, preprocessing
import pandas as pd
import statistics

class TradingStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetBenchmark("SPY")
        self.SetStartDate(2016, 2, 1)
        self.SetEndDate(2023, 2, 1)
        self.SetCash(100000)
        self.first_iteration = True
        self.market_condition = 1
        self.previous_value = 100000
        self.SetWarmUp(100)

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
        self.historytickers = [spy, tqqq, xagusd, ubt, ust] # List of securities to be used for history function

        # ============= For model training =================
        self.vix = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol
        # US federal reserve 30-day AA asset-backed commercial paper interest rate
        self.interest30 = self.AddData(Fred, Fred.CommercialPaper.Three0DayAAAssetbackedCommercialPaperInterestRate, Resolution.Daily).Symbol
        # Yield curve data
        self.ustres = self.AddData(USTreasuryYieldCurveRate, "USTYCR").Symbol
        # Set momentum indicator
        self.manual_mom = Momentum(30)
        self.manual_mom.Updated += (lambda sender, updated: self.mom_window.Add(updated))
        self.mom_window = RollingWindow[IndicatorDataPoint](30)
        
        # Set initial equal weights - initialises portfolio
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

        # Set Leverage, and lower cash buffer to reduce unfilled orders
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        self.Settings.FreePortfolioValuePercentage = 0.05

        # Model setup
        self.model_training = False
        self.model = self.TrainModel(2001, 21)

    def CustomSecurityInitializer(self, security):
        """Set leverage to 5."""
        security.SetLeverage(5)

    def Rebalance(self):
        """Called by Schedule function in QuantConnect automatically every week."""
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
        """Called by Schedule function in QuantConnect automatically every month.
        Update portfolio weights every month depending on market conditions."""
        self.market_condition = self.PredictModel()
        historical_data = self.History(self.historytickers, 30, Resolution.Daily)

        # Call strategy from strategies.py based on market condition
        if self.market_condition == 1:
            updated_portfolio = strategies.crisis_strategy(historical_data, self.ticker)
        elif self.market_condition == 0:
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
        """Called every time data updates automatically.
        :param dataframe data: Historical data on assets in our portfolio
        """
        if self.IsWarmingUp or self.model_training:
            return

        # On first iteration, set initial portfolio weights as a baseline
        if self.first_iteration:
            for symbol in self.weightBySymbol:
                self.SetHoldings(symbol, self.weightBySymbol[symbol])
            self.first_iteration = False
        self.Benchmark.Evaluate(self.Time)
        # Stop loss
        if self.Portfolio.TotalPortfolioValue/self.previous_value < 0.95:
            self.Liquidate()
        
    def TrainModel(self, startyear, numyears):
        """Method to train model. Should only be called once at initialisation unless update needed.
        Note the model cluster assingments may change if argument random_state is not fixed.
        :param int startyear: Start year of training data in YYYY format
        :param int years: Number of years of training data to use
        :return sklearn GMM model: trained model
        """
        self.Debug(str(('Start training at {}'.format(self.Time))))
        self.model_training = True
        
        # We want historic data grouped by month. Can do this by requesting data for one month at a time or slicing the whole history array using datetime indices. Here, we will request a month at a time.
        years = map(str, range(startyear, startyear+numyears+1))
        months = map(str, range(1, 13))
        
        # Save data for training
        spy_returns = []
        spy_volatilities = []
        vix_volatilities = []
        spy_sharpes = []
        spy_momentums = []
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

                #1. Monthly return of SPY (percent)
                market_return = history.loc["spy"].close.pct_change().dropna().mean()
                market_return = market_return * 100
                spy_returns.append(market_return)
                #2. Monthly volatility of SPY (percent)
                market_volatility = history.loc["spy"].close.pct_change().dropna().std()
                market_volatility = market_volatility * 100
                spy_volatilities.append(market_volatility)
                #3. Monthly VIX average for volatility
                vix_volatilities.append(history.loc["vix"].close.dropna().mean())
                #4. Sharpe ratio of SPY
                sharpe_ratio = (market_return - self.risk_free_rate) / market_volatility
                spy_sharpes.append(sharpe_ratio)
                #5. Monthly average day to day momentum of market
                for time, price in history.loc["spy"]["close"].items():
                    self.manual_mom.Update(time, price)
                momentum_list = [item.Value for item in self.mom_window]
                if momentum_list:
                    spy_momentums.append(statistics.fmean(momentum_list))
                else: # No data, act as if neutral
                    spy_momentums.append(0)
                #6. US federal reserve 30-day AA asset-backed commercial paper interest rate (percent)
                interest30_history = self.History(self.interest30, start_date, end_date, Resolution.Daily)
                interest_rates.append(interest30_history["value"].dropna().mean())

        # 2D array: years*12 x n where n is number of predictive variables
        data = list(zip(spy_returns, spy_volatilities, vix_volatilities, spy_sharpes, spy_momentums, interest_rates))

        ## Scale data option
        # data = np.array(data)
        # scaler = preprocessing.StandardScaler().fit(data)
        # data_scaled = scaler.transform(data)

        # Train model
        model = mixture.BayesianGaussianMixture(n_components=4, covariance_type='full', weight_concentration_prior=1, random_state=0).fit(data)
        self.Log(str(model.means_))
        self.Log(str(model.covariances_))
        self.model_training = False
        return model

    def PredictModel(self):
        """Predict on one datapoint averaged from data from one month."""
        # Reset momentum rolling window
        self.mom_window.Reset()
        # Get history of all securities
        history = self.History(self.Securities.Keys, 30, Resolution.Daily)
        # Initialise test dataset - 1xn where n is number of predictive variables
        test_data = np.empty((1,6))

        #1. Monthly return of SPY (percent)
        market_return = history.loc["spy"].close.pct_change().dropna().mean()
        market_return = market_return * 100
        test_data[0,0] = market_return
        #2. Monthly volatility of SPY (percent)
        market_volatility = history.loc["spy"].close.pct_change().dropna().std()
        market_volatility = market_volatility * 100
        test_data[0,1] = market_volatility
        #3. Monthly VIX average for volatility
        vix = history.loc["vix"].close.mean()
        test_data[0,2] = vix
        #4. Sharpe ratio of SPY
        sharpe_ratio = (market_return - self.risk_free_rate) / market_volatility
        test_data[0,3] = sharpe_ratio
        #5. Momentum of SPY
        for time, price in history.loc["spy"]["close"].items():
            self.manual_mom.Update(time, price)
        momentum_list = [item.Value for item in self.mom_window]
        if momentum_list:
            test_data[0,4] = statistics.fmean(momentum_list)
        else: # No data, act as if neutral
            test_data[0,4] = 0
        #6. 30-day AA asset-backed commercial paper interest rate (percent)
        interest30_history = self.History(self.interest30, 30, Resolution.Daily)
        test_data[0,5] = interest30_history["value"].dropna().mean()

        return self.model.predict(test_data)[0]
