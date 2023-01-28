from AlgorithmImports import *
import numpy as np
import strategies
import rebalance
from sklearn import mixture
import pandas as pd


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
        # list of securities to be used for history function
        self.historytickers = [spy, tqqq, xagusd, ubt, ust]
        self.spy = spy
        
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

        # Model setup
        self.model_training = False
        self.model = self.train_model(2010, 23)
        

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
        self.market_condition = self.predict_model()
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

    def train_model(self, start_year, num_years):
            """Method to train model. Should only be called once at initialisation unless update needed.
            :param int start_year: Start year of training data in YYYY format
            :param int num_years: Number of years of training data to use
            """
            self.Debug(str(('Start training at {}'.format(self.Time))))
            self.model_training = True
            
            # We want historic data grouped by month. Can do this by requesting data for one month at a time or slicing the whole history array using datetime indices. Here, we will request a month at a time.
            years = list(map(str, range(start_year, start_year+num_years+1)))
            months = list(map(str, range(1, 13)))
            
            spy_returns = []
            spy_volatilities = []
            n = num_years*12
            # n by 62 array to store all data
            all_data = np.empty((n, 8))
            
            # Use S&P as market data
            for i in range(num_years):
                for j in range(12):
                    data_idx = i*12 + j
                    
                    start_date_str = years[i] + " " + months[j] + " 01"
                    start_date = Time.ParseDate(start_date_str)

                    end_date_str = years[i] + " " + months[j] + " 30"
                    end_date = Time.ParseDate(end_date_str)

                    market_history = self.History(self.spy, start_date, end_date, Resolution.Daily)

                    #1. Monthly return of the market
                    market_return = market_history.close.pct_change().dropna().mean() * 100
                    all_data[data_idx, 0] = market_return
                    spy_returns.append(market_return)
                    #2. Monthly volatility of the market
                    market_volatility = market_history.close.pct_change().dropna().std() * 100
                    all_data[data_idx, 1] = market_volatility
                    spy_volatilities.append(market_volatility)
                    
                    all_returns = []
                    all_volatilities = []
                    all_covariances = []
                    all_betas = []
                    all_alphas = []
                    all_sharpe_ratios = []
                    
                    # For each security
                    for security in self.historytickers[1:]: # omit first ticker (SPY)
                        history = self.History(security, start_date, end_date, Resolution.Daily)

                        #1. Monthly return of each security
                        monthly_return = history.close.pct_change().dropna().mean() * 100
                        all_returns.append(monthly_return)
                        #2. Monthly volatility of each security
                        monthly_volatility = history.close.pct_change().dropna().std() * 100
                        all_volatilities.append(monthly_volatility)
                        #3. Covariance of monthly returns of each security and the market
                        covariance = history.close.pct_change().dropna().cov(market_history.close.pct_change().dropna())
                        all_covariances.append(covariance)
                        #4. Beta of each security
                        beta = covariance / market_volatility
                        all_betas.append(beta)
                        #5. Alpha of each security
                        alpha = monthly_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))
                        all_alphas.append(alpha)
                        #6. Sharpe ratio of each security
                        sharpe_ratio = (monthly_return - self.risk_free_rate) / monthly_volatility
                        all_sharpe_ratios.append(sharpe_ratio)
                        # #7. Treynor ratio of each security
                        # treynor_ratio = (monthly_return - self.risk_free_rate) / beta
                        # #8. Information ratio of each security
                        # information_ratio = monthly_return / monthly_volatility
                        # #9. Sortino ratio of each security
                        # sortino_ratio = (monthly_return - self.risk_free_rate) / monthly_volatility
                        # #10. Jensen's alpha of each security
                        # jensens_alpha = (monthly_return - self.risk_free_rate) - (beta * (market_return - self.risk_free_rate))
                    
                    all_data[data_idx, 2] = np.mean(np.array(all_returns))
                    all_data[data_idx, 3] = np.mean(np.array(all_volatilities))
                    all_data[data_idx, 4] = np.mean(np.array(all_covariances))
                    all_data[data_idx, 5] = np.mean(np.array(all_betas))
                    all_data[data_idx, 6] = np.mean(np.array(all_alphas))
                    all_data[data_idx, 7] = np.mean(np.array(all_sharpe_ratios))
            
            model = mixture.BayesianGaussianMixture(n_components=4, covariance_type='full', random_state=0).fit(all_data)
            self.Debug(str(model.means_))
            self.model_training = False
            return model
    
    def predict_model(self):
        """Predict on one datapoint averaged from data from one month"""
        test_data = np.empty((1,2))
        test_history = self.History(self.spy, timedelta(days=30), Resolution.Daily)
        market_return = test_history.close.pct_change().dropna().mean()
        market_return = market_return * 100
        test_data[0,0] = market_return
        market_volatility = test_history.close.pct_change().dropna().std()
        market_volatility = market_volatility * 100
        test_data[0,1] = market_volatility
        self.Debug(str(self.model.predict(test_data)))
        return self.model.predict(test_data)[0]