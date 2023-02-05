#region imports
from AlgorithmImports import *
#endregion
import numpy as np

# Rebalance portfolio based on current portfolio performance
def adjust(current_portfolio, market_condition, historical_data, risk_free_rate, thresholds, portfolio_returns):
    """Rebalance portfolio according to performance.
    :param dict[str, float] current_portfolio: symbols and weights of currently held portfolio
    :param int market_condition: current market situation
    :param dataframe historical_data: last 14 days of market data on securities held
    :param int risk_free_rate: risk free rate used for calculations
    :param dict[str, float] thresholds: threshold values
    :param int portfolio_returns: value of portfolio returns between each period
    :return dict[str, float] current_portfolio: symbols and weights of new portfolio
    """
    # Calculate performance factors for each asset
    risks, returns, diversification = calculate_factors(historical_data, current_portfolio)
    
    # Calculate current sharpe ratios for each asset
    # Calculate threshold sharpe ratios for each asset - cutoff point for when we reallocate asset, based on diverification, return, and potential risks.
    sharpe_ratios = {}
    threshold_sharpe_ratios = {}
    for symbol, weight in current_portfolio.items():
        sharpe_ratios[symbol] = (returns[symbol] - risk_free_rate) / risks[symbol]
        threshold_sharpe_ratios[symbol] = (thresholds['risk_factor'] * risks[symbol]) + (thresholds['return_factor'] * returns[symbol]) + (thresholds['diversification_factor'] * diversification[symbol])

    # Remove any assets that are underperforming, and replace them with greater allocations of "good" assets
    # Depending on the market condition, we may want to reallocate more or less aggressively
    for symbol, weight in current_portfolio.items():

        # In crisis market, bias towards selling; in steady states bias towards top momentum performers
        # In high inflation situations, rebalance less; in high volatility situations tend to reduce size
        strong_buy = [1.2, 1.5, 1.2, 1.2]
        buy = [1, 1, 1, 1]
        sell = [-1.2, 1, 0.8, 0.8]
        strong_sell = [-1.5, 0.6, 0.7, 0.7]

        # Calculate MACD for each asset
        prices = historical_data.loc[symbol]['close']
        macd = calculate_macd(prices, thresholds['short_window'], thresholds['long_window'])

        # If the sharpe ratio is above the threshold, and the MACD is positive, increase allocation to asset
        if macd > 0:
            if sharpe_ratios[symbol] >= threshold_sharpe_ratios[symbol]:
                current_portfolio[symbol] = abs(weight)*strong_buy[market_condition - 1]
            else:
                current_portfolio[symbol] = abs(weight)*buy[market_condition - 1]
        # If the sharpe ratio is below the threshold, and the MACD is negative, decrease allocation to asset
        else:
            if sharpe_ratios[symbol] <= threshold_sharpe_ratios[symbol]:
                current_portfolio[symbol] = abs(weight)*strong_sell[market_condition - 1]
            else:
                current_portfolio[symbol] = abs(weight)*sell[market_condition - 1]
                
        # If the portfolio is doing well, scale up
        if portfolio_returns > 0:
            if market_condition == 2:
                current_portfolio[symbol] = abs(weight)*0.6
            else:
                current_portfolio[symbol] = abs(weight)*1.2
        else:
            current_portfolio[symbol] = abs(weight)*0.5

    return current_portfolio

def numpy_ewma(data, window):
    """Strategy in crisis situations. Called on market_condition.
    Take exponential weighted moving average, using in MACD calculation.
    :param array data: array to calculate EWMA on
    :param int window: what window to use
    """
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def calculate_macd(prices, short_window, long_window):
    """Calculate Moving Average Covergence Divergence (MACD).
    :param array prices: historical prices for assets
    :param int short_window: short window for EWMA
    :param int long_wondow: long window for EWMA
    """
    prices = np.array(prices)
    # Calculate short window exponential moving average
    short_ema = numpy_ewma(prices, short_window).mean()

    # Calculate long window exponential moving average
    long_ema = numpy_ewma(prices, long_window).mean()

    # Calculate moving average convergence divergence
    macd = short_ema - long_ema

    return macd

def calculate_factors(historical_data, portfolio):
    """Calculate sharpe-related performance factors for each asset.
    :param array historical_data: 30-day data for securities
    :param dict portfolio: list of securities and their respective weights
    """
    risks = {}
    mean_returns = {}
    diversification = {}

    for symbol in portfolio:
        prices = historical_data.loc[symbol]['close'].resample('d').last()
        returns = np.diff(prices) / prices[:-1]

        # Get risk and return for asset
        risks[symbol] = np.std(returns)
        mean_returns[symbol] = np.mean(returns)

        # Calculate pairwise correlations between asset and all other assets
        correlations = []
        for compare_symbol in portfolio:
            if symbol != compare_symbol:
                compare_prices = historical_data.loc[compare_symbol]['close'].resample('d').last()
                compare_returns = np.diff(compare_prices) / compare_prices[:-1]

                # Make sure the two assets have the same number of returns
                if len(returns) > len(compare_returns):
                    returns = returns[:len(compare_returns)]
                elif len(compare_returns) > len(returns):
                    compare_returns = compare_returns[:len(returns)]
                correlation = np.corrcoef(returns, compare_returns)[0][1]
                correlations.append(correlation)
        
    # Calculate average pairwise correlation
        avg_correlation = np.mean(correlations)
    
    # Calculate diversification benefit
        diversification_benefit = 1 - avg_correlation
        diversification[symbol] = diversification_benefit

    return risks, mean_returns, diversification



