import numpy as np

def adjust(current_portfolio, historical_data, risk_free_rate, thresholds):
    
    risks, returns, diversification = calculate_factors(historical_data, current_portfolio)
    
    #calculate current sharpe ratios for each asset
    #calculate threshold sharpe ratios for each asset - cutoff point for when we reallocate asset, based on diverification, return, and potential risks.
    sharpe_ratios = {}
    threshold_sharpe_ratios = {}
    for symbol, weight in current_portfolio.items():
        sharpe_ratios[symbol] = (returns[symbol] - risk_free_rate) / risks[symbol]
        threshold_sharpe_ratios[symbol] = (thresholds['risk_factor'] * risks[symbol]) + (thresholds['return_factor'] * returns[symbol]) + (thresholds['diversification_factor'] * diversification[symbol])

    # Remove any assets that are underperforming, and replace them with greater allocations of "good" assets
    for symbol, weight in current_portfolio.items():
        prices = historical_data.loc[symbol]['close']
        macd = calculate_macd(prices, thresholds['short_window'], thresholds['long_window'])
        if sharpe_ratios[symbol] >= threshold_sharpe_ratios[symbol]:
            if macd > 0:
                current_portfolio[symbol] = weight*1.5
            else:
                current_portfolio[symbol] = weight*1.2
        else:
            if macd > 0:
                current_portfolio[symbol] = weight*0.8
            else:
                current_portfolio[symbol] = weight*0.5

    total_weight = max(sum(current_portfolio.values()), 1)
    rebalanced_portfolio = {key: value/total_weight for key, value in current_portfolio.items()}
    return rebalanced_portfolio

def numpy_ewma(data, window):

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
    prices = np.array(prices)
    # Calculate short window exponential moving average
    short_ema = numpy_ewma(prices, short_window).mean()

    # Calculate long window exponential moving average
    long_ema = numpy_ewma(prices, long_window).mean()

    # Calculate moving average convergence divergence
    macd = short_ema - long_ema

    return macd
    
def calculate_factors(historical_data, portfolio):
    risks = {}
    mean_returns = {}
    diversification = {}
    # Initialize empty list to store risks
    # Calculate standard deviation of returns for each asset
    for symbol in portfolio:
        prices = historical_data.loc[symbol]['close'].resample('d').last()
        returns = np.diff(prices) / prices[:-1]
        risks[symbol] = np.std(returns)
        mean_returns[symbol] = np.mean(returns)
    # Calculate pairwise correlations between asset and all other assets
        correlations = []
        for compare_symbol in portfolio:
            if symbol != compare_symbol:
                compare_prices = historical_data.loc[compare_symbol]['close'].resample('d').last()
                compare_returns = np.diff(compare_prices) / compare_prices[:-1]
                correlation = np.corrcoef(returns, compare_returns)[0][1]
                correlations.append(correlation)
        
    # Calculate average pairwise correlation
        avg_correlation = np.mean(correlations)
    
    # Calculate diversification benefit
        diversification_benefit = 1 - avg_correlation
        diversification[symbol] = diversification_benefit

    return risks, mean_returns, diversification
    



