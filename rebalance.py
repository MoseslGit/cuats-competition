import numpy as np

def adjust(portfolio, data, risk_free_rate, thresholds):
    
    risks = calculate_risk(data)
    returns = calculate_return(data)
    diversification = calculate_diversification(data)
    sharpe_ratios = (returns - risk_free_rate) / risks
    
    # Threshold sharpe ratio as cutoff point for when we reallocate asset, based on diverification, return, and potential risks.
    threshold_sharpe_ratios = (thresholds['risk_factor'] * risks) + (thresholds['return_factor'] * returns) + (thresholds['diversification_factor'] * diversification)

    new_weights = []
    
    # Remove any assets that are underperforming, and replace them with greater allocations of "good" assets
    for i, symbol in enumerate(data.Keys):
        if sharpe_ratios[i] >= threshold_sharpe_ratios[i]:
            new_weight = 1 / len(data.Keys)  # equally weight all assets
        else:
            new_weight = 0
            
        new_weights.append(new_weight)
    
    # Normalize new weights to sum to 1
    new_weights = new_weights / np.sum(new_weights)
    
    # Update portfolio
    portfolio = new_weights
    # Calculate MACD for each asset
    macd_values = []
    for symbol in data.Keys:
        prices = data[symbol].Prices
        macd = calculate_macd(prices, thresholds['short_window'], thresholds['long_window'])
        macd_values.append(macd)

    # Initialize empty list to store MACD weights
    macd_weights = []

    # Calculate MACD weights for each asset
    for i, symbol in enumerate(data.Keys):
        if macd_values[i] > 0:
            macd_weight = 1 / len(data.Keys)  # equally weight all assets
        else:
            macd_weight = 0
        macd_weights.append(macd_weight)

    # Normalize MACD weights to sum to 1
    macd_weights = macd_weights / np.sum(macd_weights)

    # Combine Sharpe ratio weights and MACD weights
    combined_weights = (portfolio + macd_weights) / 2

    # Update portfolio
    portfolio = combined_weights

    return portfolio

def calculate_macd(prices, short_window, long_window):
    # Calculate short window exponential moving average
    short_ema = prices.ewm(span=short_window, adjust=False).mean()

    # Calculate long window exponential moving average
    long_ema = prices.ewm(span=long_window, adjust=False).mean()

    # Calculate moving average convergence divergence
    macd = short_ema - long_ema

    return macd
    
def calculate_risk(data, portfolio):
    # Initialize empty list to store risks
    risks = []
    
    # Calculate standard deviation of returns for each asset
    for symbol in portfolio:
        returns = data[symbol].Returns
        risk = np.std(returns)
        risks.append(risk)
        
    return risks

def calculate_return(data, portfolio):
    # Initialize empty list to store returns
    returns = []
    
    # Calculate mean of returns for each asset
    for symbol in portfolio:
        asset_returns = data[symbol].Returns
        mean_return = np.mean(asset_returns)
        returns.append(mean_return)
        
    return returns

def calculate_diversification(data, portfolio):
    # Initialize empty list to store diversification benefits
    diversification = []
    
    # Calculate diversification benefits for each asset
    for symbol in portfolio:
        # Calculate pairwise correlations between asset and all other assets
        correlations = []
        for compare_symbol in portfolio:
            if symbol != compare_symbol:
                asset_returns = data[symbol].Returns
                compare_asset_returns = data[compare_symbol].Returns
                correlation = np.corrcoef(asset_returns, compare_asset_returns)[0][1]
                correlations.append(correlation)
        
        # Calculate average pairwise correlation
        avg_correlation = np.mean(correlations)
        
        # Calculate diversification benefit
        diversification_benefit = 1 - avg_correlation
        diversification.append(diversification_benefit)
        
    return diversification
