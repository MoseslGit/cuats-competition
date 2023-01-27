import numpy as np

# Define function to execute trading strategy for crisis market condition
def crisis_strategy(current_portfolio, historical_data, securities):

    # Trend following, bonds, and gold
    # Select three symbols with greatest momentum

    updated_portfolio = {}
    momentum_comparison = {}

    for symbol in securities:
        prices = historical_data.loc[symbol]['close'].resample('d').last()
        returns = np.diff(prices) / prices[:-1]
        # Calculate the momentum over the last 2 weeks
        momentum_comparison[symbol] = momentum = returns[-14:].mean()
    # Put the three symbols with the greatest momentum into a list
    basket = sorted(momentum_comparison, key=momentum_comparison.get, reverse=True)[:3]
    for symbol in basket:
        updated_portfolio[symbol] = 0.33
    updated_portfolio["UBT", "UST"] = 0.33

    return updated_portfolio

# Define function to execute trading strategy for steady state market condition
def steady_state_strategy(current_portfolio, historical_data, securities):
    # Follow market, equities
    # Select three symbols with greatest momentum
    updated_portfolio = {}
    momentum_comparison = {}

    for symbol in securities:
        prices = historical_data.loc[symbol]['close'].resample('d').last()
        returns = np.diff(prices) / prices[:-1]
        # Calculate the momentum over the last 2 weeks
        momentum_comparison[symbol] = momentum = returns[-14:].mean()
    # Put the three symbols with the greatest momentum into a list
    basket = sorted(momentum_comparison, key=momentum_comparison.get, reverse=True)[:3]
    if "QQQ" not in basket:
        basket.append("QQQ")
    for symbol in basket:
        updated_portfolio[symbol] = 0.8

    return updated_portfolio

# Define function to execute trading strategy for inflation market condition
def inflation_strategy(current_portfolio, historical_data, securities):
    # Use inflation hedges and gold

    updated_portfolio = {}
    basket = ["GLD", "UBT", "UST"]
    for symbol in basket:
        updated_portfolio[symbol] = 0.8
    return updated_portfolio

# Define function to execute trading strategy for Walking on Ice market condition
def woi_strategy(current_portfolio, historical_data, securities):
    # Scale down on position sizes
    updated_portfolio = {}
    updated_portfolio["UST", "UBT"] = 0.2
    return updated_portfolio
