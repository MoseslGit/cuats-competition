import numpy as np

# Execute trading strategy for crisis market condition
def crisis_strategy(historical_data, securities):

    # Short equities, small allocation to bonds and gold as we go risk off
    updated_portfolio = {}

    # Set initial weights for rebalancing
    updated_portfolio["SPY"] = -0.2
    updated_portfolio["XAGUSD"] = 0.2
    updated_portfolio["UST"] = 0.2


    return updated_portfolio

# Execute trading strategy for steady state market condition
def steady_state_strategy(historical_data, securities):
    
    # Follow market, biased towards tech equities
    updated_portfolio = {}
    momentum_comparison = {}

    # Calculate momentum for each security
    for symbol in securities:
        if symbol in historical_data:
            prices = historical_data.loc[symbol]['close'].resample('d').last()
            returns = np.diff(prices) / prices[:-1]
            # Calculate the momentum over the last 2 weeks
            momentum_comparison[symbol] == returns[-14:].mean()

    # Put the three symbols with the greatest momentum into a list
    basket = sorted(momentum_comparison, key=momentum_comparison.get, reverse=True)[:3]

    # Bias the portfolio towards tech equities
    if "TQQQ" not in basket:
        basket.append("TQQQ")

    # Even out weights
    for symbol in basket:
        updated_portfolio[symbol] = 1/len(basket)

    return updated_portfolio

# Execute trading strategy for inflation market condition
def inflation_strategy(historical_data, securities):
    
    # Buy Gold as an inflation hedge, and invest in bonds
    updated_portfolio = {}

    updated_portfolio["XAGUSD"] = 0.5
    updated_portfolio["UBT"] = 0.25
    updated_portfolio["UST"] = 0.25

    return updated_portfolio

# Execute trading strategy for Walking on Ice market condition
def woi_strategy(historical_data, securities):
    # High volatility and uncertainty, so we go risk off with a bias towards bonds
    updated_portfolio = {}

    updated_portfolio["UBT"] = 0.33
    updated_portfolio["UST"] = 0.33

    return updated_portfolio
