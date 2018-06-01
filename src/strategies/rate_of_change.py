def rate_of_change_strategy(prices, lookback):
    operating_array = prices[-(lookback * 10):]
    min_delta = 0
    max_delta = 0
    for i, price in enumerate(operating_array[:len(operating_array)-lookback]):
        delta = operating_array[i] - operating_array[i + lookback]
        min_delta = min(min_delta, delta)
        max_delta = max(max_delta, delta)
    return (min_delta, max_delta)

# Scale an input delta to a ratio of the max/min from price history
def scale_roc(delta, extreme_deltas):
    if delta > 0:
        return delta / extreme_deltas[1]
    return delta / extreme_deltas[0]