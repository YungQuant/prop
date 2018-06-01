#TIME WEIGHTED AVERAGE PRICE
def twap(arr, ll):
    a = arr[-ll:]
    high = max(a)
    low = min(a)
    close = a[len(a) - 1]
    return (high + low + close) / 3