import ccxt

def filter_non_btc(currencies):
    new_currencies = []
    for i, currency in enumerate(currencies):
        if currency[-3:] == "BTC":
            new_currencies.append(currency)
    return new_currencies

def hs1(prof_goal):
    binance_client = ccxt.binance()
    currencies = binance_client.fetch_markets()
    currencies = [currency['symbol'] for currency in currencies]
    currencies = filter_non_btc(currencies)
    print(currencies)

hs1(1)

#hs1 should go through every currency available, buy a bag, wait fon an x% gain in price, and sell that position
#hstf1 should simulate that ^ process on historical data and record metrics (namely profitability)