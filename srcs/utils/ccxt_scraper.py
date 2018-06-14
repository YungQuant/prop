import ccxt
import os

def filter_non_btc(currencies):
    new_currencies = []
    for i, currency in enumerate(currencies):
        if currency[-3:] == "BTC":
            new_currencies.append(currency)
    return new_currencies

binance_client = ccxt.binance()
currencies = binance_client.fetch_markets()
currencies = [currency['symbol'] for currency in currencies]
currencies = filter_non_btc(currencies)

for i in range(len(currencies)):
    data = binance_client.fetch_ohlcv(currencies[i], "5m")
    fileName = f'../../data/{currencies[i][:-4]}_5m_OHLCV.txt'
    if os.path.isfile(fileName) == False:
        aw_bool = "w"
    else:
        aw_bool = "a"
    fileWrite = open(fileName, aw_bool)
    fileWrite.write([data[i]] for)
    fileWrite.close()