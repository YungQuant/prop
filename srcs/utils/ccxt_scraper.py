import ccxt
import os

def filter_non_btc(currencies):
    new_currencies = []
    for i, currency in enumerate(currencies):
        if currency[-3:] == "BTC":
            new_currencies.append(currency)
    return new_currencies

client = ccxt.binance()
currencies = client.fetch_markets()
currencies = [currency['symbol'] for currency in currencies]
currencies = filter_non_btc(currencies)
interval = "1w"

for i in range(len(currencies)):
    data = client.fetch_ohlcv(currencies[i], interval)
    fileName = f'../../data/{currencies[i][:-4]}_{interval}_OHLCV.txt'
    print(f'fetching {currencies[i]} {interval} OHLCV data')

    if os.path.isfile(fileName) == False:
        aw_bool = "w"
    else:
        print(f'{fileName} data file already exists')
        break

    fileWrite = open(fileName, aw_bool)
    fileWrite.write(str([datum for datum in data]))
    fileWrite.close()