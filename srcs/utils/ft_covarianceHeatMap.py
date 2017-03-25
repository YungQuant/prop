import numpy as np
import urllib
from urllib import request
import matplotlib.pyplot as plt
import os, copy


def  getNum(str):
    tmp = ""
    for i, l in enumerate(str):
        if l.isnumeric() or l == ".":
            tmp += l
    return float(tmp)



def CryptoQuote1(the_symbol):
    class ohlcvObj():
        open, high, low, close, volume = [], [], [], [], []
    the_url = "https://poloniex.com/public?command=returnChartData&currencyPair={0}&start=1435699200&end=9999999999&period=300".format(the_symbol)
    response = urllib.request.urlopen(the_url).read().decode("utf-8").split(",")
    print(the_url)
    for i, curr in enumerate(response):
        if curr.find('open') > 0:
            ohlcvObj.open.append(getNum(curr))
        elif curr.find('high') > 0:
            ohlcvObj.high.append(getNum(curr))
        elif curr.find('low') > 0:
            ohlcvObj.low.append(getNum(curr))
        elif curr.find('close') > 0:
            ohlcvObj.close.append(getNum(curr))
        elif curr.find('volume') > 0:
            ohlcvObj.volume.append(getNum(curr))
    return ohlcvObj

ticker = ["BTC_ETH", "BTC_XMR", "BTC_DASH", "BTC_XRP", "BTC_FCT", "BTC_MAID", "BTC_ZEC", "BTC_LTC"]
fileTicker = []
for i, tick in enumerate(ticker):
    fileTicker.append("../../data/" + tick + ".txt")
for i, file in enumerate(fileTicker):
    if (os.path.isfile(file) == False):
        fileWrite = open(file, 'w')
        dataset = CryptoQuote1(ticker[i]).close
        print(dataset[1:10])
        # tick = yahoo_finance.Share(ticker[i]).get_historical('2015-01-02', '2017-01-01')
        # dataset = np.zeros(len(tick))
        # i = len(tick) - 1
        # while i >= 0:
        #     ik = 0
        #     dataset[ik] = tick[i]['Close']
        #     i -= 1
        #     ik += 1
        for i, close in enumerate(dataset):
            fileWrite.write(str(close))
            fileWrite.write('\n')
        fileWrite.close()

stock = []; size = 10000;
tickers = np.zeros((len(ticker), size), dtype=object)
for i, tik in enumerate(fileTicker):
    with open(tik, 'r') as f:
        stock1 = f.readlines()
    f.close()
    for k in range(len(stock1)):
        stock.append(getNum(stock1[k]))
    tickers[i] = np.array(stock[-size:])
    #tickers[i] = np.random.uniform(0, 10, size=1000)

#COVARIANCE ARRAY OF ALL CRYPTO'S IN TESTING ENVIRONMENT \/
Arr = np.corrcoef(np.array(tickers.astype(float)))
print(ticker)
for i in range(len(ticker)):
    print(ticker[i], Arr[i])

#HEATMAP \/
plt.pcolor(Arr, cmap=plt.cm.Blues)
plt.show()


