import numpy as np
import quandl
import urllib.request
import urllib, time, datetime
import scipy.stats as sp
from matplotlib import pyplot as plt
import os.path
from multiprocessing import Pool
from joblib import Parallel, delayed
import yahoo_finance
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def plot(a, xLabel = 'Price', yLabel = 'Time Periods'):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()

def plot2(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r')
    plt.ylabel('Price')
    plt.xlabel('Time Periods')
    plt.show()

def getNum(str):
    tmp = ""
    for i, l in enumerate(str):
        if l.isnumeric() or l == ".":
            tmp += l

    if str[-4:].find('e-') > 0 or str[-4:].find('e+') > 0:
        tmp += str[-3:]

    return float(tmp)

def CryptoQuote1(the_symbol):
    class ohlcvObj():
        open, high, low, close, volume = [], [], [], [], []
    the_url = "https://poloniex.com/public?command=returnChartData&currencyPair={0}&start=1435699200&end=9999999999&period=86400".format(the_symbol)
    response = urllib.request.urlopen(the_url).read().decode("utf-8").split(",")
    print(the_symbol, response[0:10])
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

def global_warming(ticker, cuml=1, tradeCost=0.0025, rebal_tol=0.1, perf_fee=0.2, plt_bool=False):
    profit = 0
    fileTicker = []
    fileOutput = []
    fileCuml = []
    dataset = []
    hist_vals = [100000000, 10000000]
    for r, tick in enumerate(ticker):
        if len(tick) < 9:
            fileTicker.append("../../data/" + tick + ".txt")
            fileOutput.append("../../output/" + tick + "envTest1_output.txt")
        elif len(tick) > 9:
            fileTicker.append("../../data/" + "BITSTAMP_USD_BTC.txt")
            fileOutput.append("../../output/" + "BITSTAMP_USD_BTC_envTest1_output.txt")

    for i, file in enumerate(fileTicker):
        if (os.path.isfile(file) == False):
            fileWrite = open(file, 'w')
            if len(ticker[i]) < 9:
                dataset = CryptoQuote1(ticker[i]).close
            elif len(ticker[i]) > 9:
                data = quandl.get(ticker[i], column_index=4, exclude_column_names=True)
                data = np.array(data)
                for i in range(len(data)):
                    if float(data[i][-6:]) > 0:
                        dataset.append(float(data[i][-6:]))

            # tick = yahoo_finance.Share(ticker[i]).get_historical('2015-01-02', '2017-01-01')
            # dataset = np.zeros(len(tick))
            # i = len(tick) - 1
            # ik = 0
            # while i >= 0:
            #     dataset[ik] = tick[i]['Close']
            #     i -= 1
            #     ik += 1
            for l, close in enumerate(dataset):
                fileWrite.write(str(close))
                fileWrite.write('\n')

    cumulative_diffs = []; cumulative_prices = [];
    for y in range(len(fileTicker)):
        stock = []; diffs = [];
        with open(fileTicker[y], 'r') as f:
            stock1 = f.readlines()
        f.close()
        for i, stocks in enumerate(stock1):
            stock.append(float(stocks))
        for u in range(len(stock) - 1):
            diffs.append((stock[u + 1] - stock[u]) / stock[u])
        cumulative_diffs.append(diffs)
        cumulative_prices.append(stock)

    avg_diffs = []; allocs = []; cumld = [];
    for z in range(len(ticker)):
        allocs.append(cuml / len(ticker))

    for n in range(min([int(np.floor(len(cumulative_diffs[f]) * 1)) for f in range(len(cumulative_diffs))])):
        prices = [cumulative_prices[y][n] for y in range(len(cumulative_prices))]
        diffs = [cumulative_diffs[x][n] for x in range(len(cumulative_diffs))]
        for g in range(len(allocs)):
            allocs[g] += allocs[g] * diffs[g]
        cuml = sum(allocs)
        hist_vals.append(cuml)
        if hist_vals[-1] > hist_vals[-2]:
            profit += (hist_vals[-1] - hist_vals[-2]) * perf_fee
            cuml -= profit
        if max(allocs) - min(allocs) > np.mean(allocs) * rebal_tol:
            for m in range(len(allocs)):
                allocs[m] = ((cuml / len(allocs)) * (1 - tradeCost))
        #print(allocs)
        cumld.append(sum(allocs))

    if plt_bool == True:
        plot(cumld, xLabel="Days", yLabel="Percent Gains (starts at 100%)")

    return cuml, profit

ticker = ["BTC_ETH", "BTC_XMR", "BTC_XRP", "BTC_MAID", "BTC_LTC", "BCHARTS/BITSTAMPUSD"]
k1 = 0.001; k2 = 100; k = k1; o1 = 0.001; o2 = 1; o = o1; results = [0, 0, 0]; profits = [0, 0, 0];
while o < o2:
    while k < k2:
        result, profit = global_warming(ticker, 1, tradeCost=0.005, rebal_tol=k, perf_fee=o, plt_bool=False)
        profits.append(profit)
        results.append(result)
        k += 0.01
        if len(results) > 1 and results[-1] > max(results[:-1]) and profits[-1] > max(profits[:-1]):
        #if len(results) > 1 and profits[-1] > max(profits[:-1]):
            print("rebal_tol:", k, "perf_fee:", o, "result:", results[-1], "profit:", profit)
    o += 0.01
    k = k1
    print(o, "/", o2)
