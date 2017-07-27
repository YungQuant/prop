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

def plot(a, xLabel = 'X', yLabel = 'Y'):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()

def plot2(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()

def plot3(a, b, c):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r', y, c, 'b')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()

def plot4(a, b, c, d):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r', y, c, 'b', y, d, 'y')
    plt.ylabel('Y')
    plt.xlabel('X')
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

def squash(allocs, cuml):
    new_allocs = []
    for i in range(len(allocs)):
        new_allocs.append(allocs[i] / cuml)
    return new_allocs

def write_that_shit(log, k, rebalsss, result, mdd):
    if os.path.isfile(log):
        th = 'a'
    else:
        th = 'w'
    file = open(log, th)
    file.write("K:\t")
    file.write(str(k))
    file.write("\nRebals:\t")
    file.write(str(rebalsss))
    file.write("\nResults:\t")
    file.write(str(result))
    file.write("\nMax Drawdown:\t")
    file.write(str(mdd))
    file.write("\n\n")
    file.close()

def global_warming(ticker, cuml=1, tradeCost=0.0025, rebal_tol=0.1, plt_bool=False):
    fileTicker = []
    fileOutput = []
    fileCuml = []
    dataset = []
    rebalsss = 0
    for r, tick in enumerate(ticker):
        if len(tick) < 9:
            fileTicker.append("../../data/" + tick + ".txt")
        elif len(tick) > 9:
            fileTicker.append("../../data/" + "BITSTAMP_USD_BTC.txt")

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
        #for i, stocks in enumerate(stock1[int(len(stock1) * 0.8):]):
        for i, stocks in enumerate(stock1[-120:]):
            stock.append(float(stocks))
        for u in range(len(stock) - 1):
            diffs.append((stock[u + 1] - stock[u]) / stock[u])
        cumulative_diffs.append(diffs)
        cumulative_prices.append(stock)

    avg_diffs = []; allocs = []; cumld = []; dd = 0; mdd = 0;
    for z in range(len(ticker)):
        allocs.append(cuml / len(ticker))

    for n in range(min([int(np.floor(len(cumulative_diffs[f]) * 1)) for f in range(len(cumulative_diffs))])):
        prices = [cumulative_prices[y][n] for y in range(len(cumulative_prices))]
        diffs = [cumulative_diffs[x][n] for x in range(len(cumulative_diffs))]
        for g in range(len(allocs)):
            allocs[g] += allocs[g] * diffs[g]
        cuml = sum(allocs)
        if np.var(squash(allocs, cuml)) > np.mean(squash(allocs, cuml)) * rebal_tol:
        #STILL NEEDS NP.VAR AND SP.KURTOSIS TESTING
            rebalsss += 1
            cumld.append(0)
            for m in range(len(allocs)):
                allocs[m] = ((cuml / len(allocs)) * (1 - tradeCost))
        #print(allocs)
        else:
            cumld.append(sum(allocs))

    for i in range(len(cumld)):
        if i > 1:
            peak = max(cumld[:i])
            trough = min(cumld[i:])
            dd = (peak - trough) / peak
            if dd > mdd:
                mdd = dd

    if plt_bool == True:
        plot(cumld, xLabel="Days", yLabel="Percent Gains (starts at 100%)")

    return cuml, rebalsss, mdd

ticker = ["BTC_ETH", "BTC_XEM", "BTC_XMR", "BTC_SJCX", "BTC_DASH", "BTC_XRP", "BTC_MAID", "BTC_LTC"]

result, rebalsss, mdd = global_warming(ticker, 1, tradeCost=0.005, rebal_tol=0.025, plt_bool=True)
result, rebalsss, mdd = global_warming(ticker, 1, tradeCost=0.005, rebal_tol=215, plt_bool=True)
