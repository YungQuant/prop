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

def SMAn(a, n):                         #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    if (len(a) < n):
        n = len(a)
    n = int(np.floor(n))
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

def BBn(a, n, stddevD, stddevU): #GETS BOLLINGER BANDS OF "N" PERIODS AND STDDEV"UP" OR STDDEV"DOWN" LENGTHS
    cpy = a[-n:] #RETURNS IN FORMAT: LOWER BAND, MIDDLE BAND, LOWER BAND
    midb = SMAn(a, n) #CALLS SMAn
    std = np.std(cpy)
    ub = midb + (std * stddevU)
    lb = midb - (std * stddevD)
    return lb, midb, ub

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

def write_that_shit(results, v, k, l):
    log = "../../output/envTest2.1_output.txt"
    if os.path.isfile(log):
        th = 'a'
    else:
        th = 'w'
    file = open(log, th)
    file.write("\n\n Cumulative Diff:\t")
    file.write(str(results))
    file.write("\nstddev:\t")
    file.write(str(l))
    file.write("\nlookback:\t")
    file.write(str(k))
    file.write("\nbitchCunt:\t")
    file.write(str(v))
    file.write("\n\n")
    file.close()


def duz_i_buy(cumulative_prices, n, allocs, bitchCunts, tradeCost, lookback, stddev):
    cuml = sum(allocs)
    len_buys = 0
    for i in range(len(allocs)):
        op_arr = cumulative_prices[i][:n]
        lb, mb, ub = BBn(op_arr, lookback, stddev, stddev)
        #print("curr price 111:", cumulative_prices[i][n], "upper, middle, lower", ub, mb, lb)
        if allocs[i] > 0:
            len_buys += 1
        if allocs[i] < bitchCunts[i] and allocs[i] > 0:
            len_buys -= 1
        elif cumulative_prices[i][n] > ub and allocs[i] == 0:
            len_buys += 1

        #print("Len buys post fuckery:", len_buys)

    #print("len buys:", len_buys, "cuml:", cuml)

    if len_buys < 1:
        res = []
        for i in range(len(allocs) - 1):
            #print("LIQUIDATING TO BITCOIN")
            res.append(0)
        res.append(cuml)
        return res
    else:
        for i in range(len(allocs)):
            op_arr = cumulative_prices[i][:n]
            lb, mb, ub = BBn(op_arr, lookback, stddev, stddev)
            #print("curr price:", cumulative_prices[i][n], "upper, middle, lower", ub, mb, lb)
            if (allocs[i] < bitchCunts[i] and allocs[i] > 0) or (allocs[i] == 0):
                #print("started:", allocs[i])
                allocs[i] = 0
                #print("alloced:", allocs[i])
            elif cumulative_prices[i][n] > ub and allocs[i] == 0:
                #print("started:", allocs[i])
                allocs[i] = cuml * (1 - tradeCost) / len_buys
                #print("alloced:", allocs[i])
            elif allocs[i] > 0:
                #print("started:", allocs[i])
                allocs[i] = cuml * (1 - tradeCost) / len_buys
                #print("alloced:", allocs[i])

    # if sum(allocs) != cuml * (1 - tradeCost):
    #     print("ALLOCATION CALCULATION ERROR sum(allocs)/cuml * (1 - tradeCost):", sum(allocs), "/", cuml * (1 - tradeCost))
    return allocs


def global_warming(ticker, cuml=1, bitchCunt=0.1, tradeCost=0.0025, lookback=10, stddev=3, plt_bool=False):
    fileTicker = []
    fileOutput = []
    fileCuml = []
    dataset = []
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

    cumulative_prices = []; cumulative_diffs = [];
    for y in range(len(fileTicker)):
        stock = []; diffs = [];
        with open(fileTicker[y], 'r') as f:
            stock1 = f.readlines()
        f.close()
        for i, stocks in enumerate(stock1[int(np.floor(len(stock1) * 0)):]):
            stock.append(float(stocks))
        for u in range(len(stock) - 1):
            diffs.append((stock[u + 1] - stock[u]) / stock[u])
        cumulative_diffs.append(diffs)
        cumulative_prices.append(stock)

    avg_diffs = []; allocs = []; cumld = []; bitchCunts = [];
    for z in range(len(ticker)):
        allocs.append(cuml / len(ticker))
        bitchCunts.append(0)

    for n in range(min([int(np.floor(len(cumulative_diffs[f]) * 1)) for f in range(len(cumulative_diffs))])):
        if n > lookback:
            diffs = [cumulative_diffs[x][n] for x in range(len(cumulative_diffs))]
            for g in range(len(allocs)):
                allocs[g] += allocs[g] * diffs[g]
                if allocs[g] > 0 and (allocs[g] * (1 - bitchCunt)) > bitchCunts[g]:
                    bitchCunts[g] = allocs[g] * (1 - bitchCunt)
                elif allocs[g] == 0:
                    bitchCunts[g] = 0
            cuml = sum(allocs)
            allocs = duz_i_buy(cumulative_prices, n, allocs, bitchCunts, tradeCost, lookback, stddev)
            #print(allocs)
            cumld.append(sum(allocs))

    if plt_bool == True:
        plot(cumld, xLabel="Days", yLabel="Percent Gains (starts at 100%)")

    return cuml

ticker = ["BTC_ETH", "BTC_XEM", "BTC_XMR", "BTC_SJCX", "BTC_DASH", "BTC_XRP", "BTC_MAID", "BTC_LTC", "BCHARTS/BITSTAMPUSD"]
v1 = 0.001; v2 = 0.4; v = v1; k1 = 10; k2 = 300; k = k1; l1 = 1; l2 = 10; l = l1; results = [];
while k < k2:
    while l < l2:
        while v < v2:
            results.append(global_warming(ticker, 1, bitchCunt=v, tradeCost=0.005, lookback=int(np.floor(k)), stddev=l, plt_bool=False))
            v *= 1.2
            if results[-1] > max(results) or len(results) % 10 == 0:
                if results[-1] > 4.85:
                    global_warming(ticker, 1, bitchCunt=v, tradeCost=0.005, lookback=int(np.floor(k)), stddev=l,plt_bool=True)
                    write_that_shit(results[-1], v, int(np.floor(k)), l)
                    for i in range(5):
                        print("$$$$$$$$$$$$$$$$$")
                print("lookback:", k, "stddev:", l, "bitchCunt:", v, "result:", results[-1])
        v = v1
        l *= 1.2
    l = l1
    k *= 1.1

plot(results)