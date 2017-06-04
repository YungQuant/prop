import numpy as np
import quandl
import urllib.request
import urllib, time, datetime
import scipy.stats as sp
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
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


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

def get_the_shit(prices, alloc, lookback, n, len_buys):
    predicts = []
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X, Y = create_dataset(prices[:n], lookback)
    # X = scaler.fit_transform(X)
    # R = Ridge(alpha=a, fit_intercept=True, normalize=True)
    R = LinearRegression(fit_intercept=True, normalize=True, n_jobs=8)
    R.fit(X, Y)
    predict = R.predict(prices[-lookback:])
    #predicts.append(predict)
    if alloc > 0:
        len_buys += 1
    if predict < prices[n] and alloc > 0:
        len_buys -= 1
    elif predict > prices[n] and alloc == 0:
        len_buys += 1

    return len_buys, predict

def duz_i_buy(cumulative_prices, n, allocs, tradeCost, lookback):
    cuml = sum(allocs)
    len_buys, indx = 0, 0
    predicts = []
    indx = 0
    while indx < len(allocs):
        updated_len_buys, predict = get_the_shit(cumulative_prices[indx], allocs[indx], lookback, n, len_buys)
        predicts.append(predict)
        len_buys = updated_len_buys
        indx += 1
    # len_buys, predicts = Parallel(n_jobs=8, verbose=8)(delayed(get_the_shit)
    #     (cumulative_prices[indx], allocs[indx], lookback, n, len_buys)
    #     for indx in range(len(allocs)))

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
            if (predicts[i] < cumulative_prices[i][n] and allocs[i] > 0) or (allocs[i] == 0):
                #print("started:", allocs[i])
                allocs[i] = 0
                #print("alloced:", allocs[i])
            if predicts[i] > cumulative_prices[i][n] and allocs[i] == 0:
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


def global_warming(ticker, cuml=1, tradeCost=0.0025, lookback=10, plt_bool=False):
    fileTicker = []
    fileOutput = []
    fileCuml = []
    dataset = []
    for r, tick in enumerate(ticker):
        if len(tick) < 9:
            fileTicker.append("../../data/" + tick + ".txt")
            fileOutput.append("../../output/" + tick + "envTest2.2_output.txt")
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

    avg_diffs = []; allocs = []; cumld = [];
    for z in range(len(ticker)):
        allocs.append(cuml / len(ticker))

    for n in range(min([int(np.floor(len(cumulative_diffs[f]) * 1)) for f in range(len(cumulative_diffs))])):
        if n > lookback:
            diffs = [cumulative_diffs[x][n] for x in range(len(cumulative_diffs))]
            for g in range(len(allocs)):
                allocs[g] += allocs[g] * diffs[g]
            cuml = sum(allocs)
            allocs = duz_i_buy(cumulative_prices, n, allocs, tradeCost, lookback)
            #print(allocs)
            cumld.append(sum(allocs))

    if plt_bool == True:
        plot(cumld, xLabel="Days", yLabel="Percent Gains (starts at 100%)")

    return cuml

ticker = ["BTC_ETH", "BTC_XEM", "BTC_XMR", "BTC_SJCX", "BTC_DASH", "BTC_XRP", "BTC_MAID", "BTC_LTC", "BCHARTS/BITSTAMPUSD"]
k1 = 80; k2 = 200; k = k1; results = [];
while k < k2:
    results.append(global_warming(ticker, 1, tradeCost=0.005, lookback=int(np.floor(k)), plt_bool=False))
    k += 1
    if results[-1] > 1:
        #global_warming(ticker, 1, tradeCost=0.005, lookback=int(np.floor(k)), plt_bool=True)
        for i in range(5):
            print("$$$$$$$$$$$$$$$$$")
        print("lookback:", k, "result:", results[-1])