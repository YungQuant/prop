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

def write_that_shit(results, k, l):
    log = "../../output/PAMRtest1_v2_output.txt"
    if os.path.isfile(log):
        th = 'a'
    else:
        th = 'w'
    file = open(log, th)
    file.write("\n\n Cumulative Diff:\t")
    file.write(str(results))
    file.write("\nepsilon:\t")
    file.write(str(l))
    file.write("\nC:\t")
    file.write(str(k))
    file.write("\n\n")
    file.close()


def simplex_proj(y):
    """ Projection of y onto simplex. """
    m = len(y)
    bget = False
    s = sorted(y, reverse=True)
    tmpsum = 0

    for ii in range(m - 1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1);
        if tmax >= s[ii + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m

    return np.maximum(y - tmax, 0)

# def simplex_proj(y):
#     """ Projection of y onto simplex. """
#     m = len(y)
#     bget = False
#
#     s = sorted(y, reverse=True)
#     tmpsum = 0.
#
#     for ii in range(m - 1):
#         tmpsum = tmpsum + s[ii]
#         tmax = (tmpsum - 1) / (ii + 1);
#         if tmax >= s[ii + 1]:
#             bget = True
#             break
#
#     if not bget:
#         tmax = (tmpsum + s[m - 1] - 1) / m
#
#     return np.maximum(y - tmax, 0.)

def simplex_proj1(a, y):
    l = y / a
    idx = np.argsort(l)
    d = len(l)

    evalpL = lambda k: np.sum(a[idx[k:]] * (y[idx[k:]] - l[idx[k]] * a[idx[k:]])) - 1

    def bisectsearch():
        idxL, idxH = 0, d - 1
        L = evalpL(idxL)
        H = evalpL(idxH)

        if L < 0:
            return idxL

        while (idxH - idxL) > 1:
            iMid = int((idxL + idxH) / 2)
            M = evalpL(iMid)

            if M > 0:
                idxL, L = iMid, M
            else:
                idxH, H = iMid, M

        return idxH

    k = bisectsearch()
    lam = (np.sum(a[idx[k:]] * y[idx[k:]]) - 1) / np.sum(a[idx[k:]])

    x = np.maximum(0, y - lam * a)

    return x

def duz_i_buy(allocs, C, epsilon, diffs):
    var = 2
    cuml = sum(allocs)
    mean_diff = np.mean(diffs)
    le = max([0, np.dot(allocs, diffs) - epsilon])
    if var == 0:
        lam = le / np.linalg.norm(diffs - mean_diff) ** 2
    elif var == 1:
        lam = min(C, le / np.linalg.norm(diffs - mean_diff) ** 2)
    elif var == 2:
        lam = le / (np.linalg.norm(diffs - mean_diff) ** 2 + 0.5 / C)

    lam = min([100000, lam])
    allocs = allocs - lam * (diffs - mean_diff)
    # print("lamda:", lam, "le:", le)
    # print("mid-calc cuml:", sum(allocs), "mid-calc allocs:", allocs)
    # x_mean = np.mean(x)
    # le = max(0., np.dot(b, x) - eps)
    #
    # if self.variant == 0:
    #     lam = le / np.linalg.norm(x - x_mean) ** 2
    # elif self.variant == 1:
    #     lam = min(C, le / np.linalg.norm(x - x_mean) ** 2)
    # elif self.variant == 2:
    #     lam = le / (np.linalg.norm(x - x_mean) ** 2 + 0.5 / C)
    #
    # # limit lambda to avoid numerical problems
    # lam = min(100000, lam)
    #
    # # update portfolio
    # b = b - lam * (x - x_mean)
    #
    # # project it onto simplex
    # return tools.simplex_proj(b)
    return simplex_proj(allocs)
    #return simplex_proj1(allocs, allocs * (np.ones(len(diffs)) - diffs))


def global_warming(ticker, cuml=1, C=500, epsilon=0.5, plt_bool=False):
    fileTicker = []
    fileOutput = []
    fileCuml = []
    dataset = []
    profits = 0
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
        if n > 2:
            diffs = [cumulative_diffs[x][n] for x in range(len(cumulative_diffs))]
            for g in range(len(allocs)):
                allocs[g] += allocs[g] * diffs[g]

            pre_cuml = sum(allocs)
            # print("allocs IN:", allocs)
            # print("cuml IN:", cuml)
            allocs = duz_i_buy(allocs, C, epsilon, diffs)
            post_cuml = sum(allocs)
            profits += pre_cuml - post_cuml
            allocs *= 0.995
            # for i in range(len(allocs)):
            #     allocs[i] += profits / len(allocs)
            # print("cuml OUT:", post_cuml)
            # print("allocs OUT:", allocs)
            # print("\n")
            cumld.append(sum(allocs) + profits)

    if plt_bool == True:
        plot(cumld, xLabel="Days", yLabel="Percent Gains (starts at 100%)")

    return cumld[-1]

ticker = ["BTC_ETH", "BTC_XEM", "BTC_XMR", "BTC_SJCX", "BTC_DASH", "BTC_XRP", "BTC_MAID", "BTC_LTC"]
k1 = 400; k2 = 600; k = k1; l1 = 0.1; l2 = 0.9; l = l1; results = [];
while k < k2:
    while l < l2:
        results.append(global_warming(ticker, 1, C=int(np.floor(k)), epsilon=l, plt_bool=False))
        if results[-1] > np.mean(results) or len(results) % 10 == 0:
            if results[-1] > np.mean(results):
                #global_warming(ticker, 1, C=int(np.floor(k)), epsilon=l,plt_bool=True)
                write_that_shit(results[-1], int(np.floor(k)), l)
                for i in range(5):
                    print("$$$$$$$$$$$$$$$$$")
            print("C:", k, "epsilon:", l, "result:", results[-1])
        l += 0.1
    l = l1
    k += 1

plot(results)