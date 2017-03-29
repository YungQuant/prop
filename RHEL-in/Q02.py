import numpy as np
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

def plot(a):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('Price')
    plt.xlabel('Time Periods')
    plt.show()

def plot2(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r')
    plt.ylabel('Price')
    plt.xlabel('Time Periods')
    plt.show()

def wildersSmoothingN(a, n): #COMPUTES WILDERS SMOOTHING, COMPARABLE TO EMA WITH DIFFERENT VALUES
    l = len(a)
    e = np.zeros(l)
    m = 1 / n
    for i in range(l):
        if i < n:
            e[i] = a[i]
        else:
            y = (a[i - 1] * m) + (a[i - 2] * (1 - m))
            e[i] = (a[i] * m) + (y * (1 - m))
    return e

def EMAn(a, n): #GETS EXPONENTIAL MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    l = len(a)
    e = 0
    m = 2 / (n + 1)
    if n < l:
        y = SMAn(a, n)
        e = (a[len(a) - 1] * m) + (y * (1 - m))
    return e

def rsiN(a, n): #GETS RSI VALUE FROM "N" PERIODS OF "A" ARRAY
    n = int(np.floor(n))
    cpy = a[-n:]
    l = len(cpy)
    lc, gc, la, ga = 0.01, 0.01, 0.01, 0.01
    for i in range(1, l):
        if a[i] < a[i - 1]:
            lc += 1
            la += a[i - 1] - a[i]
        if a[i] > a[i - 1]:
            gc += 1
            ga += a[i] - a[i - 1]
    la /= lc
    ga /= gc
    rs = ga/la
    rsi = 100 - (100 / (1 + rs))
    return rsi

#THIS SHITS KINDA IFFY, IF ITS ACTING WIERD GRAB ME

def SMAn(a, n):                         #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    if (len(a) < n):
        n = len(a)
    n = int(np.floor(n))
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

def stochK(a, ll): #GETS STOCHK VALUE OF "LL" PERIODS FROM "A" ARRAY
    Ki = 0
    ll = int(np.floor(ll))
    if len(a) > ll:
        cpy = a[-ll:]
        h = max(cpy)
        l = min(cpy)
        if h - l > 0:
            Ki = (cpy[len(cpy) - 1] - l) / (h - l)
        else:
            Ki = (cpy[len(cpy) - 1] - l / .01)
    return Ki

def stoch_K(a, ll):
    K = np.zeros(len(a))
    ll = int(np.floor(ll))
    i = 1
    while i < len(a):
        if i > ll : #ll = STOCH K INPUT VAL
            cpy = a[i-ll:i + 1]
            h = max(cpy)
            l = min(cpy)
        else :
            cpy = a[0:i + 1]
            h = max(cpy)
            l = min(cpy)
        if h - l > 0:
            Ki = (a[i] - l) / (h - l)
        else:
            Ki = 0
        K[i] = Ki
        i += 1
    return K

def stochD(a, d, ll):
    d = int(np.floor(d))
    ll = int(np.floor(ll))
    K = stoch_K(a, ll)
    D = SMAn(K, d) # d = STOCH D INPUT VAL, ll = STOCH K INPUT VAL
    return D
def BBn(a, n, stddevD, stddevU): #GETS BOLLINGER BANDS OF "N" PERIODS AND STDDEV"UP" OR STDDEV"DOWN" LENGTHS
    cpy = a[-n:] #RETURNS IN FORMAT: LOWER BAND, MIDDLE BAND, LOWER BAND
    midb = SMAn(a, n) #CALLS SMAn
    std = np.std(cpy)
    ub = midb + (std * stddevU)
    lb = midb - (std * stddevD)
    return lb, midb, ub

def bbK(arr, Kin):
    close = arr[len(arr)-1]
    lb1, midb1, ub1 = BBn(arr, Kin, 2, 2)
    lb2, midb2, ub2 = BBn(arr, Kin, 3, 3)
    if (close > ub2):
        return 1
    elif (close > ub1):
        return 0.25
    elif (close < lb1):
        return 0.75
    elif (close < lb2):
        return 0
    else:
        return 0.5

def bbD(arr, Din):
    close = arr[len(arr) - 1]
    lb1, midb1, ub1 = BBn(arr, Din, 2, 2)
    lb2, midb2, ub2 = BBn(arr, Din, 3, 3)
    if (close > ub2):
        return 0
    elif (close > ub1):
        return 0.75
    elif (close < lb1):
        return 0.25
    elif (close < lb2):
        return 1
    else:
        return 0.5

#TIME WEIGHTED AVERAGE PRICE
def twap(arr, ll):
    a = arr[-ll:]
    high = max(a)
    low = min(a)
    close = a[len(a) - 1]
    return (high + low + close) / 3

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

def write_that_shit(log, tick, kin, din, perc, cuml, bitchCunt):
    # desc = sp.describe(perc)
    if os.path.isfile(log):
        th = 'a'
    else:
        th = 'w'
    file = open(log, th)
    file.write("Tick:\t")
    file.write(tick)
    file.write("\nK in:\t")
    file.write(str(int(np.floor(kin))))
    file.write("\nD in:\t")
    file.write(str(int(np.floor(din))))
    file.write("\nK1 in:\t")
    file.write("\nLen:\t")
    file.write(str(len(perc)))
    # file.write("\n\n\nPercent Diff:\n")
    # file.write(str(perc))
    # file.write("\n\nDescribed Diff:\n")
    # file.write(str(desc))
    file.write("\n\n twap,sma Cumulative Diff:\t")
    file.write(str(cuml))
    file.write("\nbitchCunt:\t")
    file.write(str(bitchCunt))
    file.close()
    # print("Described diff")
    # print(desc)
    # print("Cumulative Diff")
    # print("len:", len(perc))
    # print(cuml[j])


def fucking_paul(tik, log, Kin, Din, save_max, max_len, bitchCunt, tradeCost):
    stock = []
    with open(tik, 'r') as f:
        stock1 = f.readlines()
    f.close()
    for i, stocks in enumerate(stock1):
        stock.append(float(stocks))


    arr = []; buy = []; sell = [];  diff = []; perc = []; desc = []
    kar = []; dar = []; cumld = []; kar1 = []; dar1 = []; Kvl = np.zeros(2)
    Dvl = Kvl; s1ar = []; s2ar = []; shortDiff = []; cuml = 0.0
    stockBought = False
    stopLoss = False
    bull = 0; shit = 0; max = 0;

    for i, closeData in enumerate(stock):
        arr.append(closeData)
        if i >= int(Din) and i >= int(Kin):
                Kv = twap(arr, int(np.floor(Kin)))
                #kar.append(Kv)
                Dv = SMAn(arr, int(np.floor(Din)))
                #dar.append(Dv)
                if stockBought == True and closeData > max:
                    max = closeData
                if ((Kv > Dv) and (stockBought == False and stopLoss == False)):
                    buy.append(closeData * (1+tradeCost))
                    bull += 1
                    stockBought = True
                elif ((Kv < Dv) and stockBought == True):
                    sell.append(closeData * (1-tradeCost))
                    max = 0
                    shit += 1
                    stockBought = False
                elif (closeData < (max * (1-bitchCunt)) and stockBought == True):
                    sell.append(closeData * (1-tradeCost))
                    max = 0
                    shit += 1
                    stockBought = False
                    stopLoss = True
                elif ((Kv > Dv) and stopLoss == True):
                    stopLoss = False
    if stockBought == True:
        sell.append(stock[len(stock) - 1])
        shit += 1
    for i in range(bull):
        diff.append(sell[i] - buy[i])
        if i < bull - 1:
            shortDiff.append(sell[i] - buy[i + 1])
    for i in range(bull):
        perc.append(diff[i] / buy[i])
    for i in range(bull - 1):
        perc[i] += shortDiff[i] / sell[i]
    for i in range(bull):
        cuml += cuml * perc[i]

    print("len:", len(perc))

    if cuml > save_max and len(perc) <= max_len:
        write_that_shit(log, tik, Kin, Din, perc, cuml, bitchCunt)
# DONT FUCKING MOVE/INDENT WRITE_THAT_SHIT!!!!
    # plot(perc)
    # plot2(s1ar, s2ar)
    return cuml

def pillowcaseAssassination(fileTicker, k, i, fileOutput, save_max, max_len, bitchCunt, tradeCost):
    n_proc = 8
    verbOS = 10
    inc = 0
    Parallel(n_jobs=n_proc, verbose=verbOS)(delayed(fucking_paul)
            (fileTicker[inc], fileOutput[inc], k, i, save_max, max_len, bitchCunt, tradeCost)
            for inc, file in enumerate(fileTicker))


ticker = ["BTC_ETH", "BTC_XMR", "BTC_DASH", "BTC_XRP", "BTC_FCT", "BTC_MAID", "BTC_ZEC", "BTC_LTC"]
fileTicker = []
fileOutput = []
fileCuml = []
dataset = []
for i, tick in enumerate(ticker):
    fileTicker.append("../data/" + tick + ".txt")
    fileOutput.append("../output/" + tick + "_output.txt")
for i, file in enumerate(fileTicker):
    if (os.path.isfile(file) == False):
        fileWrite = open(file, 'w')
        dataset = CryptoQuote1(ticker[i]).close
        # tick = yahoo_finance.Share(ticker[i]).get_historical('2015-01-02', '2017-01-01')
        # dataset = np.zeros(len(tick))
        # i = len(tick) - 1
        # ik = 0
        # while i >= 0:
        #     dataset[ik] = tick[i]['Close']
        #     i -= 1
        #     ik += 1
        for i, close in enumerate(dataset):
            fileWrite.write(str(close))
            fileWrite.write('\n')

#fucking_paul(fileTicker, 10, 30, 15, 40, fileOutput, fileCuml, save_max=1.02, save_min=0.98, max_len=100000, bitchCunt=0.05, tradeCost=0.00)


def run():
    k1 = 2
    k2 = 3000
    l1 = 2
    l2 = 3000
    j1 = 0.000
    j2 = 0.15
    k = k1
    i = l1
    j = j1
    returns = []
    while (k < k2):
        while (i < l2):
            while (j < j2):
                if i > 0:
                    if (int(np.floor(i)) % 2 == 0):
                        print(int(np.floor(i)), "/", l2, int(np.floor(k)), "/", k2)
                    pillowcaseAssassination(fileTicker, k, i, fileOutput, save_max=1.01, max_len=3000000, bitchCunt=j, tradeCost=0.0005)
                if (j < 0.01):
                    j += 0.0035
                else:
                    j *= 1.3
            j = j1
            if (i < 10):
                i += 1
            else:
                i *= 1.2
        i = l1
        if (k < 10):
            k += 1
        elif (k < 1000):
            k *= 1.2
        elif (k < 10000):
            k *= 1.05
        else:
            k *= 1.01

run()
