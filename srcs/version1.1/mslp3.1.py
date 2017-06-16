import matplotlib.pyplot as plt
import pandas as pd
import math
# from keras.models import Sequential
# from keras.layers import LSTM, Dropout, Dense
# from keras.metrics import binary_accuracy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import yahoo_finance
import numpy as np
import urllib
import urllib, time, datetime
import scipy.stats as sp
#from matplotlib import pyplot as plt
import os.path
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
    lc, gc, la, ga = 0.01
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

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


class Quote(object):
    DATE_FMT = '%Y-%m-%d'
    TIME_FMT = '%H:%M:%S'

    def __init__(self):
        self.symbol = ''
        self.date, self.time, self.open_, self.high, self.low, self.close, self.volume = ([] for _ in range(7))

    def append(self, dt, open_, high, low, close, volume):
        self.date.append(dt.date())
        self.time.append(dt.time())
        self.open_.append(float(open_))
        self.high.append(float(high))
        self.low.append(float(low))
        self.close.append(float(close))
        self.volume.append(int(volume))

    def to_csv(self):
        return ''.join(["{0},{1},{2},{3:.2f},{4:.2f},{5:.2f},{6:.2f},{7}\n".format(self.symbol,
                                                                                   self.date[bar].strftime('%Y-%m-%d'),
                                                                                   self.time[bar].strftime('%H:%M:%S'),
                                                                                   self.open_[bar], self.high[bar],
                                                                                   self.low[bar], self.close[bar],
                                                                                   self.volume[bar])
                        for bar in range(len(self.close))])

    def write_csv(self, filename):
        with open(filename, 'w') as f:
            f.write(self.to_csv())

    def read_csv(self, filename):
        self.symbol = ''
        self.date, self.time, self.open_, self.high, self.low, self.close, self.volume = ([] for _ in range(7))
        for line in open(filename, 'r'):
            symbol, ds, ts, open_, high, low, close, volume = line.rstrip().split(',')
            self.symbol = symbol
            dt = datetime.datetime.strptime(ds + ' ' + ts, self.DATE_FMT + ' ' + self.TIME_FMT)
            self.append(dt, open_, high, low, close, volume)
        return True

    def __repr__(self):
        return self.to_csv()


class GoogleIntradayQuote(Quote):
    ''' Intraday quotes from Google. Specify interval seconds and number of days '''

    def __init__(self, symbol, interval_seconds=600, num_days=1):
        super(GoogleIntradayQuote, self).__init__()
        self.symbol = symbol.upper()
        url_string = "http://www.google.com/finance/getprices?q={0}".format(self.symbol)
        url_string += "&i={0}&p={1}d&f=d,o,h,l,c,v".format(interval_seconds, num_days)
        thing = urllib.request.urlopen(url_string)
        csv = thing.read().decode('utf-8').split('\n')
        for bar in range(7, len(csv)):
            if csv[bar].count(',') != 5: continue
            offset, close, high, low, open_, volume = csv[bar].split(',')
            if offset[0] == 'a':
                day = float(offset[1:])
                offset = 0
            else:
                offset = float(offset)
            open_, high, low, close = [float(x) for x in [open_, high, low, close]]
            dt = datetime.datetime.fromtimestamp(day + (interval_seconds * offset))
            self.append(dt, open_, high, low, close, volume)

def createBinaryTrainingSet(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:i+look_back]
        X.append(a)
        if dataset[i + look_back + 1] > dataset[i + look_back]:
            Y.append(1)
        else:
            Y.append(0)
    return np.array(X), np.array(Y)

def create_orderbook_training_set(buy_arr, sell_arr, lookback):
    lookback *= 100
    x, y = [], []
    k = 0
    while k < (len(buy_arr) - lookback):
        x.append(sell_arr[k:k + lookback] + buy_arr[k:k + lookback])
        y.append(np.mean([float(sell_arr[k + lookback]), float(buy_arr[k + lookback])]))
        k += 2
    return np.array(x), np.array(y)

def create_orderbook_magnitude_training_set(buy_arr, sell_arr, lookback):
    lookback *= 10
    x, y = [], []
    k = 0
    while k < (len(buy_arr) - lookback):
        x.append(sell_arr[k:k + lookback] + buy_arr[k:k + lookback])
        y.append((np.mean([float(sell_arr[k + lookback]), float(buy_arr[k + lookback])]) -
                 np.mean([float(sell_arr[k + lookback - 2]), float(buy_arr[k + lookback - 2])])) /
                 np.mean([float(sell_arr[k + lookback - 2]), float(buy_arr[k + lookback - 2])]))
        k += 2
    return np.array(x), np.array(y)

def create_binary_orderbook_training_set(buy_arr, sell_arr, lookback):
    lookback *= 10
    x, y = [], []
    k = 2
    while k < (len(buy_arr) - lookback - 2):
        x.append(sell_arr[k:k + lookback] + buy_arr[k:k + lookback])
        if np.mean([float(sell_arr[k + lookback]),
                    float(buy_arr[k + lookback])]) > np.mean([float(sell_arr[(k + lookback) + 2]),
                                                              float(buy_arr[(k + lookback) + 2])]):
            y.append(0)
        else:
            y.append(1)

        k += 2
    return np.array(x), np.array(y)

def books2arrays(buy_tick, sell_tick):
    buy_arr, sell_arr = [], []
    with open(buy_tick, 'r', errors='replace') as bf:
        with open(sell_tick, 'r', errors='replace') as sf:
            buy_file = bf.readlines()
            sell_file = sf.readlines()
            if len(buy_file) != len(sell_file): print(buy_tick, "SCRAPER DATA LENGTH DISCREPANCY!!!!")
            for i in range(min([len(buy_file), len(sell_file)])):
                bObj = buy_file[i].split("\t")
                sObj = sell_file[i].split("\t")
                if len(bObj) + len(sObj) == 4 and len(bObj[0]) + len(bObj[1]) + len(sObj[0]) + len(sObj[1]) < 50:
                    bp, bv = bObj[0], bObj[1]
                    sp, sv = sObj[0], sObj[1]
                    buy_arr.append(float(bp))
                    buy_arr.append(float(bv))
                    sell_arr.append(float(sp))
                    sell_arr.append(float(sv))
                else:
                    break
    bf.close()
    sf.close()
    return buy_arr, sell_arr


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
    the_url = "https://poloniex.com/public?command=returnChartData&currencyPair={0}&start=1435699200&end=9999999999&period=300".format(the_symbol)
    response = urllib.request.urlopen(the_url).read().decode("utf-8").split(",")
    print(response[1:10])
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


def write_that_shit(log, tick, kin, a, perc, cuml, bitchCunt):
    # desc = sp.describe(perc)
    if os.path.isfile(log):
        th = 'a'
    else:
        th = 'w'
    file = open(log, th)
    file.write("Tick:\t")
    file.write(tick)
    file.write("\n Nin:\t")
    file.write(str(int(np.floor(kin))))
    file.write("\nAlpha:\t")
    file.write(str(a))
    file.write("\nLen:\t")
    file.write(str(len(perc)))
    # file.write("\n\n\nPercent Diff:\n")
    # file.write(str(perc))
    if len(perc) > 10:
        desc = sp.describe(perc)
        file.write("\n\nDescribed Diff:\n")
        file.write(str(desc))
    file.write("\n\nCumulative Diff:\t")
    file.write(str(cuml))
    file.write("\nbitchCunt:\t")
    file.write(str(bitchCunt))
    file.close()
    # print("Described diff")
    # print(desc)
    # print("Cumulative Diff")
    # print("len:", len(perc))
    # print(cuml[j])

def fucking_paul(tick, Nin, a, log, save_max, max_len, bitchCunt, tradeCost):
    cuml = []
    jj = 0
    while jj < len(tick) - 1:
        with open(tick[jj + 2], 'r') as f:
            stock1 = f.readlines()
        f.close()
        stock = []
        for i, stocks in enumerate(stock1):
            if len(stocks) < 15:
                stock.append(float(stocks))
        arr = []; buy = []; sell = [];  diff = []; perc = []; desc = []
        kar = []; dar = []; cumld = []; shortDiff = []; errorArr = [];
        stockBought = False; stopLoss = False
        bull = 0; shit = 0; maxP = 0;
        scaler = MinMaxScaler(feature_range=(-1, 1))
        cuml.append(1)
        R = Ridge(alpha=a, fit_intercept=True, normalize=True)
        buys, sells = books2arrays(tick[jj], tick[jj + 1])
        trainX, trainY = create_orderbook_training_set(buys[:int(np.floor(len(buys) * 0.8))],
                                                              sells[:int(np.floor(len(sells) * 0.8))], Nin)
        #trainX = scaler.fit_transform(trainX)
        R.fit(trainX, trainY)
        #print(len(buys), len(stock))
        for i, closeData in enumerate(stock):
            arr.append(closeData)
            if i > int(np.floor(len(stock) * .8) + Nin) and i * 100 < len(buys):
                #print("\n\ninput array:", arr)
                new_i = i * 100
                arry = sells[new_i - Nin * 100:new_i] + buys[new_i - Nin * 100:new_i]
                #arry = scaler.fit_transform(arry)
                #arry = arr[-Nin:]
                # arry = np.reshape(arry, (1, 1, arry.shape[0]))
                predict = R.predict(arry)
                if i < len(stock) - 1:
                    errorArr.append(abs(predict - stock[i + 1]) / closeData)
                # invert predictions
                # arry = scaler.inverse_transform(arry)
                # arry = np.reshape(arry, (1, Nin))
                #predict = scaler.inverse_transform(predict)
                #predict = predict[0][0]
                #kar.append(predict)
                #if i > 100:
                #plot(trainPredict)
                #print("predict:", predict, "Y[t]:", closeData)
                #print("predicted:", kar)
                # calculate root mean squared error
                if ((float(predict) > closeData) and (stockBought == False and stopLoss == False)):
                    buy.append(closeData * (1 + tradeCost))
                    bull += 1
                    stockBought = True
                if stockBought == True and closeData > maxP:
                    maxP = closeData
                elif ((float(predict) < closeData) and stockBought == True):
                    sell.append(closeData * (1 - tradeCost))
                    maxP = 0
                    shit += 1
                    stockBought = False
                elif (closeData < (maxP * (1 - bitchCunt)) and stockBought == True):
                    sell.append(closeData * (1 - tradeCost))
                    maxP = 0
                    shit += 1
                    stockBought = False
                    stopLoss = True
                elif ((float(predict) < closeData) and stopLoss == True):
                    stopLoss = False
        if stockBought == True:
            sell.append(stock[len(stock)-1])
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
            cuml[int(np.floor(jj / 3))] += cuml[int(np.floor(jj / 3))] * perc[i]
            cumld.append(cuml[int(np.floor(jj / 3))])

        print("len:", len(perc), "cuml:", cuml[int(np.floor(jj / 3))], "alpha:", a, "mean % error:", np.mean(errorArr))
        # plot(perc)
        # plot(cumld)
        if cuml[int(np.floor(jj / 3))] > save_max and len(perc) <= max_len:
            write_that_shit(log[int(np.floor(jj / 3))], tick[jj], Nin, a, perc, cuml[int(np.floor(jj / 3))], bitchCunt)
        jj += 3
        stock = []


ticker = ["BTC-XMR", "BTC-DASH", "BTC-XEM", "BTC-MAID", "BTC-LTC", "BTC-XRP", "BTC-ETH"]
fileTicker = []
fileOutput = []
fileCuml = []
dataset = []
for i, tick in enumerate(ticker):
    fileTicker.append("../../../../../Desktop/comp/HD_60x100_outputs/books/" + tick + "_buy_books.txt")
    fileTicker.append("../../../../../Desktop/comp/HD_60x100_outputs/books/" + tick + "_sell_books.txt")
    fileTicker.append("../../../../../Desktop/comp/HD_60x100_outputs/prices/" + tick + "_prices.txt")
    fileOutput.append("../../output/" + tick + "_mslp3.1_6.14.17_unscaled_x0.8_1intervalPred_output.txt")

for i, file in enumerate(fileTicker):
    if (os.path.isfile(file) == False):
        print("missing:", file)

# for i, file in enumerate(fileTicker):
#     if (os.path.isfile(file) == False):
#         fileWrite = open(file, 'w')
#         dataset = CryptoQuote1(ticker[i]).close
#         # tick = yahoo_finance.Share(ticker[i]).get_historical('2015-01-02', '2017-01-01')
#         # dataset = np.zeros(len(tick))
#         # i = len(tick) - 1
#         # while i >= 0:
#         #     dataset[ik] = tick[i]['Close']
#         #     i -= 1
#         #     ik += 1
#         for i, close in enumerate(dataset):
#             fileWrite.write(str(close))
#             fileWrite.write('\n')

j = 1
a = 0.01
k = 0
while k < 0.3:
    while a < 0.99:
        while j < 3:
            fucking_paul(fileTicker, int(np.floor(j)), a, fileOutput, save_max=1.00, max_len=100000, bitchCunt=0.50, tradeCost=0.0025)
            print("j:", j)
            j += 1
        j = 1
        a *= 1.2
    a = 0.01
    k += 0.01
#fucking_paul(fileTicker, 1, 0.2, fileOutput, save_max=1.00, max_len=100000, bitchCunt=0.50, tradeCost=0.0025)