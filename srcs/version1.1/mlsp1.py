import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yahoo_finance
import numpy as np
import urllib.request
import urllib, time, datetime
import scipy.stats as sp
from matplotlib import pyplot as plt
import os.path

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

def write_that_shit(log, tick, kin, perc, cuml, bitchCunt):
    # desc = sp.describe(perc)
    file = open(log, 'w')
    file.write("Tick:\t")
    file.write(tick)
    file.write("\nK in:\t")
    file.write(str(int(np.floor(kin))))
    # file.write("\nD in:\t")
    # file.write(str(int(np.floor(din))))
    file.write("\nLen:\t")
    file.write(str(len(perc)))
    # file.write("\n\n\nPercent Diff:\n")
    # file.write(str(perc))
    # file.write("\n\nDescribed Diff:\n")
    # file.write(str(desc))
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

def fucking_paul(tick, Nin, log, fcuml, save_min, save_max, max_len, bitchCunt, tradeCost):
    cuml = []
    for j, tik in enumerate(tick):
        stock = []
        with open(tik, 'r') as f:
            stock1 = f.readlines()
        f.close()
        for i, stocks in enumerate(stock1):
            stock.append(float(stocks))

        arr = []; buy = []; sell = [];  diff = []; perc = []; desc = []
        kar = []; dar = []; cumld = []; shortDiff = []
        stockBought = False; stopLoss = False
        bull = 0; shit = 0; max = 0;
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler1 = MinMaxScaler(feature_range=(0,1))
        cuml.append(1)
        for i, closeData in enumerate(stock):
            arr.append(closeData)
            if i > len(stock) - Nin * 1:
                #print("\n\ninput array:", arr)
                arry = scaler.fit_transform(arr[-Nin:])
                dataset = scaler1.fit_transform(arr)
                train_size = int(len(dataset))
                # reshape into X=t and Y=t+1
                trainX, trainY = create_dataset(dataset, Nin)
                # reshape input to be [samples, time steps, features]
                trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                dataset = np.reshape(dataset, (1, 1, dataset.shape[0]))
                arry = np.reshape(arry, (1, 1, arry.shape[0]))
                # create and fit the LSTM network
                model = Sequential()
                model.add(LSTM(4, input_dim=Nin))
                model.add(Dense(1))
                model.compile(loss='mean_absolute_error', optimizer='Adam')
                model.fit(trainX, trainY, nb_epoch=42, batch_size=10, verbose=0)
                # make predictions
                trainPredict = model.predict(trainX)
                predict = model.predict(arry)
                # invert predictions
                arry = np.reshape(arry, (1, Nin))
                trainPredict = scaler1.inverse_transform(trainPredict)
                trainY = scaler1.inverse_transform([trainY])
                predict = scaler.inverse_transform(predict)
                predict = predict[0][0]
                arry = scaler.inverse_transform(arry)
                #kar.append(predict)
                print("arry", arry[0][Nin - 1])
                #if i > 100:
                #plot(trainPredict)
                print("predict:", predict)
                #print("predicted:", kar)
                # calculate root mean squared error
                if stockBought == True and closeData > max:
                    max = closeData
                if ((predict > closeData) and (stockBought == False and stopLoss == False)):
                    buy.append(closeData * (1 - tradeCost))
                    bull += 1
                    stockBought = True
                elif ((predict < closeData) and stockBought == True):
                    sell.append(closeData * (1 + tradeCost))
                    max = 0
                    shit += 1
                    stockBought = False
                elif (closeData < (max * (1 - bitchCunt)) and stockBought == True):
                    sell.append(closeData * (1 + tradeCost))
                    max = 0
                    shit += 1
                    stockBought = False
                    stopLoss = True
                elif ((predict < closeData) and stopLoss == True):
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
            cuml[j] = cuml[j] + (cuml[j] * perc[i])
            cumld.append(cuml[j])

    for i, cum in enumerate(cuml):
        if (cum > 0 or cum < 0):
            if (os.path.isfile(fcuml[i]) == False):
                with open(log[i]) as f:
                    with open(fcuml[i], "w") as f1:
                        for line in f:
                            #if "ROW" in line:
                            f1.write(line)
                f.close()
                f1.close()
            else:
                with open(log[i]) as f:
                    with open(fcuml[i], "a") as f1:
                        f1.write("\n\n")
                        for line in f:
                            #if "ROW" in line:
                            f1.write(line)
                f.close()
                f1.close()

ticker = ["FX"]
fileTicker = ["../../FXstatic_data/GBPJPY.2016.csv", "../../FXstatic_data/EURNZD.2016.csv"]
fileOutput = []
fileCuml = []
dataset = []
for i, tick in enumerate(ticker):
    #fileTicker.append("../../data/" + tick + ".txt")
    fileOutput.append("../../output/" + tick + "_output.txt")
    fileCuml.append("../../cuml/" + tick + "_cuml.txt")
# for i, file in enumerate(fileTicker):
#     if (os.path.isfile(file) == False):
#         fileWrite = open(file, 'w')
#         dataset = GoogleIntradayQuote(ticker[i]).close
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

fucking_paul(fileTicker, 30, fileOutput, fileCuml, save_max=1.00, save_min=0.999, max_len=100000, bitchCunt=0.10, tradeCost=0.00001)
# k1 = 1
# k2 = 3000
# l1 = 2
# l2 = 3600
# j1 = 0.000
# j2 = 0.100
# k = k1
# i = l1
# j = j1
# returns = []
# if __name__ == '__main__':
#     while (k < k2):
#         while (i < l2):
#             while (j < j2):
#                 if i > k:
#                     if (int(np.floor(i)) % 2 == 0):
#                         print(int(np.floor(i)), "/", l2, int(np.floor(k)), "/", k2)
#                     returns.append(fucking_paul(fileTicker, k, i, fileOutput, fileCuml, save_max=1.02, save_min=0.98, max_len=100000, bitchCunt=j))
#                 if j < 0.01:
#                     j += 0.001
#                 else:
#                     j *= 1.1
#             j = j1
#             if (i < 10):
#                 i += 1
#             else:
#                 i *= 1.1
#             if (i < 10):
#                 i += 1
#             else:
#                 i *= 1.1
#         i = l1
#         if (k < 10):
#             k += 1
#         else:
#             k *= 1.1