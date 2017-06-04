import matplotlib.pyplot as plt
import pandas as pd
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yahoo_finance
import numpy as np
import urllib.request
import urllib, time, datetime
import scipy.stats as sp
from matplotlib import pyplot as plt
import os.path
import scipy
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

    def __init__(self, symbol, interval_seconds=1200, num_days=10):
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

def createBinaryTrainingSet(dataset, look_back):
    X, Y = [], []
    i = 3
    while i < len(dataset)-look_back-5:
        a = dataset[i-3:i-3+look_back]
        X.append(a)
        if dataset[i + look_back + 5] > dataset[i + look_back]:
            Y.append(1)
        else:
            Y.append(0)
        i +=5
    return np.array(X), np.array(Y)


def create_orderbook_training_set(buy_arr, sell_arr, lookback):
    lookback *= 10
    x, y = [], []
    k = 0
    while k < (len(buy_arr) - lookback - 10):
        x.append(sell_arr[k:k + lookback] + buy_arr[k:k + lookback])
        y.append(np.mean([float(sell_arr[k + lookback + 10]), float(buy_arr[k + lookback + 10])]))
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

def books2arrays(buy_tick, sell_tick):
    buy_arr, sell_arr = [], []
    with open(buy_tick, 'r') as bf:
        with open(sell_tick, 'r') as sf:
            buy_file = bf.readlines()
            sell_file = sf.readlines()
            if len(buy_file) != len(sell_file): print(buy_tick, "SCRAPER DATA LENGTH DISCREPANCY!!!!")
            for i in range(min([len(buy_file), len(sell_file)])):
                bObj = buy_file[i].split("\t")
                sObj = sell_file[i].split("\t")
                bp, bv = bObj[0], bObj[1]
                sp, sv = sObj[0], sObj[1]
                buy_arr.append(float(bp))
                buy_arr.append(float(bv))
                sell_arr.append(float(sp))
                sell_arr.append(float(sv))
    bf.close()
    sf.close()
    return buy_arr, sell_arr

def write_that_shit(log, tick, d, Nin, predicts, numEpoch, numBatch, opt, diff, err, errorMetric):
    if os.path.isfile(log):
        th = 'a'
    else:
        th = 'w'
    file = open(log, th)
    file.write("Tick:\t")
    file.write(tick)
    file.write("\nN in:\t")
    file.write(str(int(np.floor(Nin))))
    file.write("\nnumBatch:\t")
    file.write(str(int(np.floor(numBatch))))
    file.write("\nnumEpoch:\t")
    file.write(str(numEpoch))
    file.write("\navg % error:")
    file.write(str(err))
    file.write("\nmeanPredict:")
    file.write(str(np.mean(predicts)))
    if len(predicts) > 10:
        desc = sp.describe(predicts)
        file.write("\nDescribed Diff:\n")
        file.write(str(desc))
    file.write("\ndropout:\t")
    file.write(str(d))
    file.write("\nopt:\t")
    file.write(str(opt))
    file.write("\nerrorMetric:\t")
    file.write(str(errorMetric))
    file.write("\nmean directional inverse error:\t")
    file.write(str(np.mean(diff)))
    file.write("\n\n")
    # file.write("\nerror variance:\t")
    # file.write(str(np.var(diff)))
    # file.write("\nerror kurtosis:\t")
    # file.write(str(scipy.stats.kurtosis(diff, fisher=True)))
    file.close()

def fucking_peter(tick, Nin, drop, err, opt, log, numEpoch, numBatch):
    cuml, stock = [], []
    jj = 0
    while jj < len(tick) - 1:
        with open(tick[jj + 2], 'r') as f:
            stock1 = f.readlines()
        f.close()
        for i, stocks in enumerate(stock1):
            stock.append(float(stocks))
        arr = []; predictArray = []; errorCnt = 0; correct_margin_array = []; false_margin_array = [];
        buy = [];sell = []; diff = []; perc = []; desc = []; kar = []; dar = []; cumld = []; errors = [];
        shortDiff = []; stockBought = False; stopLoss = False; bull = 0; shit = 0; maxP = 0;
        scaler = MinMaxScaler(feature_range=(-1, 1))
        cuml.append(1)
        buys, sells = books2arrays(tick[jj], tick[jj + 1])
        trainX, trainY = create_orderbook_training_set(buys[:int(np.floor(len(buys) * 0.8))],
                                                       sells[:int(np.floor(len(sells) * 0.8))], Nin)
        # print("training x[-1], y[-1]", trainX[-1], trainY[-1])
        # dataset = scaler1.fit_transform(stock[:len(stock) - Nin * .9])
        # dataset = stock[:int(np.floor(len(stock) * .95))]
        trainX = scaler.fit_transform(trainX)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

        model = Sequential()
        model.add(Dense(Nin * 20, input_shape=(1, Nin * 20), activation='tanh'))
        model.add(Dropout(drop))
        # model.add(Dense(Nin, activation='relu'))
        model.add(LSTM(Nin * 20, activation='tanh'))
        model.add(Dense(Nin, activation='tanh'))
        model.add(Dense(1))
        model.compile(loss=err, optimizer=opt, metrics=['accuracy'])
        model.fit(trainX, trainY, nb_epoch=numEpoch, batch_size=numBatch, verbose=0)

        for i, closeData in enumerate(stock):
            arr.append(closeData)
            if i > int(np.floor(len(stock) * .8)):
                # print("\n\ninput array:", arr)
                new_i = i * 10
                arry = sells[new_i - Nin * 10:new_i] + buys[new_i - Nin * 10:new_i]
                arry = scaler.fit_transform(arry)
                # arry = arr[-Nin:]
                arry = np.reshape(arry, (1, 1, len(arry)))
                # arry = np.reshape(arry, (1, 1, arry.shape[0]))
                predict = model.predict(arry)
                # invert predictions
                arry = np.reshape(arry, (1, len(arry[0][0])))
                arry = scaler.inverse_transform(arry)
                #predict = scaler.inverse_transform(predict)
                predict = predict[0][0]
                predictArray.append(predict)
                # kar.append(predict)
                if i < len(stock) - 10:
                    difference = abs(predict - stock[i + 10]) / closeData
                    errors.append(difference)
                    #print("arry_diff:", difference)
                    # print("predict:", predict)
                    if (stock[i + 10] > arry[0][-1] and (predict > closeData)) or \
                            (stock[i + 10] < arry[0][-1] and predict < closeData):
                        diff.append(1)
                        #print("correct, margin:", predict - .5)
                        #correct_margin_array.append(predict - .5)
                    else:
                        diff.append(0)
                        #print("incorrect, margin:", predict - .5)
                        #false_margin_array.append(predict - .5)
                # if i > 100:
                # plot(trainPredict)

                # print("predicted:", kar)
                # calculate root mean squared error

        # print("errors:", diff)
        print("tik:", tick[jj], "\n",
              "Nin:", Nin, "\n",
              "numEpoch:", numEpoch, "\n",
              "numBatch:", numBatch, "\n",
              "opt:", opt, "\n",
              "err:", err, "\n",
              "dropout:", drop)
        #if errorCnt > 0: print("tengo", errorCnt, "problemas)")
        print("avg prediction:", np.mean(predictArray))
        print("mean directional inverse error (% correct):", np.mean(diff))
        print("mean % error:", np.mean(errors))
        #print("mean correct margin:", np.mean(correct_margin_array))
        #print("mean error margin:", np.mean(false_margin_array))
        print("described:", sp.describe(predictArray))
        print("\n\n")
        # print("error kurtosis", scipy.stats.kurtosis(diff, fisher=True))
        # print("error variance", np.var(diff))

        if np.mean(errors) < 0.05:
            write_that_shit(log[int(np.floor(jj / 3))], tick[jj], drop, Nin, predictArray,
                            numEpoch, numBatch, opt, diff, np.mean(errors), err)
        jj += 3


    return cuml

ticker = ["BTC-XMR", "BTC-DASH", "BTC-MAID", "BTC-LTC", "BTC-XRP", "BTC-ETH"]
#ticker = ["BTC-XMR"]
fileTicker = []
fileOutput = []
fileCuml = []
dataset = []
for i, tick in enumerate(ticker):
    fileTicker.append("../../../../../Desktop/comp/HD_6x100_outputs/books/" + tick + "_buy_books.txt")
    fileTicker.append("../../../../../Desktop/comp/HD_6x100_outputs/books/" + tick + "_sell_books.txt")
    fileTicker.append("../../../../../Desktop/comp/HD_6x100_outputs/prices/" + tick + "_prices.txt")
    fileOutput.append("../../output/" + tick + "_mltf1_tanhEdition_5.29.17_x0.8_1intervalPred_output.txt")

for i, file in enumerate(fileTicker):
    if (os.path.isfile(file) == False):
        print("missing:", file)

#opts = ['Adadelta', 'RMSprop', 'Adagrad', 'Nadam']
opts = ['Adamax', 'adam']
errs = ['mean_absolute_error']
#errs = ['binary_crossentropy']
#nins = np.arange(1, 15, step=3)
nins = [1, 5]
#nins = [10]
batchs = [5]
#batchs = [10, 100]
#epoch_scalars = np.arange(5, 16, step=5)
epoch_scalars = [5]
#drops = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
drops = [0.2]

#nins = [300]; batchs = [50]; epochs = [100];

for i in range(len(errs)):
    for l in range(len(opts)):
        for j in range(len(batchs)):
            for d in range(len(drops)):
                for m in range(len(nins)):
                    for k in range(len(epoch_scalars)):
                        print(nins[m], drops[d], errs[i], opts[l], nins[m] * epoch_scalars[k], batchs[j])
                        fucking_peter(fileTicker, nins[m] * 10, drops[d], errs[i], opts[l], fileOutput, nins[m] * epoch_scalars[k], batchs[j])
                        # try:
                        #     fucking_peter(fileTicker, nins[m], errs[i], opts[l], fileOutput, epochs[k], batchs[j])
                        # except:
                        #     print("NOT GOOD")
#fucking_peter(fileTicker, 75, 'mean_absolute_error', 'sgd', fileOutput, fileCuml, 30, 10)
