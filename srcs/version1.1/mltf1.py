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

def write_that_shit(log, tick, Nin, numEpoch, numBatch, opt, err, diff):
    file = open(log, 'w')
    file.write("Tick:\t")
    file.write(tick)
    file.write("\nN in:\t")
    file.write(str(int(np.floor(Nin))))
    file.write("\nnumBatch:\t")
    file.write(str(int(np.floor(numBatch))))
    file.write("\nnumEpoch:\t")
    file.write(str(numEpoch))
    file.write("\nerrorCalc:")
    file.write(str(err))
    file.write("\nopt:\t")
    file.write(str(opt))
    file.write("\nerrors:\t")
    for i in range(len(diff) - 1):
        file.write(str(diff[i]))
        file.write(" ")
    file.write("\nmean error:\t")
    file.write(str(np.mean(diff)))
    file.write("\nerror variance:\t")
    file.write(str(np.var(diff)))
    file.write("\nerror kurtosis:\t")
    file.write(str(scipy.stats.kurtosis(diff, fisher=True)))
    file.close()

def fucking_peter(tick, Nin, err, opt, log, fcuml, numEpoch, numBatch):
    cuml = []
    for j, tik in enumerate(tick):
        stock = []
        with open(tik, 'r') as f:
            stock1 = f.readlines()
        f.close()
        for i, stocks in enumerate(stock1):
            stock.append(float(stocks))
        arr = []; diff = [];
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler1 = MinMaxScaler(feature_range=(0,1))
        cuml.append(1)
        for i, closeData in enumerate(stock):
            arr.append(closeData)
            if i > (len(stock) - 10):
                #print("\n\ninput array:", arr)
                arry = scaler.fit_transform(arr[-Nin:])
                dataset = scaler1.fit_transform(arr)
                # reshape into X=t and Y=t+1
                trainX, trainY = create_dataset(dataset, Nin)
                # reshape input to be [samples, time steps, features]
                trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                arry = np.reshape(arry, (1, 1, arry.shape[0]))
                # create and fit the LSTM network
                model = Sequential()
                model.add(LSTM(4, input_dim=Nin))
                model.add(Dense(1))
                model.compile(loss= err, optimizer=opt)
                model.fit(trainX, trainY, nb_epoch=numEpoch, batch_size=numBatch, verbose=0)
                # make predictions
                #trainPredict = model.predict(trainX)
                predict = model.predict(arry)
                # invert predictions
                arry = np.reshape(arry, (1, Nin))
                #trainPredict = scaler1.inverse_transform(trainPredict)
                trainY = scaler1.inverse_transform([trainY])
                predict = scaler.inverse_transform(predict)
                predict = predict[0][0]
                arry = scaler.inverse_transform(arry)
                error = predict - arry[0][Nin - 1]
                if error < 0:
                    error *= -1
                diff.append(error)
                #kar.append(predict)
                #print("arry", arry[0][Nin-1])
                #if i > 100:
                #plot(trainPredict)
                #print("predict:", predict)

        # print("errors:", diff)
        # print("mean error:", np.mean(diff))
        # print("error kurtosis", scipy.stats.kurtosis(diff, fisher=True))
        # print("error variance", np.var(diff))

        write_that_shit(log[j], tik, Nin, numEpoch, numBatch, opt, err, diff)

    for i, cum in enumerate(fcuml):
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

    return cuml

ticker = ["MNKD", "RICE", "FNBC", "RTRX"]
fileTicker = []
fileOutput = []
fileCuml = []
dataset = []
for i, tick in enumerate(ticker):
    fileTicker.append("../../data/" + tick + ".txt")
    fileOutput.append("../../output/" + tick + "_output.txt")
    fileCuml.append("../../cuml/" + tick + "_cuml.txt")
for i, file in enumerate(fileTicker):
    if (os.path.isfile(file) == False):
        fileWrite = open(file, 'w')
        #dataset = GoogleIntradayQuote(ticker[i]).close
        tick = yahoo_finance.Share(ticker[i]).get_historical('2015-01-02', '2017-01-01')
        dataset = np.zeros(len(tick))
        i = len(tick) - 1
        while i >= 0:
            ik = 0
            dataset[ik] = tick[i]['Close']
            i -= 1
            ik += 1
        for i, close in enumerate(dataset):
            fileWrite.write(str(close))
            fileWrite.write('\n')

opts = ['sgd', 'Adam', 'Adadelta']
errs = ['mean_absolute_error', 'mean_squared_error']
nins = [10, 20, 30, 60]
batchs = [10, 30, 90, 150, 270, 1000]
epochs = [10, 30, 90, 150, 270, 1000]

for i in range(len(errs)):
    for j in range(len(batchs)):
        for k in range(len(epochs)):
            for l in range(len(opts)):
                for m in range(len(nins)):
                    fucking_peter(fileTicker, nins[m], errs[i], opts[l], fileOutput, fileCuml, epochs[k], batchs[j])

#fucking_peter(fileTicker, 75, 'mean_absolute_error', 'sgd', fileOutput, fileCuml, 30, 10)
