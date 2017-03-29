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
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:i+look_back]
        X.append(a)
        if dataset[i + look_back + 1] > dataset[i + look_back]:
            Y.append(1)
        else:
            Y.append(1)
    return np.array(X), np.array(Y)

def write_that_shit(log, tick, Nin, numEpoch, numBatch, opt, err, diff):
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
    file.write("\nerrorCalc:")
    file.write(str(err))
    file.write("\nopt:\t")
    file.write(str(opt))
    file.write("\nerrors:\t")
    for i in range(len(diff) - 1):
        file.write(str(diff[i]))
        file.write(" ")
    file.write("\nmean inverse error:\t")
    file.write(str(np.mean(diff)))
    # file.write("\nerror variance:\t")
    # file.write(str(np.var(diff)))
    # file.write("\nerror kurtosis:\t")
    # file.write(str(scipy.stats.kurtosis(diff, fisher=True)))
    file.close()

def fucking_peter(tick, Nin, err, opt, log, numEpoch, numBatch):
    Nin *= 5
    cuml = []
    for j, tik in enumerate(tick):
        stock = []
        with open(tik, 'r') as f:
            stock1 = f.readlines()
        f.close()
        #for i, stocks in enumerate(stock1):
        for i, stocks in enumerate(stock1[int(np.floor(len(stock1) * .97)):]):
            stock.append(float(stocks))
        arr = []; diff = []; false_margin_array = []; correct_margin_array = [];
        errorCnt = 0; predictArray = [];
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler1 = MinMaxScaler(feature_range=(-1,1))
        cuml.append(1)

        dataset = scaler1.fit_transform(stock[:int(np.floor(len(stock) * .95))])
        #dataset = stock[:int(np.floor(len(stock) * .95))]
        trainX, trainY = createBinaryTrainingSet(dataset, Nin)
        #print("len X:", len(trainX), "len Y:", len(trainY))
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        #trainY = np.reshape(trainY, (1, 1, trainY.shape[0]))
        prelu = keras.layers.advanced_activations.PReLU(init='zero', weights=None)

        model = Sequential()
        model.add(Dense(Nin, input_shape=(1, Nin)))
        model.add(prelu)
        model.add(LSTM(Nin, activation='relu'))
        #model.add(Dropout(0.3, input_shape=(1, Nin)))
        model.add(Dense(Nin))
        model.add(Dense(1))
        model.add(prelu)
        model.compile(loss=err, optimizer=opt, metrics=['binary_accuracy'])
        model.fit(trainX, trainY, nb_epoch=numBatch, batch_size=numBatch, verbose=0)

        for i, closeData in enumerate(stock):
            arr.append(closeData)
            if i > (int(np.floor(len(stock) * .95))):
                arry = scaler.fit_transform(arr[-Nin:])
                arry = np.reshape(arry, (1, 1, len(arry)))
                predict = model.predict(arry)
                # invert predictions
                arry = np.reshape(arry, (1, Nin))
                arry = scaler.inverse_transform(arry)
                # predict = scaler.inverse_transform(predict)
                predict = predict[0][0]
                predictArray.append(predict)
                if predict > 1 or predict < 0: errorCnt += 1
                # kar.append(predict)
                if i < len(stock) - 1:
                    difference = arry[0][Nin - 1] - stock[i + 1]
                    # print("arry_diff:", difference)
                    # print("predict:", predict)
                    if stock[i + 1] > arry[0][-1] and (predict > .5):
                        diff.append(1)
                        #print("correct, margin:", predict - .5)
                        correct_margin_array.append(predict - .5)
                    else:
                        diff.append(0)
                        #print("incorrect, margin:", predict - .5)
                        false_margin_array.append(predict - .5)
                # if i > 100:
                # plot(trainPredict)

                # print("predicted:", kar)
                # calculate root mean squared error

        # print("errors:", diff)
        print("tik:", tik, "\n",
              "Nin:", Nin, "\n",
              "numEpoch:", numEpoch, "\n",
              "numBatch:", numBatch, "\n",
              "opt:", opt, "\n",
              "err:", err)
        if errorCnt > 0: print("tengo", errorCnt, "problemas, avg prediction:", np.mean(predict))
        print("mean inverse error (% correct):", np.mean(diff))
        print("mean correct margin:", np.mean(correct_margin_array))
        print("mean error margin:", np.mean(false_margin_array))
        print("\n\n")
        # print("error kurtosis", scipy.stats.kurtosis(diff, fisher=True))
        # print("error variance", np.var(diff))

        if np.mean(diff) > 0.3:
            write_that_shit(log[j], tik, Nin, numEpoch, numBatch, opt, err, diff)


    return cuml

ticker = ["BTC_ETH", "BTC_XMR", "BTC_DASH", "BTC_XRP", "BTC_FCT", "BTC_ZEC", "BTC_LTC"]
fileTicker = []
fileOutput = []
fileCuml = []
dataset = []
for i, tick in enumerate(ticker):
    fileTicker.append("../../data/" + tick + ".txt")
    fileOutput.append("../../output/" + tick + "_mlOutput.txt")
for i, file in enumerate(fileTicker):
    if (os.path.isfile(file) == False):
        fileWrite = open(file, 'w')
        dataset = CryptoQuote1(ticker[i])
        O = dataset.open
        H = dataset.high
        L = dataset.low
        C = dataset.close
        V = dataset.volume
        # tick = yahoo_finance.Share(ticker[i]).get_historical('2015-01-02', '2017-01-01')
        # dataset = np.zeros(len(tick))
        # i = len(tick) - 1
        # while i >= 0:
        #     ik = 0
        #     dataset[ik] = tick[i]['Close']
        #     i -= 1
        #     ik += 1
        for i in range(len(C)):
            fileWrite.write(str(O[i]))
            fileWrite.write('\n')
            fileWrite.write(str(H[i]))
            fileWrite.write('\n')
            fileWrite.write(str(L[i]))
            fileWrite.write('\n')
            fileWrite.write(str(C[i]))
            fileWrite.write('\n')
            fileWrite.write(str(V[i]))
            fileWrite.write('\n')

opts = ['Adam', 'Adadelta', 'RMSprop', 'Adagrad', 'Adamax', 'Nadam', 'TFOptimizer']
#errs = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error']
errs = ['binary_crossentropy']
nins = np.arange(5, 100, step=5)
batchs = np.arange(5, 50, step=5)
epochs = np.arange(10, 500, step=10)

for i in range(len(errs)):
    for j in range(len(batchs)):
        for k in range(len(epochs)):
            for l in range(len(opts)):
                for m in range(len(nins)):
                    fucking_peter(fileTicker, nins[m], errs[i], opts[l], fileOutput, epochs[k], batchs[j])

#fucking_peter(fileTicker, 75, 'mean_absolute_error', 'sgd', fileOutput, fileCuml, 30, 10)
