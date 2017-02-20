import pandas
import yahoo_finance
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import scipy.stats as sp
import urllib, time, datetime
#import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math


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

    def __init__(self, symbol, interval_seconds=2, num_days=10):
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

def ft_describe(a):
    dis = sp.describe(a)
    print("range:", dis[1])
    print("mean:", dis[2])
    print("variance:", dis[3])
    print("skewness", dis[4])
    print("kurtosis:", dis[5])
    print("scipy fisher kurt:", sp.kurtosis(a, fisher=True))
    print("scipy non-fisher kurt:", sp.kurtosis(a, fisher=False))
    print("numpy stdDev ddof=1:", np.std(a, ddof=1))

def SMAn(a, n): 
    s = np.zeros(len(a))
    si = 0
    for i in range(len(a)):
        for k in range(n):
            if i - k >= 0: si += a[i - k]
        if i >= n : si /= n
        elif i < n : si = a[i]
        s[i] = si
        si = 0
    return s

def EMAn(a, n):
    l = len(a)
    e = np.zeros(l)
    m = 2 / (n + 1)
    for i in range(l):
        if i < n:
            e[i] = a[i]
        else:
            y = (a[i - 1] * m) + (a[i - 2] * (1 - m))
            e[i] = (a[i] * m) + (y * (1 - m))
    return e

def stochK(a, ll):
    K = np.zeros(len(a))
    i = 1
    while i < len(a):
        if i > ll : #ALL ll'S ARE STOCH K INPUT VAL
            cpy = a[i-ll:i + 1]
            h = max(cpy)
            l = min(cpy)
        else :
            cpy = a[0:i + 1]
            h = max(cpy)
            l = min(cpy)
        Ki = (a[i] - l) / (h - l)
        K[i] = Ki
        i += 1
    return K

def stochD(a, d, ll):
    K = stochK(a, ll)
    D = SMAn(K, d) # d = STOCH D INPUT VAL
    return D

def plot(a):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('Price')
    plt.xlabel('Time Periods')
    plt.show()
    return 0

def plot2(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r')
    plt.ylabel('Price')
    plt.xlabel('Time Periods')
    plt.show()
    return 0

def plot3(a, b, c):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r', y, c, 'b')
    plt.ylabel('Price')
    plt.xlabel('Time Periods')
    plt.show()
    return 0

def plotLin(x_values, y_values): #NONFUNTIONAL, FOR LINEAR REGRESSION TESTING
    body_reg = linear_model.LinearRegression()
    body_reg.fit(x_values, y_values, sample_weight=None)

    plt.scatter(x_values, y_values)
    plt.plot(x_values, body_reg.predict(x_values))
    plt.show()

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

tick = yahoo_finance.Share("GOOG").get_historical('2016-02-02', '2017-01-01')
data = np.zeros(len(tick))
for i in range(len(tick)):
    data[i] = tick[i]['Close']
#data = GoogleIntradayQuote('GOOG').close

np.random.seed(7)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
# reshape into X=t and Y=t+1
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=12, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# plot baseline and predictions
for i in range(10):
    trainPredict = np.insert(trainPredict, 0, data[i], axis=0)
for i in range(10):
    testPredict = np.insert(testPredict, 0, data[len(trainPredict)], axis=0)
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredict)
plt.show()
plt.plot(testPredict, "o")
plt.plot(scaler.inverse_transform(dataset[len(trainPredict):]), "o")
plt.show()







#x = [3,4,2,3,4,2,1] #NONFUNCTIONAL, FOR SEABORN TESTING
#y = [3,4,3,2,3,1,2]
#with sns.axes_style("white"):
    #sns.jointplot(x=x, y=y, kind="hex", color="k")





