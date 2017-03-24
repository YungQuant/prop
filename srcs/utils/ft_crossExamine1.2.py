import numpy as np
import urllib.request
import urllib, time, datetime
import scipy.stats as sp
from matplotlib import pyplot as plt
import os.path
from multiprocessing import Pool
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
    lb1, midb1, ub1 = BBn(arr, Kin, 1, 1)
    lb2, midb2, ub2 = BBn(arr, Kin, 2, 2)
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

def twap(arr, ll):
    a = arr[-ll:]
    high = max(a)
    low = min(a)
    close = a[len(a) - 1]
    return (high + low + close) / 3

def BBmomma(arr, Kin):
    lb, mb, ub = BBn(arr, Kin, 2.5, 2.5)
    srange = ub - lb
    pos = arr[-1] - lb
    if srange > 0:
        return pos/srange
    else:
        return 0.5

def  getNum(str):
    tmp = ""
    for i, l in enumerate(str):
        if l.isnumeric() or l == ".":
            tmp += l
    return float(tmp)
class ohlcvObj():
    open, high, low, close, volume = [],[],[],[],[]

def CryptoQuote1(the_symbol):
    obj = ohlcvObj
    the_url = "https://poloniex.com/public?command=returnChartData&currencyPair={0}&start=1435699200&end=9999999999&period=300".format(the_symbol)
    response = urllib.request.urlopen(the_url).read().decode("utf-8").split(",")
    for i, curr in enumerate(response):
        if curr.find('open') > 0:
            obj.open.append(getNum(curr))
        elif curr.find('high') > 0:
            obj.high.append(getNum(curr))
        elif curr.find('low') > 0:
            obj.low.append(getNum(curr))
        elif curr.find('close') > 0:
            obj.close.append(getNum(curr))
        elif curr.find('volume') > 0:
            obj.volume.append(getNum(curr))
    return obj

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

    def __init__(self, symbol, interval_seconds=60, num_days=10):
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


def fucking_paul(tick, Kin, Din, bitchCunt, tradeCost, tickers):
    cuml = []
    for j, tik in enumerate(tick):
        stock = []
        with open(tik, 'r') as f:
            stock1 = f.readlines()
        f.close()
        for i, stocks in enumerate(stock1):
            stock.append(float(stocks))

        arr = []; buy = []; sell = [];  diff = []; perc = []; desc = [];
        kar = []; dar = []; cumld = []; kar1 = []; dar1 = []; Kvl = np.zeros(2);
        Dvl = Kvl; s1ar = []; s2ar = []; shortDiff = []; cumlArr = [];
        stockBought = False
        stopLoss = False
        bull = 0; shit = 0; max = 0; tot = 0;
        cuml.append(1)

        for i, closeData in enumerate(stock):
            arr.append(closeData)
            scaler = MinMaxScaler(feature_range=(0, 1))
            if i >= int(Din) and i >= int(Kin):
                    Kv = stochK(arr, int(np.floor(Kin)))
                    kar.append(Kv)
                    Dv = SMAn(kar, int(np.floor(Din)))
                    dar.append(Dv)
                    Kv1 = bbK(arr, int(np.floor(Kin)))
                    kar1.append(Kv1)
                    Dv1 = bbD(arr, int(np.floor(Kin)))
                    dar1.append(Dv1)
                    # Kv2 = SMAn(arr, Kin2)
                    # kar2.append(Kv2)
                    # Dv2 = SMAn(arr, Din2)
                    # dar2.append(Dv2)
                    Kvl = [Kv, Kv1]
                    Dvl = [Dv, Dv1]
                    Kvl = scaler.fit_transform(Kvl)
                    Dvl = scaler.fit_transform(Dvl)
                    s1 = (Kvl[0] + Kvl[1]) / 2
                    s2 = (Dvl[0] + Dvl[1]) / 2
                    s1ar.append(s1)
                    s2ar.append(s2)
                    if stockBought == True and closeData > max:
                        max = closeData
                    if ((s1 > s2) and (stockBought == False and stopLoss == False)):
                        buy.append(closeData * (1+tradeCost))
                        bull += 1
                        stockBought = True
                    elif ((s1 < s2) and stockBought == True):
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
                    elif ((s1 < s2) and stopLoss == True):
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
        #plot(perc)
        #plot(cumld)
        # plot2(s1ar, s2ar)
        cumlArr.append(cuml[j])
    for i in range(len(cumlArr)):
        tot += cumlArr[i]

    print(tickers, Kin, Din, bitchCunt, ":", tot/len(cumlArr))

    return cuml

ticker = ["MNKD", "RICE", "FNBC", "RTRX", "PTLA", "EGLT", "OA", "NTP"]
fileTicker = []
fileOutput = []
fileCuml = []
dataset = []
for i, tick in enumerate(ticker):
    fileTicker.append("../../data/" + tick + ".txt")
for i, file in enumerate(fileTicker):
    if (os.path.isfile(file) == False):
        fileWrite = open(file, 'w')
        #dataset = GoogleIntradayQuote(ticker[i]).close
        tick = yahoo_finance.Share(ticker[i]).get_historical('2015-01-02', '2017-01-01')
        dataset = np.zeros(len(tick))
        i = len(tick) - 1
        ik = 0
        while i >= 0:
            dataset[ik] = tick[i]['Close']
            i -= 1
            ik += 1
        for i, close in enumerate(dataset):
            fileWrite.write(str(close))
            fileWrite.write('\n')

#fucking_paul(fileTicker, 10, 30, 15, 40, fileOutput, fileCuml, save_max=1.02, save_min=0.98, max_len=100000, bitchCunt=0.05, tradeCost=0.00)
parameters = [10, 5, 89, 4, 29, 21, 61, 7, 74, 2, 14, 13, 12, 9, 12, 2, 74, 13, 4, 2]
j = 0.0035; j1 = 0; j2 = 0.05; i = 0;

while i <= len(parameters) - 1:
    if i < len(parameters) - 2:
        while (j < j2):
            p1 = parameters[i]
            p2 = parameters[i + 1]
            print(fucking_paul(fileTicker, p1, p2, j, 0.0005, ticker))
            j += 0.01
        j = 0.0035
    i += 2