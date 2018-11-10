import ccxt
import sklearn as sk
import requests, pandas as pan
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings(action="ignore", module="sklearn", message="^internal gelsd")

exchange = ccxt.bitmex()
''' TESTNET BITMEX 0'''
exchange.apiKey = 'XisZrCcq4qDmLhs2T53MpFMJ'
exchange.secret = 'ApEVapMkwqiEhV3FGr7NPjZHPG7P8xPAbtHxrbEs5RxA8NDz'


def load_bitmex_data(symbol, time_frequency, n_periods=500, baseURI='https://www.bitmex.com/api/v1'):
    '''
    request = requests.get('https://www.bitmex.com/api/v1/trade/bucketed?binSize=1m&partial=false&symbol=xbt&count=500&reverse=true')

    '''
    endpoint = '/trade/bucketed/'
    if time_frequency == 'half_hour':
        bin_size = '5m'
    elif time_frequency == '1m':
        bin_size = '1m'
    elif time_frequency == 'bi_hourly' or time_frequency == 'hourly':
        bin_size = '1h'
    elif time_frequency == 'daily':
        bin_size = '1d'

    request = requests.get(baseURI + endpoint, params={'symbol': symbol,
                                                       'binSize': bin_size,
                                                       'count': n_periods,
                                                       'reverse': False})

    raw_data, keys = request.json(), request.json()[0].keys()
    reformatted_data = pan.DataFrame()

    for i in range(0, len(request.json())):
        appended_data = pan.DataFrame([raw_data[i][key] for key in keys]).T
        reformatted_data = pan.concat([reformatted_data, appended_data])

    reformatted_data = reformatted_data.reset_index(drop='index')
    reformatted_data.columns = keys
    if time_frequency == 'half_hour':
        acceptable_units = ['30', '00']
        index = [i for i in range(0, len(reformatted_data)) if
                 reformatted_data['timestamp'][i][14:16] in acceptable_units]
        reformatted_data = reformatted_data.ix[index, :]

    elif time_frequency == 'bi_hourly':
        acceptable_units = ['02', '04', '06', '08', '10', '12', '14', '16', '18', '20']
        index = [i for i in range(0, len(reformatted_data)) if reformatted_data['date'][i][11:13] in acceptable_units]
        reformatted_data = reformatted_data.ix[index, :]

    return reformatted_data[::-1].reset_index(drop='index')

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
    lc, gc, la, ga = 1, 1, 0.01, 0.01
    for i in range(1, l):
        if cpy[i] < cpy[i - 1]:
            lc += 1
            la += cpy[i - 1] - cpy[i]
        if cpy[i] > cpy[i - 1]:
            gc += 1
            ga += cpy[i] - cpy[i - 1]
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

#TIME WEIGHTED AVERAGE PRICE
def twap(arr, ll):
    a = arr[-ll:]
    high = max(a)
    low = min(a)
    close = a[len(a) - 1]
    return (high + low + close) / 3

def BBmomma(arr, Kin):
    lb, mb, ub = BBn(arr, Kin, 3, 3)
    srange = ub - lb
    pos = arr[-1] - lb
    if srange > 0:
        return pos/srange
    else:
        return 0.5

def isFloat(n):
    try:
        n = float(n)
        return True
    except:
        return False


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)):
        if i > look_back:
            dataX.append(dataset[i-look_back:i])
            dataY.append(dataset[i])
    return np.array(dataX), np.array(dataY)

def create_binary_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)):
        if i > look_back:
            dataX.append(dataset[i-look_back:i])
            if dataset[i] > dataset[i-1]:
                dataY.append(1)
            else:
                dataY.append(0)
    return np.array(dataX), np.array(dataY)

def create_bounded_binary_dataset(dataset, look_back, b=0.01):
    dataX, dataY = [], []
    for i in range(len(dataset)-1):
        if i > look_back:
            dataX.append(dataset[i-look_back:i])
            if dataset[i] > dataset[i-1] * (1 + b):
                dataY.append(1)
            elif dataset[i] < dataset[i-1] * (1 - b):
                dataY.append(-1)
            else:
                dataY.append(0)
    return np.array(dataX), np.array(dataY)

def get_CDD_data(currency, interval='1h'):
    filename = f'../../cDDdata/{currency}_{interval}.csv'
    data = []
    if os.path.isfile(filename) == False:
        print(f'could not source ../../cDDdata/{currency}_{interval}.csv data')
    else:
        fileP = open(filename, "r")
        lines = fileP.readlines()
        for i, line in enumerate(lines):
            linex = line.split(",")[2:6]
            data.append(linex)
    return data[2:]

#
# currency_pairs, currencies = ['LTCU18'], ['ltc']
# errs = []
#
# #print(exchange.create_order(currency_pairs[0], ))
# #orders = exchange.fetch_ohlcv(symbol=currency_pairs[0], timeframe='1m')
# orders = load_bitmex_data(currency_pairs[0], "hourly", n_periods=500)
# #print(orders)
# #ps = [order[-2] for order in orders if isFloat(order[-2])]
# ps = orders.ix[:, 'close']
# print(ps)
# X, Y = create_dataset(ps, 10)
#
# trainX, trainY, testX, testY = X[:int(np.floor(len(X)/1.2))], Y[:int(np.floor(len(X)/1.2))], X[int(np.floor(len(X)/1.2)):], Y[int(np.floor(len(X)/1.2)):]

''' VVV 1H / 10 WINDOW / LTCU18 CLOSE PRICES / REG API VVV '''
# model = linear_model.BayesianRidge() # Mean % Error: 0.02025354214895091
# model = linear_model.Ridge() # Mean % Error: 0.050484925340484356
# model = linear_model.HuberRegressor() # Mean % Error: 0.014063285819013012
# model = linear_model.Lars() # Mean % Error: 0.010000416749448777
# model = linear_model.Lasso() # Mean % Error: 0.05048526961697335
# model = linear_model.LinearRegression() # Mean % Error: 0.010073936285900685
# model = linear_model.ARDRegression() # Mean % Error: 0.01382198958869686
# model = linear_model.RANSACRegressor() # Mean % Error: 0.008092937591752074

''' VVV 1H / 10 WINDOW / LTCU18 CLOSE PRICES / RB API (500) VVV '''
# model = linear_model.LinearRegression() # Mean % Error: 0.005631468683177288
# model = linear_model.BayesianRidge() # Mean % Error: 0.005703020058904807
# model = linear_model.Ridge() # Mean % Error: 0.053695055696198084
# model = linear_model.HuberRegressor() # Mean % Error: 0.006980393898696223
# model = linear_model.Lars() # Mean % Error: 0.0055651230021880665
# model = linear_model.Lasso() # Mean % Error: 0.05370095606086554
# model = linear_model.ARDRegression() # Mean % Error: 0.005545638051673693
# model = linear_model.RANSACRegressor() # Mean % Error: 0.00553197478063615

#model = linear_model.Ridge()
#model.fit(trainX, trainY)

# for i in range(len(testX)):
#     sTXi = np.reshape(testX[i], [1, -1])
#     pY, rY = model.predict(sTXi), testY[i]
#     errP = abs(pY - rY) / rY
#     errs.append(errP)
#     print("sTXi:", sTXi)
#     print("pY:", pY, "rY:", rY, "err %:", errP)
#
# print("\n\nMean % Error:", np.mean(errs))



def write_that_shit(log, kin, din,  perc, cuml, mdd, bitchCunt):
    if os.path.isfile(log):
        th = 'a'
    else:
        th = 'w'
    file = open(log, th)
    # file.write("Tick:\t")
    # file.write(tick)
    file.write("\nK in:\t")
    file.write(str(int(np.floor(kin))))
    file.write("\nD in:\t")
    file.write(str(din))
    file.write("\nbitchCunt:\t")
    file.write(str(bitchCunt))
    # file.write("\nK1 in:\t")
    # file.write(str(int(np.floor(kin1))))
    # file.write("\nD1 in:\t")
    # file.write(str(int(np.floor(din1))))
    # file.write("\nK2 in:\t")
    # file.write(str(int(np.floor(kin2))))
    # file.write("\nD2 in:\t")
    # file.write(str(int(np.floor(din2))))
    file.write("\nLen:\t")
    file.write(str(len(perc)))
    # file.write("\n\n\nPercent Diff:\n")
    # file.write(str(perc))
    # if len(perc) > 10:
    #     desc = sp.describe(perc)
    #     file.write("\n\nDescribed Diff:\n")
    #     file.write(str(desc))
    file.write("\nMDD:\t")
    file.write(str(mdd))
    file.write("\n [elastic, static bitchcunt distance] Cumulative Diff:\t")
    file.write(str(cuml))
    file.write("\n\n")
    file.close()
    # print("Described diff")
    # print(desc)
    # print("Cumulative Diff")
    # print("len:", len(perc))
    # print(cuml[j])


def fucking_paul(stock, log, Kin, Din, save_max, max_len, bitchCunt, tradeCost):
    arr = []; buy = []; sell = [];  diff = []; perc = []; cumld = []
    kar = []; dar = []; cumld = []; kar1 = []; dar1 = []; Kvl = np.zeros(2)
    Dvl = Kvl; s1ar = []; s2ar = []; shortDiff = []; cuml = 1.0; mdd = 0
    position = 0
    stopLoss = False
    bull = 0; shit = 0; maxP = 0; minP = 0;

    for i, closeData in enumerate(stock):
        model = linear_model.ElasticNet(alpha=29.91)
        arr.append(closeData)
        trainX, trainY = create_dataset(arr, Din)
        if i > Kin:
            rarr = np.reshape(arr[-Din:], [1, -1])
            model.fit(trainX, trainY)
            p = model.predict(rarr)
            if ((p > closeData) and position == -1 or position == 0):
                buy.append(closeData * (1+tradeCost))
                bull += 1
                position = 1
            elif (p < closeData) and position == 1 or position == 0:
                sell.append(closeData * (1 - tradeCost))
                maxP = 0
                shit += 1
                position = -1
            if position == 1 and closeData > maxP:
                maxP = closeData
            elif position == -1 or position == 0:
                maxP = closeData
            if position == -1 and closeData < minP:
                minP = closeData
            elif position == 1 or position == 0:
                minP = closeData
            #DYNAMIC BITCHCUNT DISTANCE IN DEVELOPMENT
            #WILL BE BASED ON ANALYSIS OF VARIANCE, AND CORRELATION WITH ENVIRONMENTAL VOLITILITY
            if (closeData < (maxP * (1-bitchCunt)) and position == 1):
                sell.append(closeData * (1-tradeCost))
                maxP = 0
                shit += 1
                position = -1
            if (closeData > (minP * (1 + bitchCunt)) and position == -1):
                buy.append(closeData * (1+tradeCost))
                shit += 1
                position = 1
    if position == 1:
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
        cumld.append(cuml)
        cuml += cuml * perc[i]

    for i in range(len(cumld)):
        if i > 1:
            peak = max(cumld[:i])
            trough = min(cumld[i:])
            dd = (peak - trough) / peak
            if dd > mdd:
                mdd = dd

    print("tik:", log, "cuml:", cuml)

    if cuml > save_max and len(perc) <= max_len:
        write_that_shit(log, Kin, Din, perc, cuml, mdd, bitchCunt)
        print(f'\tYEEEEEE BOIIIIS: Kin: {Kin} Din: {Din} BitchCunt: {bitchCunt} MDD = {mdd} LEN = {len(perc)} CUML = {cuml}')
        # DONT FUCKING MOVE/INDENT WRITE_THAT_SHIT!!!!
        # if mdd < 0.5:
        #     plot(cumld)
        # plot2(s1ar, s2ar)
    return cuml

def pillowcaseAssassination(data, k, i, fileOutput, save_max, max_len, bitchCunt, tradeCost):
    #print("Assasinating, w/pillows")
    n_proc = 1; verbOS = 0; inc = 0
    Parallel(n_jobs=n_proc, verbose=verbOS)(delayed(fucking_paul)
            (data[inc], fileOutput[inc], k, i, save_max, max_len, bitchCunt, tradeCost)
            for inc in range(len(data)))




ticker = ["BTCUSD", "ETHUSD", "LTCBTC", "XRPBTC", "ADABTC"]
#ticker = ["XBTUSD", "ETHUSD", "XRPU18", "TRXU18", "LTCU18", "EOSU18", "ADAU18", "BCHU18", "XRPU18"]
fileTicker = []
fileOutput = []
fileCuml = []
dataset = []
for i, tick in enumerate(ticker):
    #orders = load_bitmex_data(tick, "1m", n_periods=666)
    orders = get_CDD_data(tick)
    print(tick, ":", len(orders))
    print(orders[0])
    ps = [float(order[3]) for order in orders]
    #ps = orders.ix[:, 'close']
    dataset.append(ps)

for i, tick in enumerate(ticker):
    #fileOutput.append(tick)
    fileOutput.append("../../output/elastic_1h_phat" + tick + "_10.19.18_output.txt")
#
# for i, file in enumerate(fileTicker):
#     if (os.path.isfile(file) == False):
#         print("missing:", file)

#fucking_paul(fileTicker, 10, 30, 15, 40, fileOutput, fileCuml, save_max=1.02, save_min=0.98, max_len=100000, bitchCunt=0.05, tradeCost=0.00)


def run():
    k1 = len(dataset[0]) * 0.6; k2 = len(dataset[0]) * 0.61
    l1 = 5; l2 = 15
    # d1 = 2; d2 = 300
    # s1 = 2; s2 = 30
    j1 = 0.01; j2 = 0.15
    k = k1; i = l1; j = j1 #; d = d1; s = s1
    returns = []
    while (k < k2):
        while (i < l2):
            while (j < j2):
                #while (d < d2):
                    #while (s < s2):
                if i > 0:
                    if k % 10 == 0: print(i, "/", l2, int(np.floor(k)), "/", k2, j, "/", j2)
                    pillowcaseAssassination(dataset, k, i, fileOutput, save_max=1.001, max_len=3000000, bitchCunt=j, tradeCost=0.0025)
                    #     if (s < 10):
                    #         s += 1
                    #     else:
                    #         s *= 1.3
                    # s = s1
                #     if (d < 10):
                #         d += 1
                #     else:
                #         d *= 1.3
                # d = d1
                if (j < 0.1):
                    j += 0.01
                else:
                    j *= 1.3
            j = j1
            i += 1
        i = l1
        if (k < 10):
            k += 1
        elif (k < 1000):
            k += len(dataset) / 10
        elif (k < 10000):
            k *= 1.05
        else:
            k *= 1.01

run()

