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

def plot(a): #PLOTTING FUNCTION FOR ONE DATASET
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('Price')
    plt.xlabel('Time Periods')
    plt.show()

def plot2(a, b): #PLOTTING FUNCTION FOR 2 DATASETS
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

def rsiN(a, n): #GETS RELATIVE STRENGTH INDEX VALUE FROM "N" PERIODS OF "A" ARRAY
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

def SMAn(a, n):  #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    if (len(a) < n):
        n = len(a)
    n = int(np.floor(n))
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

def stochK(a, ll): #GETS STOCH OSCILLATOR K VALUE OF "LL" PERIODS FROM "A" ARRAY
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

def stoch_K(a, ll): #RETURNS ARRAY OF STOCHASTIC OSCILLATOR K VALUE DATA FROM "ll" PERIODS OF "a" ARRAY
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

def stochD(a, d, ll): #GETS STOCH OSCILLATOR D VALUE OF "LL" PERIODS FROM "A" ARRAY USING STOCHK W/ "ll" LOOKBACK PERIOD
    d = int(np.floor(d))
    ll = int(np.floor(ll))
    K = stoch_K(a, ll)
    D = SMAn(K, d)
    return D
def BBn(a, n, stddevD, stddevU): #GETS BOLLINGER BANDS OF "N" PERIODS AND STDDEV "UP" OR STDDEV "DOWN" LENGTHS
    cpy = a[-n:] #RETURNS IN FORMAT: LOWER BAND, MIDDLE BAND, LOWER BAND
    midb = SMAn(a, n)
    std = np.std(cpy)
    ub = midb + (std * stddevU)
    lb = midb - (std * stddevD)
    return lb, midb, ub

def BBmomma(arr, Kin): #BOLLINGER BAND BASED MOMENTUM INDICATOR
    lb, mb, ub = BBn(arr, Kin, 2.5, 2.5)
    srange = ub - lb
    pos = arr[-1] - lb
    if srange > 0:
        return pos/srange
    else:
        return 0.5

def twap(arr, ll): #TIME WEIGHTED AVERAGE PRICE
    a = arr[-ll:]
    high = max(a)
    low = min(a)
    close = a[len(a) - 1]
    return (high + low + close) / 3

def  getNum(str): #USED IN CRYPTOQUOTE
    tmp = ""
    for i, l in enumerate(str):
        if l.isnumeric() or l == ".":
            tmp += l
    return float(tmp)
class ohlcvObj(): #CLASS TYPE FOR OPEN,HIGH,LOW,CLOSE DATA ARRAYS
    open, high, low, close, volume = [],[],[],[],[]

def CryptoQuote1(the_symbol): #GETS HISTORICAL PRICE DATA FROM CRYPTOCURRENCY PAIRS IN 3 MINUTE INTERVALS
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

def write_that_shit(log, tick, kin, din, perc, cuml, bitchCunt):
    #SAVES BACKTESTS RESULTS TO LOG FILE
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


def fucking_paul(tik, log, Kin, Din, save_max, max_len, bitchCunt, tradeCost):
    #MASTER BACKTEST FUNCTION
    stock = []
    with open(tik, 'r') as f:
        stock1 = f.readlines()
    f.close()
    for i, stocks in enumerate(stock1):
        stock.append(float(stocks))
    #^CONVERTS DATA FILE FROM CRYPTOQUOTE INTO ARRAY

    arr = []; buy = []; sell = [];  diff = []; perc = []; desc = []
    kar = []; dar = []; cumld = []; kar1 = []; dar1 = []; Kvl = np.zeros(2)
    Dvl = Kvl; s1ar = []; s2ar = []; shortDiff = []; cuml = 0.0
    stockBought = False
    stopLoss = False
    bull = 0; shit = 0; max = 0;
    #^VARIABLE INITIALIZATION

    for i, closeData in enumerate(stock): #LOOPS THROUGH PRICE DATA
        arr.append(closeData) #FILLS ARRAY WITH DATA THE ALGORITHM WOULDVE HAD IF IT WAS TRADING LIVE AT THE TIME
        if i >= int(Din) and i >= int(Kin):
            Kv = rsiN(arr, int(np.floor(Kin)))
            kar.append(Kv)
            Dv = SMAn(kar, int(np.floor(Din)))
            #^LOGIC SECTION WHICH HAS THE CHOSEN FORMULAS TO TEST, THE RESULTS OF THE FORMULAS ARE USED TO MAKE TRADING
            #DECSIONS (IE; IF THE 10 PERIOD SMA IS OVER THE 20 PERIOD SMA THEN BUY, IF IT CROSSES BACK BELOW THEN SELL
            print(Kv, Dv)
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
            #^MAKES AND SAVES ALL OF THE TRADING DECSISONS WITH COMMISIONS COSTS, HANDLES TRAILING STOPLOSS
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
    #^CALCULATES HOW MUCH THE ALGORITHM WOULDVE MADE BY PASSING A "1" FLOAT THROUGH THE ARRAY OF PERCENT DIFFERENCES SAVED

    print("len:", len(perc))
    if cuml > save_max and len(perc) <= max_len:
        write_that_shit(log, tik, Kin, Din, perc, cuml, bitchCunt)
    #^SAVES THE PARAMETERS USED FOR THE ALGORITHM IF IT WAS OVER "SAVE_MAX" WHICH IS AN ARBITRARY MINIMUM PROFITABLITY
# DONT FUCKING MOVE/INDENT WRITE_THAT_SHIT!!!!
    # plot(perc)
    # plot2(s1ar, s2ar)
    return cuml
    #^RETURNS "CUML" WHICH IS THE AMOUNT THE ALGORITHM WOULDVE MADE IN PERCENT FORMAT
    #IE; IF CUML = 1.12 THEN THE ALGORITHM MADE 12%

def pillowcaseAssassination(fileTicker, k, i, fileOutput, save_max, max_len, bitchCunt, tradeCost):
    n_proc = 8
    verbOS = 10
    inc = 0
    Parallel(n_jobs=n_proc, verbose=verbOS)(delayed(fucking_paul)
            (fileTicker[inc], fileOutput[inc], k, i, save_max, max_len, bitchCunt, tradeCost)
            for inc, file in enumerate(fileTicker))
    #^HANDLES THREADING (FOR COMPUTATIONALLY INTENSIVE BACKTESTS)


ticker = ["BTC_ETH", "BTC_XMR", "BTC_DASH", "BTC_XRP", "BTC_FCT", "BTC_MAID", "BTC_ZEC", "BTC_LTC"]
#^CRYPTOCURRENCY EXCHANGE PAIRS
fileTicker = []
fileOutput = []
fileCuml = []
dataset = []
for i, tick in enumerate(ticker):
    fileTicker.append("../../data/" + tick + ".txt")
    fileOutput.append("../../output/" + tick + "_output.txt")
    #^MAKES DATA FILES FOR PRICE DATA AND OUTPUT FILES FOR SUCCESSFUL BACKTESTS
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
    #^FILLS DATA FILES WITH DATA FROM CRYPTOQUOTE

#fucking_paul(fileTicker, 10, 30, 15, 40, fileOutput, fileCuml, save_max=1.02, save_min=0.98, max_len=100000, bitchCunt=0.05, tradeCost=0.00)


def run(): #TESTS ALL REASONABLE POSSIBLE PARAMETER COMBINATIONS FOR THE BACKTESTS FORMULA
    k1 = 3
    k2 = 300
    l1 = 2
    l2 = 30
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
                    pillowcaseAssassination(fileTicker, k, i, fileOutput, save_max=1.01, max_len=20000, bitchCunt=j, tradeCost=0.0005)
                    #"FILETICKER"=DATA FILES, "K" AND "I"=PARAMETERS FOR THE DECISION MAKING FORMULAS, "FILEOUTPUT"=FILES WHERE SUCCESSFUL BACKTEST PARAMETERS ARE SAVED
                    #"SAVE_MAX"=ARBITRARY MINIMUM PROFITABLITY FOR PICKING WHICH BACKTESTS TO SAVE, "MAX_LEN"=MAXIMUM AMOUNT OF TRADES ALLOWED FOR A BACKTEST TO BE SAVED
                    #"BITCHCUNT"=TRAILING STOP DISTANCE IN PERCENT (0.01=1%), "TRADECOST"=SIMULATED COMMISIONS COST PER TRADE IN PERCENT (0.01=1%)
                if (j < 0.01):
                    j += 0.0035
                else:
                    j *= 1.3
            j = j1
            if (i < 10):
                i += 1
            else:
                i *= 1.3
        i = l1
        if (k < 10):
            k += 1
        elif (k < 1000):
            k *= 1.2
        elif (k < 10000):
            k *= 1.05
        else:
            k *= 1.01
        #^ITERATES MULTIPLICATIVELY TO SPEED UP TESTING TIME

run()
