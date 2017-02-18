##################################################################
#                                                                #
#   ch1.py                                                       #
#   author:     Logan Brentzel                                   #
#   co-author:  Duncan Lemp                                      #
#   Captain Hindsight!                                           #
#                                                                #
#               Created:      30 Jan 2017                        #
#               Last Updated: 30 Jan 2017                        #
#                                                                #
##################################################################

import numpy as np
import zipline
from zipline.api import record, order, symbol
import sys
import pandas as pd
import time
import urllib.request
import urllib, time, datetime


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

    def __init__(self, symbol, interval_seconds=60, num_days=1):
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

def SMAn(a, n):                         #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

def stochK(a, ll): #GETS STOCHK VALUE OF "LL" PERIODS FROM "A" ARRAY
    cpy = a[-ll:]
    h = max(cpy)
    l = min(cpy)
    Ki = (cpy[len(cpy) - 1] - l) / (h - l)
    return Ki

def stoch_K(a, ll):
    K = np.zeros(len(a))
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
        Ki = (a[i] - l) / (h - l)
        K[i] = Ki
        i += 1
    return K

def stochD(a, d, ll):
    K = stoch_K(a, ll)
    D = SMAn(K, d) # d = STOCH D INPUT VAL, ll = STOCH K INPUT VAL
    return D


def rsiN(a, n):                         #GETS RSI (Relative Strength Index) VALUE FROM "N" PERIODS OF "A" ARRAY
    cpy = a[-n:]
    l = len(cpy)
    lc, gc, la, ga = 0
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

def cappy(Kv, Dv, state):               #captain hindsight strikes again!

    if ((Kv >= Dv) and state <= 0):
        return 1
    elif ((Kv <= Dv) and state >= 0):
        return -1
    return state

def posSize():                          #currently static at 100 shares
    return 100

            ###DO NOT TRADE UNTIL ENOUGH DATA IS AGGREGATED###

# def sexTape(stock, max, mean, min, drawdown, profit, tradNum, tradFreq):
#     f = open('Log', 'w')
#
#     f.write('Stock: ', stock)
#     f.write('Max Profit:\t', max)
#     f.write('Mean Profit:\t', mean)
#     f.write('Min Profit:\t', min)
#     f.write('Drawdown:\t', drawdown)
#     f.write('Profit:\t', profit)
#     f.write('Number of trades:\t', tradNum)
#     f.write('Frequency of trades:\t', tradFreq)
#     f.write('\n')


        #Dirty Whore Global Variables#
state = 0
start_price = 1
current_price = 1
ordered = False
shares = posSize()
        #Big Daddy Global Variables#
maxVal = -sys.maxsize
minVal = sys.maxsize
mean = 0
numTrad = 0
freqTrad = 0
profit = 0
drawdown = 0

        #handle_data Global Variables#
stock = ""
din = 3
kin = 14

def initialize():
    #Big Daddy#
    global mean
    mean = 0
    global numTrad
    numTrad = 0
    global freqTrad
    freqTrad = 0
    global profit
    profit = 0
    global drawdown
    drawdown = 0

def before_trading_start():
    global state
    state = 0
    global start_price
    start_price = 1
    global current_price
    current_price = 1
    global ordered
    ordered = False
    global shares
    shares = posSize()

def dirtyWhore(stock, valArray, kin, din):
    stK = stochK(valArray, kin)
    stD = stochD(valArray, din, kin)
    global state, ordered, start_price, current_price, shares
    profit_min = start_price + (start_price * 0.07)

    if state <= 0:
        state = cappy(stK, stD, state)
    if state > 0 and ordered == False:
        start_price = valArray[len(valArray)-1]
        order(symbol('MNKD'), 100)
        #LIMIT_PRICE SHOULD BE 0.7% ABOVE CURRENT PRICE, STOP_PRICE IS TBD
        ordered = True
    elif (state > 0 or ((current_price / start_price) < profit_min)):
        state = cappy(stK, stD, state)
        current_price = valArray[len(valArray)-1]
    elif state < 0 and ordered == True:
        order(symbol('MNKD'), 100)
        ordered = False
    return current_price - start_price

def handle_data(stock, BarData, Kin, Din):
        initialize()
        before_trading_start()
        bigDaddy(stock, BarData, kin, din)

def bigDaddy(stock, data, kin, din):
    valArray = data.close

    tempProfit = dirtyWhore(stock, valArray, kin, din)
    global numTrad
    numTrad += 1
    global profit
    profit += tempProfit
    if maxVal < tempProfit:
        global maxVal
        maxVal= tempProfit
    if minVal > tempProfit:
        global minVal
        minVal = tempProfit
    mean += tempProfit
    if profit < 0:
        global drawdown
        drawdown += tempProfit
    global mean
    mean = mean / numTrad
    return


mnkd = GoogleIntradayQuote("MNKD")
handle_data('MNKD', mnkd, 14, 3)











