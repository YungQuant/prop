import numpy as np
import quandl
import zipline
from zipline.protocol import BarData
from zipline import run_algorithm
import sys
import pandas as pd
import time
import urllib.request
from zipline import run_algorithm
from datetime import datetime
import pytz

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

def initialize(context):
    context.security = symbol('AAPL')
    print("boop\n")

def handle_data(context, data):

    kv = data[context.security].mavg(14)                       #stochK(data.current(context.security, fields="close"), 14)
    dv = data[context.security].mavg(28)                       #stochD(data.current(context.security, fields="close"), 3, 14)

    current_price = data[context.security].price
    current_positions = context.portfolio.positions[context.security].amount
    cash = context.portfolio.cash

    if (kv > dv) and current_positions == 0:
        number_of_shares = 100
        order(context.security, number_of_shares)
    elif(kv < dv) and current_positions != 0:
        order_target(context.security, 0)



base_capital = 1000000
start = pd.datetime(2016, 1, 26, 0, 0, 0, 0, pytz.utc)
end = pd.datetime(2016, 1, 27, 0, 0, 0, 0, pytz.utc)
pdTime = pd.Timestamp(start)
from zipline.data.bundles import ingest
print("boop0\n")
#ingest('quantopian-quandl', timestamp=pdTime, show_progress=True)
print("boop1\n")

perf = run_algorithm(start, end, initialize, base_capital, handle_data, data=GoogleIntradayQuote('APPL'))

print(perf)