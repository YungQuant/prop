import urllib.request
import urllib, time, datetime
import numpy as np
import timeit
import yahoo_finance
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

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

    def __init__(self, symbol, interval_seconds=60, num_days=12):
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

def plot(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'r', b, 'g')
    plt.ylabel('Volume')
    plt.xlabel('Time Periods')
    plt.title('This would tell people what the fuck this graph is')
    green = mpatches.Patch(color='green', label='This tells you what the fuck the line is')
    plt.legend(handles=[green])
    plt.show()

def sma(data, peroid):
    temp = 0
    for x in range(len(data) - peroid):
        temp = 0
        for i in range(peroid):
            temp += data[x + i]
        data[x] = temp / peroid
    return (data)

def train(data, close):
    index = 0
    winner = 0

    for i in range(10, 500):
        hit = 0
        count = 0
        answer = sma(data, i)
        for x in range(len(answer)):
            if answer[x] > close[x]:
                hit += 1
            count += 1
        total = hit / count
        if total > winner:
            winner = total
            index = i
    return index

def chartandtest(data, close):
    total = 0
    answer = 0
    count = 0
    for i in range(len(data)):
        if data[i] > close[i]:
            count += 1
        total += 1
    answer = count / total
    print(answer)
    plot(data, close)

tick = "SPY"
google = []

close = GoogleIntradayQuote(tick).close
high = GoogleIntradayQuote(tick).high
low = GoogleIntradayQuote(tick).low
open = GoogleIntradayQuote(tick).open_

for i in range(len(close)):
    google.append((open[i] + high[i] + close[i] + low[i]) / 4)

start = timeit.default_timer()


#stuff = train(google, close)

chartandtest(sma(google, 150), close)

stop = timeit.default_timer()

print("------Time--------")
print(stop - start)

