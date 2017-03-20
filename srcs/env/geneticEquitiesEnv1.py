from random import randint
import random
from scipy.fftpack import fft, ifft, rfft, irfft, dct, idct, dst, idst
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

    def __init__(self, symbol, interval_seconds=600, num_days=10):
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

def plotFFT(a, b):
    x = np.arange(len(a))
    #b = b[int(np.floor(min(a))):(int(np.floor(max(a)) + 3))]
    yb = np.arange(min(a), max(a), step=(max(a) - min(a))/len(b))
    plt.plot(x, a, 'g', x, b, 'r')
    plt.ylabel('Price')
    plt.xlabel('Time Periods')
    plt.show()

def BBn(a, n, stddevD, stddevU): #GETS BOLLINGER BANDS OF "N" PERIODS AND STDDEV"UP" OR STDDEV"DOWN" LENGTHS
    cpy = a[-n:] #RETURNS IN FORMAT: LOWER BAND, MIDDLE BAND, LOWER BAND
    midb = SMAn(a, n) #CALLS SMAn
    std = np.std(cpy)
    ub = midb + (std * stddevU)
    lb = midb - (std * stddevD)
    return lb, midb, ub

def SMAn(a, n):                         #GETS SIMPLE MOVING AVERAGE OF "N" PERIODS FROM "A" ARRAY
    si = 0
    if (len(a) < n):
        n = len(a)
    n = int(np.floor(n))
    for k in range(n):
        si += a[(len(a) - 1) - k]
    si /= n
    return si

def get_yahoo(tick):
    dataset = []
    with open(tick, 'r') as dick:
        data = dick.readlines()
    dick.close()
    for i, stocks in enumerate(data):
        dataset.append(float(stocks))
    return dataset

def create_individual(size, total_env):
    return [ total_env[randint(0, len(total_env) -1)] for x in range(size) ]

def create_family(count, size, total_env):
    return [ create_individual(size, total_env) for x in range(count) ]

def check_range_breaks(data):
    arr = []; breaks = 0;
    for i in range(len(data) - 1):
        arr.append(data[i])
        lb, mb, ub = BBn(arr, 30, 3, 3)
        if data[i + 1] > ub or data[i + 1] < lb:
            breaks += 1
    return breaks

def assess_fitness(indv):
    data = []; tot = 0;
    for i in range(len(indv) -1):
        data = get_yahoo(indv[i])
        tot += check_range_breaks(data)
    return tot

def assess_family(fam):
    return sum([assess_fitness(fam[i]) for i in range(len(fam))]) / len(fam)

def assess_res(res):
    for i in range(len(res)):
        res[i].append(assess_family(res[i]))


def epigenetics(parent):
    avg_fit = assess_family(parent[:-1])
    gene_mod = []
    for i in range(len(parent) -1):
        if assess_fitness(parent[i]) > avg_fit:
            gene_mod.append(parent[i])
    return gene_mod

def evolve(env, population, prob_retain, entropy):

    for i, fam in enumerate(population):
        fam.append(assess_family(fam))
    population.sort(key=lambda x: x[-1])
    parents = population[:int(np.floor(prob_retain * len(population)))]

    for i in range(len(parents)):
        for k in range(len(parents[i]) -1):
            for j in range(len(parents[i][k])):
                if random.uniform(0, 1) < entropy:
                    parents[i][k][j] = env[randint(0, len(env) - 1)]

    new_pop = []; desired_len = len(population);
    while len(new_pop) < desired_len:
        p1 = parents[randint(0, len(parents) -1)]; p2 = parents[randint(0, len(parents) -1)];
        print("p1, p2", p1, p2)
        new_fam = epigenetics(p1) + epigenetics(p2)
        print("new fam:", new_fam)
        new_pop.append(new_fam)
    return new_pop

def evolution(env, len_pop, len_fam, len_indv, prob_retain, entropy, len_time):
    pop = []
    for i in range(len_pop):
        pop.append(create_family(len_fam, len_indv, env))
    pop[0] = create_family(len_fam, len_indv, env)

    for i in range(len_time):
        pop = evolve(env, pop, prob_retain, entropy)
        print("popopopop")

    for i, fam in enumerate(pop):
        fam.append(assess_family(fam))
    pop.sort(key=lambda x: x[-1])

    return pop[len(pop)]

environment = ["MNKD", "RICE", "FNBC", "RTRX", "PTLA", "EGLT", "OA", "NTP"]
fileTicker = []; fileOutput = [];
for i, tick in enumerate(environment):
    fileTicker.append("../../data/" + tick + ".txt")
    fileOutput.append("../../output/" + tick + "_output.txt")

for i, file in enumerate(fileTicker):
    if (os.path.isfile(file) == False):
        fileWrite = open(file, 'w')
        #dataset = GoogleIntradayQuote(ticker[i]).close
        tick = yahoo_finance.Share(environment[i]).get_historical('2015-01-02', '2017-01-01')
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

res = evolution(fileTicker, 10, 4, 4, 0.2, 0.1, 100)

print(res, assess_res(res))

