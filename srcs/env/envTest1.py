import numpy as np
import quandl
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


#ticker = ["BTC_ETH", "BTC_XMR", "BTC_DASH", "BTC_XRP", "BTC_FCT", "BTC_MAID", "BTC_ZEC", "BTC_LTC"]
ticker = ['BCHARTS/BITSTAMPUSD']
fileTicker = []
fileOutput = []
fileCuml = []
dataset = []
for i, tick in enumerate(ticker):
    # fileTicker.append("../../data/" + tick + ".txt")
    # fileOutput.append("../../output/" + tick + "_output.txt")
    fileTicker.append("../../data/" + "BITSTAMP_USD_BTC.txt")
    fileOutput.append("../../output/" + "BITSTAMP_USD_BTC_cip1.1.2L.S.50.50_output.txt")
for i, file in enumerate(fileTicker):
    if (os.path.isfile(file) == False):
        fileWrite = open(file, 'w')
        #dataset = CryptoQuote1(ticker[i]).close
        data = quandl.get(ticker[i], column_index=4, exclude_column_names=True)
        data = np.array(data)
        for i in range(len(data)):
            if float(data[i][-6:]) > 0:
                dataset.append(float(data[i][-6:]))
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